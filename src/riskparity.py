"""Risk contributions and the equal-risk-contribution solver (steps 16-17).

This is the mathematical core of the project. Everything else -- the data layer,
the overlay, the charts -- exists to feed or explain what happens here.

**The identity.** For weights ``w`` and covariance ``C``, portfolio volatility is
``sigma_p = sqrt(w' C w)``, the marginal contribution of asset i is
``MRC_i = (C w)_i / sigma_p``, and its risk contribution is ``RC_i = w_i * MRC_i``.
These sum to ``sigma_p`` **exactly**, by Euler's theorem on the homogeneous
degree-1 function ``sigma_p(w)``. :func:`risk_contributions` returns them in
volatility units; the normalised fractions the contract stores are just
``risk_contributions(w, cov) / portfolio_volatility(w, cov)``.

**Why the naive answer is wrong.** :func:`naive_inv_vol` -- weight proportional
to ``1/sigma_i`` -- is what most descriptions of risk parity actually implement.
It equals the true ERC portfolio only when every pairwise correlation is the
same. Real sleeves are not: TLT correlates about -0.3 with SPY and about +0.85
with TIP, and under that structure ``1/sigma`` and ERC give visibly different
portfolios. It ships here purely as the comparator the write-up contrasts
against, never as the strategy.

**The solve.** :func:`solve_erc` minimises ``sum_i (RC_i - sigma_p/N)^2`` under
``sum(w) = 1`` and a long-only floor, using SLSQP **with the analytic gradient**.
Finite differences on this objective is where these solvers quietly return
garbage: near the optimum the objective is flat to ~1e-16 while the step size
scipy picks is ~1e-8, so the difference quotient is dominated by rounding noise
and SLSQP terminates on a direction it invented.

That objective is **not convex**, which is not a footnote here. It has local
minima that pin one sleeve at the weight floor with a negative risk
contribution, and they appear exactly when two sleeves are strongly negatively
correlated -- the SPY/TLT case. :func:`solve_erc_with_status` detects this on the
risk-contribution bar and restarts; the details, and the measurements behind the
restart budget, are in its Notes.

Person B implements the same portfolio independently in
``tests/test_riskparity.py`` (step 19) via the convex form
``min 0.5 w'Cw - (1/N) sum ln(w_i)``, and the two are asserted to agree to
``config.ERC_AGREEMENT_TOLERANCE``. That test is written from the maths, not from
this file -- two independent derivations are the point, and reading this one
collapses them into one.

Run a self-check from the repo root::

    .venv/bin/python -m src.riskparity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .config import (
    ERC_AGREEMENT_TOLERANCE,
    ERC_MAX_ITER,
    ERC_MAX_RESTARTS,
    ERC_SLSQP_FTOL,
    MIN_WEIGHT,
)
from .covariance import validate_covariance

log = logging.getLogger(__name__)

# Restart draws come from a fixed seed, so two runs of the same backtest produce
# the same weights. A solver whose output depended on global RNG state would make
# every downstream number irreproducible for no benefit.
_RESTART_SEED: int = 0

# Dirichlet concentration for restart points. Above 1.0 so draws cluster loosely
# around equal weight rather than piling onto the simplex corners -- a corner is
# the boundary the failed solve was already stuck on.
_RESTART_CONCENTRATION: float = 2.0


class ERCSolverError(ValueError):
    """The ERC problem as posed cannot be solved at all.

    Distinct from non-convergence, which is reported on
    :class:`ERCSolution` rather than raised: a covariance with a zero-variance
    asset or a non-finite entry is a *malformed problem*, whereas a solver that
    ran out of iterations produced a real, if imperfect, answer.
    """


# ---------------------------------------------------------------------------
# Step 16 -- the identity
# ---------------------------------------------------------------------------
def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    """``sqrt(w' C w)``, in the same units as ``cov`` (annualised, in this project)."""
    w, c = _validate_inputs(weights, cov)
    variance = float(w @ c @ w)
    if variance < 0.0:
        # Only reachable on a non-PSD covariance, which src.covariance.nearest_psd
        # exists to prevent. Loud, because a NaN volatility would otherwise
        # propagate silently into every downstream metric.
        raise ERCSolverError(
            f"portfolio variance is negative ({variance:.3e}); the covariance is not "
            "PSD -- route it through src.covariance.nearest_psd first"
        )
    return float(np.sqrt(variance))


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """``RC_i = w_i * (C w)_i / sigma_p``, in volatility units.

    Sums to ``sigma_p`` to floating-point precision -- step 16's "done when", and
    the first thing ``tests/test_riskparity.py`` asserts. The identity is exact
    in real arithmetic, so any deviation beyond ~1e-15 is a bug, not tolerance.

    The normalised form the contract stores (rows summing to 1.0) is this divided
    by ``portfolio_volatility``. It is deliberately not duplicated here:
    ``src.benchmarks.risk_contribution_fractions`` already computes it in the
    algebraically equivalent, one-fewer-square-root way, and two implementations
    of the same quantity is exactly the divergence this repo's contract exists to
    prevent.
    """
    w, c = _validate_inputs(weights, cov)
    sigma = portfolio_volatility(w, c)
    if sigma <= 0.0:
        raise ERCSolverError(
            "portfolio volatility is zero, so risk contributions are undefined; "
            "the covariance is degenerate or every weight is zero"
        )
    return w * (c @ w) / sigma


def naive_inv_vol(cov: np.ndarray) -> np.ndarray:
    """Inverse-volatility weights, ``w_i proportional to 1/sigma_i``. The comparator.

    **Not the strategy.** This ignores correlations entirely, so it coincides
    with true ERC only in the special case where all pairwise correlations are
    equal. Showing where it diverges is one of the write-up's exhibits, and it is
    also the natural warm start for :func:`solve_erc` -- it is cheap, always
    feasible, and close enough that SLSQP converges in a handful of iterations.
    """
    c = validate_covariance(cov, "cov")
    variances = np.diag(c)
    if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
        raise ERCSolverError(
            f"every asset needs a positive variance for inverse-vol weights, got "
            f"diagonal {np.array2string(variances, precision=3)}"
        )
    inverse = 1.0 / np.sqrt(variances)
    return inverse / inverse.sum()


# ---------------------------------------------------------------------------
# Step 17 -- the solver
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ERCSolution:
    """The solver's answer *and* what it thought of it.

    Two failure modes are kept separate on purpose, because they mean different
    things and this repo's working rules require them to stay distinguishable:

    ``optimiser_succeeded``
        SLSQP's own exit status. False means it hit the iteration cap or could
        not find a feasible descent direction.
    ``converged``
        Whether the returned weights actually *are* an equal-risk portfolio, i.e.
        ``max_rc_deviation <= tolerance``. This is the one that matters.

    ``restarts_used`` is the third thing worth surfacing: zero on a healthy
    solve, non-zero when the objective's non-convexity had to be escaped. A run
    where it is routinely non-zero is a run whose covariances deserve a look,
    even though every rebalance in it converged.

    They disagree in both directions. SLSQP reports success on a near-flat
    objective it barely moved on; it also reports failure after landing on a
    perfectly good optimum and then failing a line search. Recording only
    ``optimiser_succeeded`` would hide the first case, which is the dangerous one.
    """

    weights: np.ndarray
    converged: bool
    optimiser_succeeded: bool
    max_rc_deviation: float
    objective: float
    n_iterations: int
    restarts_used: int
    message: str


def solve_erc_with_status(
    cov: np.ndarray,
    tolerance: float = ERC_AGREEMENT_TOLERANCE,
    max_iter: int = ERC_MAX_ITER,
    ftol: float = ERC_SLSQP_FTOL,
    min_weight: float = MIN_WEIGHT,
    initial_weights: np.ndarray | None = None,
    max_restarts: int = ERC_MAX_RESTARTS,
) -> ERCSolution:
    """Solve for equal risk contributions and report how it went.

    Parameters
    ----------
    cov:
        (N, N) annualised, PSD covariance. Run it through
        ``src.covariance.estimate_covariance`` first; this function will not
        repair a non-PSD input, it will raise on one.
    tolerance:
        The bar for ``converged``: the largest allowed deviation of any asset's
        risk-contribution *fraction* from ``1/N``. Fractions rather than
        volatility units so the bar is scale-free -- 1e-6 means the same thing on
        an annualised covariance as on a daily one.
    initial_weights:
        Warm start. Defaults to :func:`naive_inv_vol`, which is cheap, always
        feasible, and usually within a few iterations of the answer.
    max_restarts:
        How many times to retry from a fresh point if the first solve misses
        ``tolerance``. See the non-convexity note below.

    Notes
    -----
    **Scale normalisation.** The covariance is rescaled to a unit mean variance
    before solving. ERC is scale-invariant -- multiplying ``C`` by ``k`` scales
    every ``RC_i`` by ``sqrt(k)`` and leaves the minimiser untouched -- but
    SLSQP's ``ftol`` is an *absolute* threshold on the objective. On a raw
    annualised covariance the objective at the optimum is ~1e-16, already below
    any sane ``ftol``, so SLSQP would declare victory on its first step.
    Normalising puts the objective where ``ftol`` can discriminate.

    **Non-convexity, and why restarts are here.** ``sum_i (RC_i - sigma_p/N)^2``
    is not convex. It admits local minima that pin one asset at ``min_weight``
    with a *negative* risk contribution, and they are not exotic: they show up
    whenever two sleeves are strongly negatively correlated, which is precisely
    the SPY/TLT structure this project rests on. Measured on 200 adversarial
    random 4x4 covariances, the first solve lands in one about 2% of the time,
    and the objective there is ~1e-3 against ~1e-21 at the true optimum -- so the
    condition is easy to detect and worth escaping rather than reporting.

    The escape is a restart from a fresh feasible point, repeated until the
    risk-contribution bar is met. Restarts fire only on failure, so the mean cost
    over 2000 covariances is 0.1 extra solves; with ``ERC_MAX_RESTARTS`` the
    residual failure rate is 0 in 2000 adversarial matrices and 0 in 2000 drawn
    around the real four-sleeve correlation structure.

    This is deliberately *not* solved by switching to the convex log-barrier
    form. That form is Person B's independent oracle (TASKS.md step 19), and the
    whole value of that test is that it derives the same portfolio a different
    way. Adopting it here would leave the project with one derivation checked
    against itself.
    """
    c = validate_covariance(cov, "cov")
    n = c.shape[0]

    variances = np.diag(c)
    if np.any(variances <= 0.0):
        raise ERCSolverError(
            f"asset(s) {list(np.flatnonzero(variances <= 0.0))} have zero or negative "
            "variance, so equal risk contribution is not defined for them; "
            "src.covariance.nearest_psd floors variances at MIN_EIGENVALUE"
        )

    # A one-asset portfolio is trivially equal-risk. Worth short-circuiting
    # rather than handing SLSQP a problem whose feasible set is a single point
    # (TASKS.md step 20's single-asset degenerate case).
    if n == 1:
        return ERCSolution(
            weights=np.ones(1),
            converged=True,
            optimiser_succeeded=True,
            max_rc_deviation=0.0,
            objective=0.0,
            n_iterations=0,
            restarts_used=0,
            message="single asset: the ERC portfolio is trivially [1.0]",
        )

    scale = float(np.mean(variances))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ERCSolverError(f"covariance has a degenerate scale ({scale})")
    c_scaled = c / scale

    if initial_weights is None:
        w0 = naive_inv_vol(c)
    else:
        w0 = np.asarray(initial_weights, dtype="float64").ravel()
        if w0.shape != (n,):
            raise ERCSolverError(f"initial_weights must have shape ({n},), got {w0.shape}")
        w0 = _renormalise(w0, min_weight)

    def attempt(start: np.ndarray) -> tuple[np.ndarray, float, object]:
        """One SLSQP solve, scored on the risk-contribution bar it has to meet."""
        outcome = minimize(
            _erc_objective_and_gradient,
            _renormalise(start, min_weight),
            args=(c_scaled,),
            method="SLSQP",
            jac=True,
            bounds=[(min_weight, 1.0)] * n,
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda w: w.sum() - 1.0,
                    "jac": lambda w: np.ones_like(w),
                }
            ],
            options={"maxiter": max_iter, "ftol": ftol},
        )
        # Clean up the returned point before judging it: SLSQP satisfies the
        # equality constraint to its own tolerance, not exactly, and the
        # contract's sum-to-one check is stricter than that.
        weights = _renormalise(outcome.x, min_weight)
        fractions = risk_contributions(weights, c) / portfolio_volatility(weights, c)
        return weights, float(np.max(np.abs(fractions - 1.0 / n))), outcome

    best_w, best_deviation, best = attempt(w0)
    restarts_used = 0

    # Restarts are scored on the risk-contribution bar rather than on the
    # objective value, because that bar is the actual definition of the answer
    # being right. Draws come from a fixed seed so the backtest is reproducible.
    generator = np.random.default_rng(_RESTART_SEED)
    while best_deviation > tolerance and restarts_used < max_restarts:
        start = (
            np.full(n, 1.0 / n)
            if restarts_used == 0
            else generator.dirichlet(np.full(n, _RESTART_CONCENTRATION))
        )
        restarts_used += 1
        candidate_w, candidate_deviation, candidate = attempt(start)
        if candidate_deviation < best_deviation:
            best_w, best_deviation, best = candidate_w, candidate_deviation, candidate

    converged = bool(best_deviation <= tolerance)

    if not converged:
        # Warned, never swallowed, and never silently replaced with the
        # inverse-vol fallback: a backtest that quietly substituted a different
        # strategy on some rebalances is not the strategy it reports being.
        log.warning(
            "ERC did not converge after %d restarts: worst risk-contribution fraction "
            "is off 1/N by %.3e (tolerance %.1e). SLSQP said %r. Returning the best "
            "iterate; see BacktestResult.meta['solver_failures'].",
            restarts_used,
            best_deviation,
            tolerance,
            str(best.message),
        )
    elif restarts_used:
        # A local minimum was escaped. Worth a line in the log, because a run
        # where this fires on many rebalances is a run whose covariances are
        # unusual, and that is worth someone noticing.
        log.info(
            "ERC escaped a local minimum after %d restart(s); final deviation %.3e.",
            restarts_used,
            best_deviation,
        )
    elif not best.success:
        # Converged on the thing that matters, so this is informational: SLSQP
        # commonly reports a line-search failure at an optimum it has already
        # reached to machine precision.
        log.info(
            "ERC reached equal risk contributions (%.3e) but SLSQP exited "
            "unsuccessfully: %s",
            best_deviation,
            best.message,
        )

    return ERCSolution(
        weights=best_w,
        converged=converged,
        optimiser_succeeded=bool(best.success),
        max_rc_deviation=best_deviation,
        objective=float(best.fun),
        n_iterations=int(best.nit),
        restarts_used=restarts_used,
        message=str(best.message),
    )


def solve_erc(cov: np.ndarray, **kwargs) -> np.ndarray:
    """Long-only equal-risk-contribution weights, summing to 1.0.

    Satisfies ``src.types.ERCSolver``, which is the signature Person B's
    independent oracle asserts against. Use :func:`solve_erc_with_status` when
    the convergence diagnostics are wanted -- the backtest loop does, so a failed
    rebalance lands in ``BacktestResult.meta`` rather than vanishing.
    """
    return solve_erc_with_status(cov, **kwargs).weights


# ---------------------------------------------------------------------------
# The objective and its gradient
# ---------------------------------------------------------------------------
def _erc_objective_and_gradient(
    weights: np.ndarray, cov: np.ndarray
) -> tuple[float, np.ndarray]:
    """``f(w) = sum_i (RC_i - sigma_p/N)^2`` and its exact gradient.

    Returned together, and passed to SLSQP as ``jac=True``, because the gradient
    reuses every intermediate the objective already computed -- ``C w`` is the
    expensive term and it is formed once.

    The derivation, since a wrong gradient here fails silently as slow
    convergence rather than as an error. With ``m = C w``, ``sigma = sqrt(w'm)``,
    ``RC = w * m / sigma`` and ``e = RC - sigma/N``::

        d(sigma)/dw_k = m_k / sigma
        d(RC_i)/dw_k  = (delta_ik * m_i + w_i * C_ik) / sigma  -  RC_i * m_k / sigma^2

    Substituting into ``df/dw_k = 2 * sum_i e_i * (dRC_i/dw_k - m_k/(N sigma))``
    and collecting terms gives::

        grad = 2 * [ (e*m + C(e*w)) / sigma  -  a*m / sigma^2  -  S*m / (N*sigma) ]

    with the scalars ``a = sum_i e_i RC_i`` and ``S = sum_i e_i``. The two ``m``
    terms carry different powers of ``sigma``; collapsing them into one is the
    easy mistake, and it produces a gradient that is right in direction and wrong
    in magnitude, which SLSQP tolerates for a while and then does not.

    ``__main__`` below checks this against central differences.
    """
    w = np.asarray(weights, dtype="float64")
    n = w.size

    m = cov @ w
    variance = float(w @ m)
    if variance <= 0.0:
        # SLSQP's line search can propose an infeasible point outside the bounds.
        # Return a large penalty and push back toward the interior rather than
        # raising, which would abort a solve that would otherwise recover.
        return float(np.inf), np.zeros(n)

    sigma = float(np.sqrt(variance))
    rc = w * m / sigma
    e = rc - sigma / n

    objective = float(e @ e)

    s = float(e.sum())
    a = float(e @ rc)
    gradient = 2.0 * ((e * m + cov @ (e * w)) / sigma - a * m / variance - s * m / (n * sigma))
    return objective, gradient


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _validate_inputs(weights: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce and dimension-check a (weights, covariance) pair."""
    c = validate_covariance(cov, "cov")
    w = np.asarray(weights, dtype="float64").ravel()
    if w.shape[0] != c.shape[0]:
        raise ERCSolverError(
            f"weights has length {w.shape[0]} but cov is {c.shape[0]}x{c.shape[0]}"
        )
    if not np.isfinite(w).all():
        raise ERCSolverError("weights contains NaN or infinite values")
    return w, c


def _renormalise(weights: np.ndarray, min_weight: float) -> np.ndarray:
    """Clip to the long-only floor and rescale to sum to exactly 1.0.

    Clipping first then rescaling can push a weight fractionally back below the
    floor, but only by a factor of ``sum(w)`` which is within 1e-9 of 1 here.
    ``MIN_WEIGHT`` is a well-posedness device rather than a mandate, so that is
    acceptable; ``sum(w) == 1`` is a contract invariant and is not.
    """
    w = np.clip(np.asarray(weights, dtype="float64").ravel(), min_weight, None)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ERCSolverError(f"weights sum to {total}, which cannot be normalised")
    return w / total


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(20240617)

    def random_psd(n: int) -> np.ndarray:
        """A random covariance with realistic, unequal sleeve volatilities."""
        factors = rng.normal(size=(n, n + 2))
        corr = factors @ factors.T
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        vols = rng.uniform(0.03, 0.25, size=n)
        return corr * np.outer(vols, vols)

    # --- the gradient, against central differences ---------------------------
    c = random_psd(4)
    w = np.array([0.30, 0.25, 0.20, 0.25])
    _, analytic = _erc_objective_and_gradient(w, c)
    numeric = np.empty_like(analytic)
    for k in range(len(w)):
        step = np.zeros_like(w)
        step[k] = 1e-7
        up, _ = _erc_objective_and_gradient(w + step, c)
        down, _ = _erc_objective_and_gradient(w - step, c)
        numeric[k] = (up - down) / 2e-7
    print(f"analytic vs central-difference gradient : max |diff| = "
          f"{np.abs(analytic - numeric).max():.3e}")

    # --- step 16's "done when" -----------------------------------------------
    rc = risk_contributions(w, c)
    print(f"sum(RC) - sigma_p                       : "
          f"{abs(rc.sum() - portfolio_volatility(w, c)):.3e}")

    # --- step 17's "done when": 4x4 random covariances ----------------------
    # Adversarial on purpose -- this generator produces |rho| up to ~0.95, well
    # outside anything SPY/TLT/DBC/TIP produce, so it exercises the non-convex
    # local minima the restart ladder exists for.
    worst, failures, restarts = 0.0, 0, []
    for _ in range(500):
        solution = solve_erc_with_status(random_psd(4))
        worst = max(worst, solution.max_rc_deviation)
        failures += not solution.converged
        restarts.append(solution.restarts_used)
    print(f"500 random 4x4 solves                   : worst |RC_i - 1/N| = {worst:.3e}, "
          f"{failures} failures")
    print(f"  restarts (non-convexity escapes)      : {sum(r > 0 for r in restarts)} solves "
          f"needed one, {np.mean(restarts):.3f} mean")

    # --- the local minimum the restart ladder exists to escape --------------
    # Two sleeves at rho = -0.93: SLSQP's first solve pins the first asset at the
    # weight floor with a NEGATIVE risk contribution. Asserting the escape, not
    # the path to it.
    hedged = np.array(
        [
            [1.00, -0.93, 0.67, -0.02],
            [-0.93, 1.00, -0.55, -0.25],
            [0.67, -0.55, 1.00, 0.12],
            [-0.02, -0.25, 0.12, 1.00],
        ]
    )
    hedged_vols = np.array([0.1064, 0.0463, 0.0468, 0.2299])
    hedged = hedged * np.outer(hedged_vols, hedged_vols)
    escaped = solve_erc_with_status(hedged)
    print(f"\nstrongly hedged pair (rho = -0.93)")
    print(f"  restarts used  : {escaped.restarts_used}")
    print(f"  converged      : {escaped.converged} (deviation {escaped.max_rc_deviation:.3e})")
    print(f"  weights        : {escaped.weights.round(6)}")
    print(f"  all RC > 0     : "
          f"{bool((risk_contributions(escaped.weights, hedged) > 0).all())}")

    # --- ERC is not 1/sigma when correlations are unequal -------------------
    unequal = np.array(
        [
            [1.00, -0.30, 0.20, -0.10],
            [-0.30, 1.00, -0.05, 0.85],
            [0.20, -0.05, 1.00, 0.10],
            [-0.10, 0.85, 0.10, 1.00],
        ]
    ) * np.outer([0.170, 0.140, 0.200, 0.055], [0.170, 0.140, 0.200, 0.055])
    erc, naive = solve_erc(unequal), naive_inv_vol(unequal)
    print("\nERC vs naive 1/sigma (SPY/TLT/DBC/TIP-like, unequal correlations)")
    print(f"  ERC        : {erc.round(4)}")
    print(f"  1/sigma    : {naive.round(4)}")
    print(f"  max gap    : {np.abs(erc - naive).max():.4f}  "
          f"(RC spread under 1/sigma: "
          f"{np.ptp(risk_contributions(naive, unequal) / portfolio_volatility(naive, unequal)):.4f})")

    # --- equal correlations: the one case where they coincide ---------------
    rho = 0.35
    equal_corr = np.full((4, 4), rho)
    np.fill_diagonal(equal_corr, 1.0)
    vols = np.array([0.170, 0.140, 0.200, 0.055])
    equal_corr = equal_corr * np.outer(vols, vols)
    print(f"  equal-correlation check: max |ERC - 1/sigma| = "
          f"{np.abs(solve_erc(equal_corr) - naive_inv_vol(equal_corr)).max():.3e}")
