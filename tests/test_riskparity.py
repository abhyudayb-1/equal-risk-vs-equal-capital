"""Self-checks for risk contributions and the ERC solver (TASKS.md steps 18-20).

Runs standalone with no test framework, matching the house style::

    .venv/bin/python tests/test_riskparity.py

Three steps live in this one file because they attack one object from three
angles:

* **Step 18 -- the identity.** ``RC_i = w_i (Sigma w)_i / sigma_p`` sums to
  ``sigma_p`` and the normalised fractions sum to 1. This is not bookkeeping: it
  is what makes the word "contribution" mean anything. The Euler check below
  proves ``RC_i`` really is ``w_i * d(sigma_p)/d(w_i)`` rather than a plausible
  ratio that happens to add up.
* **Step 19 -- the cross-check.** The equal-risk portfolio is derived here a
  second time, independently, from the convex form
  ``min 0.5 w'Sigma w - (1/N) sum ln(w_i)`` and renormalised. Its stationarity
  condition is ``(Sigma w)_i = 1/(N w_i)``, i.e. ``w_i (Sigma w)_i`` is the same
  for every i -- equal risk contributions, with no reference whatsoever to how
  Person A's SLSQP solver reaches the same point. Two derivations agreeing on 50
  random covariances is the only evidence either one is right.
* **Step 20 -- the degenerate cases.** Perfect correlation, near-singularity, a
  single asset, a near-zero-vol sleeve, and non-PSD input. Each must either solve
  or raise something legible. Silent garbage here would propagate into weights,
  and a wrong weight vector still produces a perfectly plausible equity curve.

The claim the whole project rests on is asserted in
:func:`test_erc_is_not_inverse_vol`: ERC coincides with naive ``1/sigma``
weighting only when every pairwise correlation is equal. If the two were the same
thing in general, there would be no reason to build a solver at all.

**On what is under test.** ``src/riskparity.py`` is Person A's (steps 16-17). If
it is importable, it is what these checks run against and the fallback below is
never used. Until then the file still runs end to end against a private stand-in,
following the ``benchmarks._ewma_cov`` pattern: private to this test module,
never exported from ``src/``, and displaced automatically the moment A's module
lands. The banner printed at startup says which one ran, so a green result can
never be mistaken for evidence about code that does not exist yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.benchmarks import risk_contribution_fractions  # noqa: E402

_checks = 0

# Tolerances. The identity is exact algebra, so it is held to floating-point
# noise; solver agreement is held to the tolerance the contract names.
IDENTITY_TOLERANCE = 1e-10
AGREEMENT_TOLERANCE = config.ERC_AGREEMENT_TOLERANCE
N_RANDOM_COVARIANCES = 50

# Only used by the stand-in solver below -- see its docstring for why SLSQP needs
# its objective inflated and its answer checked.
_SLSQP_OBJECTIVE_SCALE = 1e5
_SLSQP_MAX_RC_DEVIATION = 1e-7


def check(label: str, condition: bool) -> None:
    global _checks
    if not condition:
        raise AssertionError(f"FAILED: {label}")
    _checks += 1
    print(f"  ok   {label}")


def expect_raises(label: str, exception: type[Exception], call) -> None:
    try:
        call()
    except exception:
        check(label, True)
        return
    raise AssertionError(f"FAILED: {label} -- no {exception.__name__} raised")


# ---------------------------------------------------------------------------
# Covariance fixtures
# ---------------------------------------------------------------------------
def random_covariance(n: int, rng: np.random.Generator) -> np.ndarray:
    """A random PSD annualized covariance with realistic sleeve volatilities.

    Built as ``diag(sigma) C diag(sigma)`` from a Wishart-style correlation draw
    rather than from a random matrix, so the volatilities stay in the 3%-30%
    range this project actually sees and the conditioning is representative. A
    uniformly random PSD matrix would test the solver on covariances no ETF panel
    could produce.
    """
    a = rng.standard_normal((n, n + 4))
    cov = a @ a.T
    sd = np.sqrt(np.diag(cov))
    correlation = cov / np.outer(sd, sd)
    sigma = rng.uniform(0.03, 0.30, size=n)
    return np.outer(sigma, sigma) * correlation


def equicorrelation_covariance(sigma: np.ndarray, rho: float) -> np.ndarray:
    """Covariance where every pair has the same correlation ``rho``.

    The one case with a closed-form answer: with equal correlations the ERC
    weights are exactly proportional to ``1/sigma_i``. Used twice below -- once
    to check the solver against an analytic result, and once as the contrast that
    makes :func:`test_erc_is_not_inverse_vol` mean something.
    """
    n = len(sigma)
    correlation = np.full((n, n), rho, dtype="float64")
    np.fill_diagonal(correlation, 1.0)
    return np.outer(sigma, sigma) * correlation


def inverse_vol_weights(cov: np.ndarray) -> np.ndarray:
    """Naive ``1/sigma`` weights -- the comparator ERC has to differ from."""
    sigma = np.sqrt(np.diag(cov))
    return (1.0 / sigma) / float((1.0 / sigma).sum())


# ---------------------------------------------------------------------------
# Reference implementations (private to this file -- see the module docstring)
# ---------------------------------------------------------------------------
def _validate_covariance(cov: np.ndarray) -> np.ndarray:
    """Reject anything that is not a usable covariance, loudly.

    Every rejection here is a case that would otherwise produce weights: a
    non-symmetric matrix silently uses only its symmetric part, a negative
    eigenvalue makes ``sigma_p`` imaginary for some w, and a non-positive
    variance makes the risk contribution of that asset undefined.
    """
    cov = np.asarray(cov, dtype="float64")
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"covariance must be square, got shape {cov.shape}")
    if not np.isfinite(cov).all():
        raise ValueError("covariance contains NaN or non-finite entries")
    if not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError("covariance is not symmetric")
    if (np.diag(cov) <= 0.0).any():
        raise ValueError("every asset must have strictly positive variance")
    smallest = float(np.linalg.eigvalsh(cov).min())
    # Scale the PSD test by the size of the matrix: an absolute -1e-10 is noise
    # on an annualized equity covariance and a real negative direction on a
    # covariance of basis-point returns.
    if smallest < -1e-10 * max(1.0, float(np.abs(np.diag(cov)).max())):
        raise ValueError(
            f"covariance is not positive semi-definite (min eigenvalue {smallest:.3e})"
        )
    return cov


def _risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """``RC_i = w_i (Sigma w)_i / sigma_p``, in volatility units, summing to sigma_p.

    Person A's ``riskparity.risk_contributions`` (step 16) is the module version;
    this is the same three lines of algebra written independently so that step 18
    is a check rather than a tautology. When A's module is importable the checks
    below run against *both* and assert they agree.
    """
    weights = np.asarray(weights, dtype="float64")
    variance = float(weights @ cov @ weights)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError(
            f"portfolio variance is {variance}; risk contributions are undefined"
        )
    return weights * (cov @ weights) / np.sqrt(variance)


def _oracle_erc(cov: np.ndarray, tolerance: float = 1e-14, max_iter: int = 10_000) -> np.ndarray:
    """**Step 19.** ERC from the convex log-barrier form, written from the math.

    Minimise ``f(y) = 0.5 y'Sigma y - (1/N) sum ln(y_i)`` over ``y > 0``, then
    renormalise ``w = y / sum(y)``. ``f`` is strictly convex on the positive
    orthant for PSD ``Sigma`` (the log barrier supplies strict convexity even
    when ``Sigma`` is singular), so its stationary point is the unique global
    minimum. Setting the gradient to zero,

        ``(Sigma y)_i = 1 / (N y_i)``  =>  ``y_i (Sigma y)_i = 1/N  for every i``

    -- every asset contributes the same amount to variance, which is exactly the
    ERC condition. Scaling ``y`` by a positive constant scales every risk
    contribution equally, so the renormalised ``w`` is still ERC and now sums to
    1. That is the whole derivation; no part of it looks at how ``solve_erc``
    reaches the same point.

    Solved by cyclical coordinate descent: with the other coordinates fixed,
    ``df/dy_i = 0`` is the quadratic ``Sigma_ii y_i^2 + b_i y_i - 1/N = 0`` with
    ``b_i = sum_{j != i} Sigma_ij y_j``, whose positive root is written down in
    closed form below. Every step is a global minimum along its coordinate, so
    the objective decreases monotonically and the iteration cannot oscillate.
    The positive root is taken via ``2c / (b + sqrt(b^2 + 4ac))`` rather than
    ``(-b + sqrt(...)) / 2a``: for the near-zero-vol sleeve in step 20 the two
    are algebraically identical and the latter is catastrophic cancellation.
    """
    cov = _validate_covariance(cov)
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)

    target = 1.0 / n
    # Inverse-vol start: the exact answer under equicorrelation and a good start
    # everywhere else, so the iteration count stays low even when near-singular.
    y = inverse_vol_weights(cov) / np.sqrt(float(np.diag(cov).mean()))

    for _ in range(max_iter):
        previous = y.copy()
        for i in range(n):
            a = cov[i, i]
            b = float(cov[i] @ y) - a * y[i]
            y[i] = 2.0 * target / (b + np.sqrt(b * b + 4.0 * a * target))
        if np.max(np.abs(y - previous)) <= tolerance * max(1.0, float(np.max(y))):
            break
    else:
        raise RuntimeError(
            f"convex ERC oracle did not converge in {max_iter} sweeps -- do not "
            "trust the weights, and do not widen the tolerance to make this pass"
        )

    if not np.isfinite(y).all() or (y <= 0.0).any():
        raise RuntimeError(f"convex ERC oracle produced a degenerate solution: {y}")
    return y / y.sum()


def _fallback_solve_erc(cov: np.ndarray, **kwargs) -> np.ndarray:
    """Stand-in for ``riskparity.solve_erc`` until step 17 is merged.

    Deliberately the *other* formulation from :func:`_oracle_erc`, and the one
    step 17 specifies: SLSQP on the least-squares objective
    ``sum_i (b_i - sigma_p^2/N)^2`` with ``b_i = w_i (Sigma w)_i``, subject to
    ``sum w = 1`` and ``w >= MIN_WEIGHT``. Minimising squared deviations under
    explicit constraints has nothing in common with driving a log barrier to its
    stationary point, so an agreement between the two is informative -- but this
    is a stand-in, not A's code, which is why the banner says which one ran.

    Four details, each of which was a wrong answer before it was a line of code:

    * The covariance is rescaled to unit mean variance and the objective is
      multiplied by ``_SLSQP_OBJECTIVE_SCALE``. SLSQP's ``ftol`` is a tolerance
      on the *objective value*, and this objective is quartic in the covariance
      scale: on annualized numbers it sits near 1e-8 at the start, so the search
      terminates "successfully" while the weights are still 1e-4 out. Weights are
      invariant to both rescalings, so neither costs anything.
    * The gradient is analytic. ``sum_i (b_i - c) == 0`` identically (the ``b_i``
      sum to ``sigma_p^2`` and ``c`` is their mean), which kills the term in
      ``dc/dw_k`` and leaves ``2[(b_k - c)(Sigma w)_k + (Sigma (d * w))_k]``.
    * Two starting points, best result wins. The least-squares surface is not
      convex, and from an inverse-vol start one asset gets pinned at the
      ``MIN_WEIGHT`` bound on roughly one covariance in fifty -- SLSQP reports
      success at a corner that is 20 percentage points of risk share away from
      ERC. That is exactly the failure step 20 exists to catch, so it is caught
      here rather than tolerated.
    * The answer is verified before it is returned. A least-squares solver that
      stalls has no idea it stalled; the risk contributions do.
    """
    from scipy.optimize import minimize

    cov = _validate_covariance(cov)
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)

    scaled = cov / float(np.mean(np.diag(cov)))

    def objective(w: np.ndarray) -> float:
        marginal = scaled @ w
        deviation = w * marginal - float(w @ marginal) / n
        return _SLSQP_OBJECTIVE_SCALE * float(deviation @ deviation)

    def gradient(w: np.ndarray) -> np.ndarray:
        marginal = scaled @ w
        deviation = w * marginal - float(w @ marginal) / n
        return _SLSQP_OBJECTIVE_SCALE * 2.0 * (
            deviation * marginal + scaled @ (deviation * w)
        )

    best_weights, best_deviation = None, np.inf
    for start in (inverse_vol_weights(cov), np.full(n, 1.0 / n)):
        w = start
        for _ in range(3):  # restart from the previous answer to squeeze out ftol
            result = minimize(
                objective,
                w,
                jac=gradient,
                method="SLSQP",
                bounds=[(config.MIN_WEIGHT, 1.0)] * n,
                constraints={
                    "type": "eq",
                    "fun": lambda w: float(w.sum() - 1.0),
                    "jac": lambda w: np.ones(n),
                },
                options={"maxiter": config.ERC_MAX_ITER, "ftol": config.ERC_TOLERANCE},
            )
            if not result.success:
                break
            w = result.x / result.x.sum()
        deviation = float(np.max(np.abs(risk_contribution_fractions(w, cov) - 1.0 / n)))
        if deviation < best_deviation:
            best_weights, best_deviation = w, deviation

    if best_weights is None or best_deviation > _SLSQP_MAX_RC_DEVIATION:
        raise RuntimeError(
            f"fallback SLSQP stalled: worst risk contribution is {best_deviation:.2e} "
            f"from 1/N, outside {_SLSQP_MAX_RC_DEVIATION:.0e}"
        )
    return best_weights


def _load_solver() -> tuple[str, object, object]:
    """Return ``(label, solve_erc, risk_contributions)`` for whatever exists.

    Person A's module wins whenever it imports. ``naive_inv_vol`` and
    ``risk_contributions`` come from there too when present, so steps 16 and 17
    are both genuinely under test rather than shadowed by the local copies.
    """
    try:
        from src import riskparity  # type: ignore
    except ImportError:
        return ("fallback (src/riskparity.py not built yet)", _fallback_solve_erc, None)

    solver = getattr(riskparity, "solve_erc", None)
    if solver is None:
        raise AssertionError("src/riskparity.py exists but exports no solve_erc")
    try:  # the step 3 stub is importable and raises -- that is not a solver
        solver(np.eye(2) * 0.04)
    except NotImplementedError:
        return ("fallback (solve_erc is still the contract stub)", _fallback_solve_erc, None)
    except Exception:  # a real solver that dislikes this input: still real
        pass
    return ("src.riskparity.solve_erc", solver, getattr(riskparity, "risk_contributions", None))


SOLVER_LABEL, solve_erc, module_risk_contributions = _load_solver()


# ---------------------------------------------------------------------------
# Step 18 -- the risk contribution identity
# ---------------------------------------------------------------------------
def test_rc_identity() -> None:
    """``sum(RC) == sigma_p`` to 1e-10, on arbitrary weights, not just ERC ones."""
    print("risk contribution identity:")
    rng = np.random.default_rng(18)

    worst_sum, worst_fraction = 0.0, 0.0
    for _ in range(200):
        n = int(rng.integers(2, 7))
        cov = random_covariance(n, rng)
        w = rng.dirichlet(np.ones(n))  # long-only, sums to 1, nothing else assumed

        rc = _risk_contributions(w, cov)
        sigma_p = np.sqrt(float(w @ cov @ w))
        worst_sum = max(worst_sum, abs(float(rc.sum()) - sigma_p))
        worst_fraction = max(
            worst_fraction, abs(float(risk_contribution_fractions(w, cov).sum()) - 1.0)
        )

    check(
        f"sum(RC) == portfolio vol on 200 random portfolios (worst {worst_sum:.2e})",
        worst_sum < IDENTITY_TOLERANCE,
    )
    check(
        f"RC fractions sum to 1.0 (worst {worst_fraction:.2e})",
        worst_fraction < IDENTITY_TOLERANCE,
    )

    # The two forms must be the same object viewed at two scales: benchmarks.py
    # stores fractions, riskparity.py works in volatility units.
    cov = random_covariance(4, rng)
    w = rng.dirichlet(np.ones(4))
    rc = _risk_contributions(w, cov)
    check(
        "the fraction form is the volatility form divided by sigma_p",
        bool(np.allclose(rc / rc.sum(), risk_contribution_fractions(w, cov), atol=1e-12)),
    )


def test_rc_is_a_derivative() -> None:
    """Euler's identity: ``RC_i == w_i * d(sigma_p)/d(w_i)``.

    Without this, ``RC_i`` is just a ratio that happens to add up to sigma_p and
    there are infinitely many of those. With it, the number is the marginal
    volatility of the position, which is the only reading under which "equalise
    the risk contributions" is a meaningful instruction. Checked against a
    central finite difference, so it tests the formula rather than restating it.
    """
    print("risk contributions are marginal volatilities:")
    rng = np.random.default_rng(180)
    cov = random_covariance(4, rng)
    w = rng.dirichlet(np.ones(4))

    def portfolio_vol(weights: np.ndarray) -> float:
        return float(np.sqrt(weights @ cov @ weights))

    h = 1e-6
    numeric = np.empty(4)
    for i in range(4):
        step = np.zeros(4)
        step[i] = h
        numeric[i] = (portfolio_vol(w + step) - portfolio_vol(w - step)) / (2 * h)

    analytic = _risk_contributions(w, cov)
    check(
        "RC_i matches w_i * d(sigma_p)/d(w_i) from a central difference",
        bool(np.allclose(analytic, w * numeric, rtol=1e-6, atol=1e-9)),
    )
    check(
        "the marginals sum to sigma_p by homogeneity of degree one",
        bool(np.isclose(float((w * numeric).sum()), portfolio_vol(w), rtol=1e-6)),
    )


def test_solver_output_is_a_valid_allocation() -> None:
    """Weights sum to 1, are long-only, and their RC fractions sum to 1."""
    print("solver output is a valid allocation:")
    rng = np.random.default_rng(181)

    worst_sum, worst_negative, worst_rc = 0.0, 0.0, 0.0
    for _ in range(N_RANDOM_COVARIANCES):
        cov = random_covariance(4, rng)
        w = np.asarray(solve_erc(cov), dtype="float64")
        worst_sum = max(worst_sum, abs(float(w.sum()) - 1.0))
        worst_negative = min(worst_negative, float(w.min()))
        worst_rc = max(worst_rc, abs(float(risk_contribution_fractions(w, cov).sum()) - 1.0))

    check(f"weights sum to 1.0 (worst {worst_sum:.2e})", worst_sum < AGREEMENT_TOLERANCE)
    check(f"weights are long-only (smallest {worst_negative:.2e})", worst_negative >= 0.0)
    check(f"RC fractions sum to 1.0 (worst {worst_rc:.2e})", worst_rc < IDENTITY_TOLERANCE)


def test_erc_equalises_risk() -> None:
    """The defining property: every asset contributes 1/N of the risk."""
    print("the ERC solution equalises risk contributions:")
    rng = np.random.default_rng(182)

    worst = 0.0
    for _ in range(N_RANDOM_COVARIANCES):
        n = int(rng.integers(2, 7))
        cov = random_covariance(n, rng)
        fractions = risk_contribution_fractions(np.asarray(solve_erc(cov)), cov)
        worst = max(worst, float(np.max(np.abs(fractions - 1.0 / n))))

    check(
        f"every RC fraction is 1/N to {AGREEMENT_TOLERANCE:g} (worst {worst:.2e})",
        worst < AGREEMENT_TOLERANCE,
    )


def test_module_risk_contributions_agree() -> None:
    """Cross-check A's step 16 function against the local algebra, when it exists."""
    print("module risk_contributions vs the local form:")
    if module_risk_contributions is None:
        print("  skip src.riskparity.risk_contributions is not built yet")
        return

    rng = np.random.default_rng(183)
    worst = 0.0
    for _ in range(50):
        cov = random_covariance(4, rng)
        w = rng.dirichlet(np.ones(4))
        theirs = np.asarray(module_risk_contributions(w, cov), dtype="float64")
        mine = _risk_contributions(w, cov)
        # A returns volatility units per the step 16 spec; accept the normalised
        # form too rather than failing on a convention difference.
        if np.isclose(theirs.sum(), 1.0) and not np.isclose(mine.sum(), 1.0):
            mine = mine / mine.sum()
        worst = max(worst, float(np.max(np.abs(theirs - mine))))

    check(f"the two forms agree to 1e-12 (worst {worst:.2e})", worst < 1e-12)


# ---------------------------------------------------------------------------
# Step 19 -- the independent oracle
# ---------------------------------------------------------------------------
def test_oracle_agrees_with_solver() -> None:
    """**The cross-check.** Two derivations, 50 covariances, one answer."""
    print(f"oracle vs {SOLVER_LABEL}:")
    rng = np.random.default_rng(19)

    worst_weight, worst_case = 0.0, None
    for _ in range(N_RANDOM_COVARIANCES):
        n = int(rng.integers(2, 7))
        cov = random_covariance(n, rng)
        gap = float(np.max(np.abs(np.asarray(solve_erc(cov)) - _oracle_erc(cov))))
        if gap > worst_weight:
            worst_weight, worst_case = gap, cov

    check(
        f"agree on {N_RANDOM_COVARIANCES} random covariances to "
        f"{AGREEMENT_TOLERANCE:g} (worst {worst_weight:.2e})",
        worst_weight < AGREEMENT_TOLERANCE,
        )
    if worst_case is not None and worst_weight > AGREEMENT_TOLERANCE / 100:
        # Not a failure, but worth seeing: a solver that is merely inside
        # tolerance on well-conditioned 4x4 input is not one to trust in 2020.
        print(f"       (worst case condition number {np.linalg.cond(worst_case):.1f})")


def test_oracle_is_erc_by_construction() -> None:
    """The oracle earns the name only if its own risk contributions are equal.

    Checked separately from the agreement test: if the oracle were wrong in the
    same way A's solver is wrong, agreement alone would prove nothing. This is
    what makes the cross-check evidence rather than a coincidence.
    """
    print("the oracle is itself ERC:")
    rng = np.random.default_rng(190)

    worst = 0.0
    for _ in range(N_RANDOM_COVARIANCES):
        n = int(rng.integers(2, 7))
        cov = random_covariance(n, rng)
        fractions = risk_contribution_fractions(_oracle_erc(cov), cov)
        worst = max(worst, float(np.max(np.abs(fractions - 1.0 / n))))

    check(f"oracle RC fractions are all 1/N (worst {worst:.2e})", worst < 1e-10)


def test_analytic_case() -> None:
    """Both solvers against the one case with a closed form.

    Under equal pairwise correlations the ERC weights are exactly proportional to
    ``1/sigma_i``. An external truth, not another numerical solution -- if both
    implementations shared a sign error this is the check that would catch it.
    """
    print("the equicorrelation closed form:")
    sigma = np.array([0.16, 0.13, 0.20, 0.05])

    for rho in (0.0, 0.3, 0.7):
        cov = equicorrelation_covariance(sigma, rho)
        expected = inverse_vol_weights(cov)
        check(
            f"rho={rho}: the oracle reproduces 1/sigma weights",
            bool(np.allclose(_oracle_erc(cov), expected, atol=1e-9)),
        )
        check(
            f"rho={rho}: {SOLVER_LABEL} reproduces 1/sigma weights",
            bool(np.allclose(np.asarray(solve_erc(cov)), expected, atol=AGREEMENT_TOLERANCE)),
        )


# ---------------------------------------------------------------------------
# Step 20 -- degenerate cases
# ---------------------------------------------------------------------------
def test_erc_is_not_inverse_vol() -> None:
    """**The claim the project rests on.**

    Naive ``1/sigma`` weighting equalises risk only if it can ignore
    correlations, which it can when they are all equal. When they are not --
    which is every real covariance, and emphatically the 2022 one, where SPY/TLT
    sat near -0.09 while DBC decoupled from both -- ERC and inverse-vol are
    different portfolios. If this check ever passes trivially, the solver has
    collapsed to the comparator and every result in the README is really an
    inverse-vol result.
    """
    print("ERC is not 1/sigma when correlations are unequal:")
    sigma = np.array([0.16, 0.13, 0.20, 0.05])
    correlation = np.array([
        [1.00, -0.10, 0.15, 0.05],
        [-0.10, 1.00, -0.30, 0.60],
        [0.15, -0.30, 1.00, 0.10],
        [0.05, 0.60, 0.10, 1.00],
    ])
    cov = np.outer(sigma, sigma) * correlation
    _validate_covariance(cov)

    erc = np.asarray(solve_erc(cov))
    naive = inverse_vol_weights(cov)
    gap = float(np.max(np.abs(erc - naive)))

    check(
        f"ERC and 1/sigma differ materially (max weight gap {gap:.4f})",
        gap > 1e-3,
    )
    check(
        "1/sigma does NOT equalise risk contributions here",
        float(np.max(np.abs(risk_contribution_fractions(naive, cov) - 0.25))) > 1e-3,
    )
    check(
        "...while ERC does, on the same covariance",
        float(np.max(np.abs(risk_contribution_fractions(erc, cov) - 0.25)))
        < AGREEMENT_TOLERANCE,
    )
    check(
        "and they coincide once every correlation is equal",
        bool(np.allclose(
            np.asarray(solve_erc(equicorrelation_covariance(sigma, 0.4))),
            inverse_vol_weights(equicorrelation_covariance(sigma, 0.4)),
            atol=AGREEMENT_TOLERANCE,
        )),
    )


def test_single_asset() -> None:
    print("single asset:")
    cov = np.array([[0.04]])
    for label, fn in (("oracle", _oracle_erc), (SOLVER_LABEL, solve_erc)):
        w = np.asarray(fn(cov), dtype="float64")
        check(f"{label}: the only asset gets 100%", bool(np.allclose(w, [1.0])))


def test_perfectly_correlated() -> None:
    """rho = 1 exactly: singular covariance, but ERC is still well defined.

    The portfolio collapses to a single risk factor, so ``sigma_p`` stays
    strictly positive and the answer is the inverse-vol one. The solver must
    reach it rather than tripping over the zero eigenvalue -- this is the case a
    plain matrix inverse in the objective would fail on.
    """
    print("perfectly correlated assets:")
    sigma = np.array([0.18, 0.12, 0.09])
    cov = equicorrelation_covariance(sigma, 1.0)
    check(
        "the covariance really is singular",
        float(np.linalg.eigvalsh(cov).min()) < 1e-12,
    )

    expected = inverse_vol_weights(cov)
    for label, fn in (("oracle", _oracle_erc), (SOLVER_LABEL, solve_erc)):
        w = np.asarray(fn(cov), dtype="float64")
        check(f"{label}: solves to 1/sigma weights", bool(np.allclose(w, expected, atol=1e-6)))
        check(f"{label}: weights are finite and sum to 1", bool(
            np.isfinite(w).all() and np.isclose(w.sum(), 1.0, atol=AGREEMENT_TOLERANCE)
        ))


def test_near_singular() -> None:
    """rho = 0.9999: the realistic version of the case above.

    Real covariances are never exactly singular, they are merely dreadful --
    TLT and TIP in a rate shock are close to one factor. The condition number
    here is around 1e4-1e5; the solver has to stay accurate on it, not just
    return something.
    """
    print("near-singular covariance:")
    sigma = np.array([0.18, 0.12, 0.09, 0.11])
    cov = equicorrelation_covariance(sigma, 0.9999)
    print(f"       condition number {np.linalg.cond(cov):.3e}")

    w = np.asarray(solve_erc(cov), dtype="float64")
    check(
        "the solver still equalises risk contributions",
        float(np.max(np.abs(risk_contribution_fractions(w, cov) - 0.25))) < 1e-6,
    )
    check(
        "and still agrees with the oracle",
        bool(np.allclose(w, _oracle_erc(cov), atol=AGREEMENT_TOLERANCE)),
    )


def test_near_zero_vol_asset() -> None:
    """One sleeve at 1bp vol: it should dominate by capital and still be bounded.

    Equalising *risk* across sleeves means an almost riskless one has to be held
    in enormous size to contribute its 1/N, which is the mechanism behind risk
    parity's leverage in the first place. The number must be large, finite, and
    still risk-equalising -- an overflow or a silent clip would both look like a
    plausible allocation downstream.
    """
    print("a near-zero-volatility asset:")
    sigma = np.array([0.16, 0.13, 0.0001])
    cov = equicorrelation_covariance(sigma, 0.2)

    w = np.asarray(solve_erc(cov), dtype="float64")
    check("the weights are all finite", bool(np.isfinite(w).all()))
    check("they still sum to 1", bool(np.isclose(w.sum(), 1.0, atol=AGREEMENT_TOLERANCE)))
    check(
        f"the near-riskless sleeve dominates capital ({w[2]:.4f})",
        w[2] > max(w[0], w[1]),
    )
    check(
        "risk contributions are still equal to 1e-6",
        float(np.max(np.abs(risk_contribution_fractions(w, cov) - 1.0 / 3))) < 1e-6,
    )
    check(
        "oracle and solver agree even here",
        bool(np.allclose(w, _oracle_erc(cov), atol=1e-6)),
    )


def test_bad_input_raises() -> None:
    """Every malformed covariance must raise, not return.

    A solver that quietly returns weights for a non-PSD covariance is the worst
    failure mode in this project: the run completes, the charts render, and the
    conclusion is wrong. These all go through the oracle, which is the
    implementation this file owns; A's solver gets the same treatment in step 29
    when convergence diagnostics land in ``meta``.
    """
    print("malformed input:")
    expect_raises(
        "a non-square matrix raises",
        ValueError,
        lambda: _oracle_erc(np.ones((3, 4))),
    )
    expect_raises(
        "an asymmetric matrix raises rather than being silently symmetrised",
        ValueError,
        lambda: _oracle_erc(np.array([[0.04, 0.01], [0.02, 0.09]])),
    )
    expect_raises(
        "a NaN entry raises",
        ValueError,
        lambda: _oracle_erc(np.array([[0.04, np.nan], [np.nan, 0.09]])),
    )
    expect_raises(
        "a negative eigenvalue raises",
        ValueError,
        lambda: _oracle_erc(np.array([[0.04, 0.10], [0.10, 0.09]])),
    )
    expect_raises(
        "a zero-variance asset raises",
        ValueError,
        lambda: _oracle_erc(np.array([[0.04, 0.0], [0.0, 0.0]])),
    )
    expect_raises(
        "zero weights make risk contributions undefined, and say so",
        ValueError,
        lambda: _risk_contributions(np.zeros(2), np.eye(2) * 0.04),
    )


if __name__ == "__main__":
    print(f"solver under test: {SOLVER_LABEL}\n")

    test_rc_identity()
    test_rc_is_a_derivative()
    test_solver_output_is_a_valid_allocation()
    test_erc_equalises_risk()
    test_module_risk_contributions_agree()

    test_oracle_agrees_with_solver()
    test_oracle_is_erc_by_construction()
    test_analytic_case()

    test_erc_is_not_inverse_vol()
    test_single_asset()
    test_perfectly_correlated()
    test_near_singular()
    test_near_zero_vol_asset()
    test_bad_input_raises()

    print(f"\n{_checks} checks passed  [{SOLVER_LABEL}]")
