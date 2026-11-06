"""Covariance estimation: EWMA, shrinkage and PSD repair.

The ERC solver in :mod:`src.riskparity` is only as trustworthy as the matrix it
is handed, and a covariance estimated from 252 daily observations of four
correlated sleeves is a noisy, frequently near-singular object. Three stages fix
that, in this order:

1. :func:`ewma_covariance` -- exponentially-weighted, so a regime change shows up
   in the weights within weeks rather than a year.
2. :func:`shrink_to_diagonal` -- pull the off-diagonals toward zero, which is
   where the estimation error concentrates.
3. :func:`nearest_psd` -- clip negative eigenvalues, so the matrix is a valid
   covariance even when stages 1-2 left it fractionally indefinite.

:func:`estimate_covariance` composes all three and is the function the rest of
the project calls. It matches ``src.benchmarks.CovEstimator``, so it wires into
the benchmarks by passing ``cov_estimator=`` rather than by editing anything::

    from src.covariance import estimate_covariance
    equal_weight(returns, cov_estimator=estimate_covariance)

``src.benchmarks`` imports this function directly, so the benchmarks and the
engine always see the same covariance.

Look-ahead: nothing in this module slices by date. Callers pass a window that
``src.data.asof_slice`` has already truncated to ``<= t``. Keeping the guard in
one place is why ``tests/test_lookahead.py`` has a single function to attack.

Run a self-check from the repo root::

    .venv/bin/python -m src.covariance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    COV_EWMA_HALFLIFE_DAYS,
    COV_SHRINKAGE,
    MAX_CONDITION_NUMBER,
    MIN_EIGENVALUE,
    TRADING_DAYS_PER_YEAR,
)

log = logging.getLogger(__name__)

# Below this, a matrix is treated as asymmetric rather than as floating-point
# noise from an outer product. Generous, because the inputs here are ~1e-4.
_SYMMETRY_TOLERANCE: float = 1e-10


class CovarianceError(ValueError):
    """The covariance input or estimate is unusable.

    Raised rather than returning a degraded matrix, for the same reason
    ``src.data`` raises ``DataUnavailableError``: a quietly wrong covariance
    produces a plausible set of weights and an entirely wrong conclusion.
    """


# ---------------------------------------------------------------------------
# Exponentially-weighted covariance
# ---------------------------------------------------------------------------
def ewma_weights(n_observations: int, halflife: float = COV_EWMA_HALFLIFE_DAYS) -> np.ndarray:
    """Normalised decay weights, oldest observation first, newest heaviest.

    ``halflife`` is in observations (trading days here): an observation that
    many periods old carries half the weight of the newest one. ``float("inf")``
    yields uniform weights, which is what makes the estimator collapse to the
    sample covariance -- see :func:`ewma_covariance`.
    """
    if n_observations < 1:
        raise CovarianceError("need at least one observation to weight")
    if halflife <= 0:
        raise CovarianceError(f"halflife must be positive, got {halflife}")

    # 1/inf is 0.0 and 0.5**0 is 1.0, so an infinite halflife lands on uniform
    # weights through the same expression rather than through a special case.
    decay = 0.5 ** (1.0 / halflife)
    weights = decay ** np.arange(n_observations - 1, -1, -1, dtype="float64")
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise CovarianceError(f"decay weights are degenerate for halflife={halflife}")
    return weights / total


def ewma_covariance(
    returns: pd.DataFrame | np.ndarray,
    halflife: float = COV_EWMA_HALFLIFE_DAYS,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Annualised exponentially-weighted covariance of daily returns.

    Parameters
    ----------
    returns:
        Daily simple returns, one column per asset, oldest row first. Already
        sliced to ``<= t`` by the caller.
    halflife:
        Decay halflife in trading days. ``float("inf")`` gives the unweighted
        sample covariance, and that limit holds
        *exactly*, not approximately, because of the bias correction below.
    periods_per_year:
        Annualisation factor. Daily here; the argument exists so the same
        estimator can be reused on another frequency without a module constant
        lying about it (the reason ``src/metrics.py`` was ported this way).

    Notes
    -----
    Two details that look like decoration and are not:

    * Deviations are taken from the **weighted** mean, not the arithmetic one.
      With a 60-day halflife over a 252-day window the two differ materially in
      a trending market, and using the arithmetic mean inflates the variance of
      whichever sleeve trended hardest -- precisely the sleeve risk parity is
      about to down-weight.
    * The ``1 - sum(w^2)`` divisor is the reliability-weight bias correction,
      the weighted analogue of ``ddof=1``. Without it the estimate is biased low,
      which would understate portfolio volatility, inflate every Sharpe ratio
      downstream, and make the vol-targeting overlay lever *up* into risk it had
      mis-measured. With uniform weights it equals ``(n-1)/n`` and recovers
      ``np.cov(..., ddof=1)`` term for term.
    """
    x = _as_observation_matrix(returns)
    n = x.shape[0]
    if n < 2:
        raise CovarianceError(
            f"need at least 2 observations for a covariance, got {n}; "
            "the rebalance calendar's warmup is too short"
        )

    w = ewma_weights(n, halflife)
    deviations = x - (w @ x)
    cov = (deviations * w[:, None]).T @ deviations

    correction = 1.0 - float((w**2).sum())
    if correction <= 0.0:
        # Reachable only with an absurdly short halflife, where essentially all
        # the weight sits on the newest observation and there is no effective
        # sample left to estimate a second moment from.
        raise CovarianceError(
            f"halflife={halflife} over {n} observations leaves an effective sample "
            f"size of {1.0 / (1.0 - correction):.2f}; the estimate is undefined"
        )
    cov /= correction

    return _symmetrise(cov * periods_per_year)


# ---------------------------------------------------------------------------
# Shrinkage, PSD repair, conditioning
# ---------------------------------------------------------------------------
def shrink_to_diagonal(cov: np.ndarray, intensity: float = COV_SHRINKAGE) -> np.ndarray:
    """Shrink toward a diagonal target: ``(1-d)*cov + d*diag(cov)``.

    The target is the matrix's own diagonal, so **variances survive untouched**
    and only the covariances move. That choice matters here: risk parity's
    weights are driven first by relative volatilities, and a target like
    ``identity * mean_variance`` would distort exactly the quantity the strategy
    keys off. Off-diagonals are where the estimation error concentrates anyway --
    with four assets there are six of them estimated from the same window that
    produced four variances.

    In correlation terms this is simply ``rho -> (1-d)*rho`` for every pair. Two
    consequences worth carrying:

    * It improves conditioning monotonically, which is the practical reason it
      runs before :func:`nearest_psd`.
    * It biases the diversification ratio *upward*, because weaker
      correlations mean a smaller portfolio volatility for the same weights. The
      2022 story is told by the ratio's trend, not its level, so a constant
      shrinkage does not distort it -- but the level should not be quoted as if
      it were unshrunk.
    """
    cov = validate_covariance(cov, "cov")
    if not 0.0 <= intensity <= 1.0:
        raise CovarianceError(f"shrinkage intensity must be in [0, 1], got {intensity}")
    if intensity == 0.0:
        return cov

    target = np.diag(np.diag(cov))
    return _symmetrise((1.0 - intensity) * cov + intensity * target)


def nearest_psd(cov: np.ndarray, min_eigenvalue: float = MIN_EIGENVALUE) -> np.ndarray:
    """Repair a fractionally indefinite covariance by clipping its eigenvalues.

    EWMA weighting plus floating-point arithmetic routinely produces eigenvalues
    of order ``-1e-18`` on matrices that are mathematically PSD. That is enough
    for ``sqrt(w'Cw)`` to return NaN at some weight vector, which surfaces as an
    unexplained solver failure three modules away.

    The repair runs in **correlation space**, which is the part worth reading:

    1. Factor out the standard deviations, ``R = D^-1 C D^-1``.
    2. Clip ``R``'s eigenvalues at ``min_eigenvalue`` and rebuild.
    3. Renormalise ``R``'s diagonal back to exactly 1.
    4. Multiply the standard deviations back in.

    Stage 3 is a congruence transform by a positive diagonal matrix, so it
    **cannot** reintroduce a negative eigenvalue -- unlike the obvious
    alternative of overwriting the diagonal in place after clipping, which
    silently undoes the repair it just performed. The payoff is that asset
    volatilities come out unchanged to floating-point precision (~1e-16
    relative, from the divide-then-multiply round trip) while the correlation
    structure absorbs the whole adjustment, which is the right place to put it:
    the variances were the well-estimated part.

    An asset with zero or non-finite variance has no correlation representation,
    so that case falls back to clipping the covariance directly, which lands the
    asset at a variance of ``min_eigenvalue`` rather than zero and keeps the
    risk-contribution identity well posed instead of dividing by zero.

    Note the limit of that fallback: it fires only on a variance of *exactly*
    zero. A merely tiny positive variance keeps its correlation representation
    and is preserved untouched -- deliberately, since silently inflating a real
    variance to a floor would be a fabricated number. An asset that small breaks
    ERC feasibility rather than PSD-ness, and it is
    ``src.riskparity.solve_erc_with_status`` that detects and raises on it (the
    near-zero-vol degenerate case).
    """
    cov = validate_covariance(cov, "cov")
    if min_eigenvalue < 0.0:
        raise CovarianceError(f"min_eigenvalue must be non-negative, got {min_eigenvalue}")

    variances = np.diag(cov)
    if np.all(np.isfinite(variances)) and np.all(variances > 0.0):
        sd = np.sqrt(variances)
        corr = cov / np.outer(sd, sd)
        repaired = _clip_eigenvalues(corr, min_eigenvalue)

        scale = np.sqrt(np.diag(repaired))
        repaired = repaired / np.outer(scale, scale)
        np.fill_diagonal(repaired, 1.0)
        out = repaired * np.outer(sd, sd)
    else:
        out = _clip_eigenvalues(cov, min_eigenvalue)

    out = _symmetrise(out)

    # Assert the postcondition rather than trusting it. The tolerance is scaled
    # by the matrix's own magnitude because an annualised equity covariance and
    # a daily one differ by 252x and a fixed epsilon would be wrong for one of
    # them.
    smallest = float(np.linalg.eigvalsh(out).min())
    floor = -1e-10 * max(1.0, float(np.abs(out).max()))
    if smallest < floor:
        raise CovarianceError(
            f"PSD repair failed: smallest eigenvalue is {smallest:.3e} after clipping"
        )
    return out


def condition_number(cov: np.ndarray) -> float:
    """Ratio of largest to smallest eigenvalue -- ``inf`` if singular or non-PSD.

    Reported rather than acted on. A high condition number does not stop the ERC
    solver from converging; it means the answer it converged to is not stable
    against small changes in the input, which is a thing the run summary should
    say out loud rather than something to silently correct.

    Deliberately reads the raw (signed) eigenvalues, not their magnitudes.
    ``validate_covariance`` checks square/finite/symmetric only -- the spectrum
    is ``nearest_psd``'s business, not this function's -- so a genuinely
    indefinite matrix can reach here. Taking ``abs()`` first would let a large
    NEGATIVE eigenvalue hide behind a small POSITIVE one (e.g. eigenvalues
    ``[-5.0, 0.01, 1.0, 2.0]`` would report condition number 500, a plausible
    finite number, instead of correctly reporting the matrix as not a valid
    covariance at all). Reading the sign means any non-positive eigenvalue --
    genuinely negative or merely singular -- returns ``inf``, which is what
    "not a valid covariance" should report.
    """
    cov = validate_covariance(cov, "cov")
    eigenvalues = np.linalg.eigvalsh(cov)
    smallest = float(eigenvalues.min())
    if smallest <= 0.0:
        return float("inf")
    return float(eigenvalues.max() / smallest)


@dataclass(frozen=True)
class CovarianceDiagnostics:
    """What the estimator did, for ``BacktestResult.meta``.

    ``min_eigenvalue_before_repair`` and ``condition_number`` answer two
    different questions and are both kept: the first says whether the matrix had
    to be repaired at all, the second says whether the repaired matrix is one a
    solver's answer can be trusted on. A run can be clean on one and alarming on
    the other.
    """

    n_observations: int
    n_assets: int
    halflife: float
    shrinkage: float
    min_eigenvalue_before_repair: float
    min_eigenvalue: float
    condition_number: float
    was_repaired: bool
    is_ill_conditioned: bool


def estimate_covariance_with_diagnostics(
    returns: pd.DataFrame | np.ndarray,
    halflife: float = COV_EWMA_HALFLIFE_DAYS,
    shrinkage: float = COV_SHRINKAGE,
    min_eigenvalue: float = MIN_EIGENVALUE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    max_condition_number: float = MAX_CONDITION_NUMBER,
) -> tuple[np.ndarray, CovarianceDiagnostics]:
    """The full pipeline: EWMA, then shrinkage, then PSD repair.

    Returns the annualised covariance alongside what happened to it. Use
    :func:`estimate_covariance` where only the matrix is wanted; this variant
    exists so the backtest loop can accumulate diagnostics into ``meta`` without
    re-deriving them.

    Shrinkage runs *before* the repair on purpose. It improves conditioning on
    its own, so on most windows it removes the need for any clipping at all, and
    a repair that never fires is a repair that never had to distort anything.
    """
    x = _as_observation_matrix(returns)
    raw = ewma_covariance(x, halflife=halflife, periods_per_year=periods_per_year)
    shrunk = shrink_to_diagonal(raw, shrinkage)

    smallest_before = float(np.linalg.eigvalsh(shrunk).min())
    repaired = nearest_psd(shrunk, min_eigenvalue)
    smallest_after = float(np.linalg.eigvalsh(repaired).min())
    cond = condition_number(repaired)

    was_repaired = smallest_before < min_eigenvalue
    ill_conditioned = cond > max_condition_number

    if was_repaired:
        log.info(
            "Covariance repaired: smallest eigenvalue %.3e -> %.3e (%d observations).",
            smallest_before,
            smallest_after,
            x.shape[0],
        )
    if ill_conditioned:
        log.warning(
            "Covariance condition number %.3e exceeds %.1e -- the solver may converge "
            "to an unstable optimum. Surfaced in meta rather than corrected.",
            cond,
            max_condition_number,
        )

    diagnostics = CovarianceDiagnostics(
        n_observations=int(x.shape[0]),
        n_assets=int(x.shape[1]),
        halflife=float(halflife),
        shrinkage=float(shrinkage),
        min_eigenvalue_before_repair=smallest_before,
        min_eigenvalue=smallest_after,
        condition_number=cond,
        was_repaired=was_repaired,
        is_ill_conditioned=ill_conditioned,
    )
    return repaired, diagnostics


def estimate_covariance(
    returns: pd.DataFrame | np.ndarray,
    halflife: float = COV_EWMA_HALFLIFE_DAYS,
    shrinkage: float = COV_SHRINKAGE,
    min_eigenvalue: float = MIN_EIGENVALUE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Annualised, shrunk, guaranteed-PSD covariance. The project's entry point.

    Signature-compatible with ``src.benchmarks.CovEstimator``, so it can be
    passed straight in as ``cov_estimator=`` and the benchmarks are then judged
    on exactly the covariance the engine uses.
    """
    cov, _ = estimate_covariance_with_diagnostics(
        returns,
        halflife=halflife,
        shrinkage=shrinkage,
        min_eigenvalue=min_eigenvalue,
        periods_per_year=periods_per_year,
    )
    return cov


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _require_chronological(index: pd.Index) -> None:
    """Reject a window whose rows are not oldest-first, or repeat a date.

    :func:`ewma_weights` returns decay weights oldest-first and pairs weight ``k``
    with row ``k`` positionally. Nothing here re-reads the index, so a frame in
    the wrong order produces a perfectly well-formed covariance that weights the
    oldest observations most heavily -- no error, just the wrong matrix.

    Measured on a window that is calm and then violent: reversing the rows reports
    a portfolio volatility of 7.9% against a true 19.8%. Understating covariance
    understates every risk contribution the solver balances and inflates every
    Sharpe downstream, so it fails in the flattering direction.

    ``src.data`` sorts on the way in and the live pipeline cannot reach this; a
    frame assembled elsewhere (a test, a notebook, a ``pd.concat`` in the wrong
    order) can. Mirrored in ``src.overlay._require_chronological``, which guards
    the same assumption for the trailing-vol estimator.
    """
    if not isinstance(index, pd.DatetimeIndex):
        # A positional index carries no chronology to verify.
        return
    if not index.is_monotonic_increasing:
        raise CovarianceError(
            "return window is not in chronological order; the EWMA decay weights "
            "are positional, so out-of-order rows silently mis-weight the estimate "
            "(a reversed window understates portfolio volatility ~2.5x). Sort the "
            "frame before slicing it"
        )
    if index.has_duplicates:
        duplicated = index[index.duplicated()].unique()
        raise CovarianceError(
            f"return window repeats {len(duplicated)} date(s), first "
            f"{duplicated[0].date()}; a duplicated observation is double-weighted "
            "by the decay and is a sign of a concatenated or re-indexed frame"
        )


def _as_observation_matrix(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Validate and coerce a (T, N) return window to a float array."""
    if isinstance(returns, pd.DataFrame):
        _require_chronological(returns.index)
        x = returns.to_numpy(dtype="float64")
    else:
        x = np.asarray(returns, dtype="float64")

    if x.ndim != 2:
        raise CovarianceError(f"returns must be 2-D (observations x assets), got shape {x.shape}")
    if x.shape[1] < 1:
        raise CovarianceError("returns has no columns")
    if not np.isfinite(x).all():
        raise CovarianceError(
            "returns contains NaN or infinite values; src.data raises on these rather "
            "than filling them, so reaching here means the window was built elsewhere"
        )
    return x


def validate_covariance(cov: np.ndarray, name: str = "cov") -> np.ndarray:
    """Coerce to a finite, symmetric, square float array."""
    m = np.asarray(cov, dtype="float64")
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise CovarianceError(f"{name} must be a square matrix, got shape {m.shape}")
    if m.shape[0] < 1:
        raise CovarianceError(f"{name} is empty")
    if not np.isfinite(m).all():
        raise CovarianceError(f"{name} contains NaN or infinite values")

    asymmetry = float(np.abs(m - m.T).max())
    scale = max(1.0, float(np.abs(m).max()))
    if asymmetry > _SYMMETRY_TOLERANCE * scale:
        raise CovarianceError(
            f"{name} is not symmetric (worst |C - C'| = {asymmetry:.3e}); a covariance "
            "that is asymmetric by more than rounding was built wrong"
        )
    return m


def _symmetrise(m: np.ndarray) -> np.ndarray:
    """Average a matrix with its transpose.

    Outer products and eigen-reconstruction leave asymmetries around 1e-17.
    ``np.linalg.eigvalsh`` reads only one triangle, so an unsymmetrised matrix
    would give eigenvalues of a matrix nobody actually holds.
    """
    return 0.5 * (m + m.T)


def _clip_eigenvalues(m: np.ndarray, floor: float) -> np.ndarray:
    """Rebuild ``m`` with every eigenvalue raised to at least ``floor``."""
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetrise(m))
    clipped = np.maximum(eigenvalues, floor)
    if np.array_equal(clipped, eigenvalues):
        return m
    return (eigenvectors * clipped) @ eigenvectors.T


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(20240614)

    # An infinite halflife is the sample covariance.
    sample_data = rng.normal(0.0, 0.01, size=(400, 4))
    limit = ewma_covariance(sample_data, halflife=float("inf"), periods_per_year=1)
    reference = np.cov(sample_data, rowvar=False, ddof=1)
    print(f"halflife -> inf vs np.cov(ddof=1) : max |diff| = "
          f"{np.abs(limit - reference).max():.3e}")

    # PSD even from a deliberately indefinite input.
    broken = np.array(
        [
            [0.0400, 0.0399, 0.0100, 0.0050],
            [0.0399, 0.0400, 0.0100, 0.0050],
            [0.0100, 0.0100, 0.0225, 0.0030],
            [0.0050, 0.0050, 0.0030, 0.0025],
        ]
    )
    broken[0, 1] = broken[1, 0] = 0.0405  # correlation > 1: not a covariance at all
    before = float(np.linalg.eigvalsh(broken).min())
    fixed = nearest_psd(broken)
    print(f"indefinite input                  : min eig {before:.3e} -> "
          f"{float(np.linalg.eigvalsh(fixed).min()):.3e}")
    print(f"  variances preserved to 1e-12    : "
          f"{np.allclose(np.diag(fixed), np.diag(broken), rtol=1e-12, atol=0.0)}")

    # And the composed pipeline on a realistic window.
    daily = rng.normal(0.0, 1.0, size=(252, 4)) @ np.linalg.cholesky(
        np.array(
            [
                [1.00, -0.30, 0.20, -0.10],
                [-0.30, 1.00, -0.05, 0.85],
                [0.20, -0.05, 1.00, 0.10],
                [-0.10, 0.85, 0.10, 1.00],
            ]
        )
    ).T * np.array([0.0110, 0.0085, 0.0120, 0.0035])
    cov, diag = estimate_covariance_with_diagnostics(
        pd.DataFrame(daily, columns=["SPY", "TLT", "DBC", "TIP"])
    )
    print("\nestimate_covariance on 252 synthetic days")
    print(f"  annualised vols   : {np.sqrt(np.diag(cov)).round(4)}")
    print(f"  condition number  : {diag.condition_number:,.1f}")
    print(f"  repaired          : {diag.was_repaired}")
    print(f"  ill-conditioned   : {diag.is_ill_conditioned}")

    # --- the ordering regression --------------------------------------------
    # The decay weights are POSITIONAL: ewma_weights
    # pairs weight k with row k and nothing here re-reads the index, so a window
    # in the wrong order produces a well-formed covariance built on the wrong
    # weighting -- oldest observations weighted most. No error, just the wrong
    # matrix, and it understates risk, which inflates every Sharpe downstream.
    # src.data sorts on the way in; a frame assembled elsewhere need not be.
    print("\nrow ordering is checked, not assumed (regression)")
    index = pd.bdate_range("2006-02-03", periods=300, name="date")
    regime = np.concatenate([np.full(150, 0.003), np.full(150, 0.025)])
    window = pd.DataFrame(
        rng.normal(0.0, regime[:, None] * np.ones((1, 4))),
        index=index,
        columns=["SPY", "TLT", "DBC", "TIP"],
    )
    equal = np.repeat(0.25, 4)
    good = float(np.sqrt(equal @ ewma_covariance(window) @ equal))
    print(f"  correctly ordered window : portfolio vol {good:.2%}")
    for label, frame in (
        ("reversed", window.iloc[::-1]),
        ("shuffled", window.sample(frac=1.0, random_state=1)),
        ("duplicated date", pd.concat([window, window.iloc[[10]]]).sort_index()),
    ):
        try:
            bad = float(np.sqrt(equal @ ewma_covariance(frame) @ equal))
            print(f"  {label:24s} : returned {bad:.2%} -- REGRESSION, should have raised")
        except CovarianceError:
            print(f"  {label:24s} : CovarianceError, as it must")
