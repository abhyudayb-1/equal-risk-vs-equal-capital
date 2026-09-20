"""Self-checks for the incremental position analysis (TASKS.md step 39).

Runs standalone with no test framework, matching the house style::

    .venv/bin/python tests/test_incremental.py

Everything here is **offline** -- ``src.incremental`` is a network-touching
module by design (it fetches a live candidate ticker), but that is exercised
once, deliberately, by ``src.incremental``'s own ``__main__`` self-check
(steps 37-38's "done when"), not by this suite. Every test below either calls
``src.incremental``'s pure functions directly on a fixture return frame, or
monkeypatches ``fetch_prices`` the same way the module's own self-check
attacks the ``INCREMENTAL_MIN_HISTORY_DAYS`` guard -- so the whole file runs
with no network access, matching every other suite in this project.

Four things are under test, one function each:

* **Perturbation weights sum to 1**, at every candidate weight in
  ``INCREMENTAL_CANDIDATE_WEIGHTS`` -- ``perturbation_scan``'s whole
  construction (scale the baseline by ``(1-w)``, insert the candidate at
  ``w``) is built to guarantee this algebraically; this is the regression
  that would catch a broken column alignment doing it anyway.
* **The compensation-test formula**, cross-checked against an independent,
  from-scratch recomputation of ``SR_i``, ``SR_p`` and ``rho(i, p)`` -- the
  same "derive it a second way, from the maths, not from the module under
  test" approach ``tests/test_riskparity.py``'s oracle uses for the ERC
  solver.
* **The 5-asset re-solve's limiting case**: a candidate built as an exact
  duplicate of an existing sleeve must land on the SAME weight as that sleeve
  at the equal-risk solution, for every sleeve in the universe -- the two
  statistically identical assets split what would otherwise be one sleeve's
  slot in the equal-risk division evenly between them.
* **Correlation-table correctness**: ``compensation_test``'s
  ``correlation_to_sleeves`` cross-checked against an independent per-column
  loop, plus the boundary case (a candidate identical to a sleeve must
  correlate with it at exactly 1.0).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src import config, incremental  # noqa: E402
from src.covariance import estimate_covariance  # noqa: E402
from src.data import DataUnavailableError  # noqa: E402
from src.riskparity import portfolio_volatility, risk_contributions  # noqa: E402
from tests.fixtures import make_fake_prices, make_fake_returns  # noqa: E402

_checks = 0

# Algebraic identities (weights/RC fractions summing to 1) are exact algebra,
# held to floating-point noise. The duplicated-sleeve symmetry runs through
# SLSQP first, so it is held to a looser, still-tight bar -- see
# test_five_asset_resolve_splits_a_duplicated_sleeve's docstring for the
# empirical margin (observed exactly 0 to 5.6e-17 across all four sleeves).
IDENTITY_TOLERANCE = 1e-9
SYMMETRY_TOLERANCE = 1e-6


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
# Fixtures
# ---------------------------------------------------------------------------
# GLD is the fifth ticker tests/fixtures.py already carries parameters for --
# using it as the "candidate" throughout means every check here exercises the
# real analyze_position() input shape (a ticker outside UNIVERSE) rather than
# a value that only happens to work in isolation.
CANDIDATE = config.GOLD_TICKER


def _candidate_and_universe_returns() -> pd.DataFrame:
    return make_fake_returns(tickers=(CANDIDATE, *config.UNIVERSE))


# ---------------------------------------------------------------------------
# Perturbation scan
# ---------------------------------------------------------------------------
def test_perturbation_weights_sum_to_one() -> None:
    returns = _candidate_and_universe_returns()
    baseline = incremental.baseline_erc(returns, config.UNIVERSE)
    scan = incremental.perturbation_scan(returns, CANDIDATE, baseline.weights, config.UNIVERSE)

    check(
        "perturbation scan has one row per INCREMENTAL_CANDIDATE_WEIGHTS entry",
        list(scan.index) == list(config.INCREMENTAL_CANDIDATE_WEIGHTS),
    )

    weight_columns = [c for c in scan.columns if c.startswith("weight_")]
    check(
        "perturbation scan carries one weight_ column per asset (candidate + universe)",
        len(weight_columns) == 1 + len(config.UNIVERSE),
    )

    row_sums = scan[weight_columns].sum(axis=1).to_numpy()
    check(
        "the five weights sum to 1.0 at every candidate weight",
        bool(np.allclose(row_sums, 1.0, atol=IDENTITY_TOLERANCE)),
    )
    check(
        "the candidate's own weight column equals the requested candidate_weight exactly",
        bool(np.allclose(scan[f"weight_{CANDIDATE}"].to_numpy(), scan.index.to_numpy())),
    )


def test_perturbation_risk_contribution_fractions_sum_to_one() -> None:
    """The Euler identity holds at every scan row, not just at the ERC solution.

    ``sum(RC_i) == sigma_p`` exactly for ANY weight vector (riskparity.py's
    identity, step 16) -- the perturbation scan's weights are not an ERC
    solution once the candidate is inserted at a fixed w, so this is
    incremental.py's own wiring being checked, not a re-test of the identity
    itself (tests/test_riskparity.py already owns that).
    """
    returns = _candidate_and_universe_returns()
    baseline = incremental.baseline_erc(returns, config.UNIVERSE)
    scan = incremental.perturbation_scan(returns, CANDIDATE, baseline.weights, config.UNIVERSE)

    rc_columns = [c for c in scan.columns if c.startswith("rc_fraction_")]
    row_sums = scan[rc_columns].sum(axis=1).to_numpy()
    check(
        "risk-contribution fractions sum to 1.0 at every candidate weight",
        bool(np.allclose(row_sums, 1.0, atol=IDENTITY_TOLERANCE)),
    )


# ---------------------------------------------------------------------------
# The compensation test
# ---------------------------------------------------------------------------
def test_compensation_test_formula_regression() -> None:
    """compensation_test()'s outputs against an independent recomputation.

    Recomputed here with plain numpy on the raw arrays -- no call to
    ``metrics.sharpe_ratio`` or ``pandas.Series.corr`` -- so a bug shared
    between this file and ``src/incremental.py`` (e.g. both using the wrong
    ddof, or a mismatched risk-free rate) would not cancel out.
    """
    returns = _candidate_and_universe_returns()
    baseline = incremental.baseline_erc(returns, config.UNIVERSE)
    result = incremental.compensation_test(returns, CANDIDATE, baseline.weights, config.UNIVERSE)

    weights = np.asarray(baseline.weights, dtype="float64")
    sleeve_matrix = returns[list(config.UNIVERSE)].to_numpy()
    portfolio = sleeve_matrix @ weights
    candidate = returns[CANDIDATE].to_numpy()

    rf_annual = config.ANNUAL_RISK_FREE_RATE
    ppy = config.TRADING_DAYS_PER_YEAR
    rf_period = (1.0 + rf_annual) ** (1.0 / ppy) - 1.0

    sr_i = (candidate.mean() - rf_period) / candidate.std(ddof=1) * np.sqrt(ppy)
    sr_p = (portfolio.mean() - rf_period) / portfolio.std(ddof=1) * np.sqrt(ppy)
    rho = np.corrcoef(candidate, portfolio)[0, 1]
    threshold = sr_p * rho

    check(
        "candidate_sharpe matches an independent recomputation",
        abs(result.candidate_sharpe - sr_i) < IDENTITY_TOLERANCE,
    )
    check(
        "portfolio_sharpe matches an independent recomputation",
        abs(result.portfolio_sharpe - sr_p) < IDENTITY_TOLERANCE,
    )
    check(
        "correlation_to_portfolio matches an independent recomputation",
        abs(result.correlation_to_portfolio - rho) < IDENTITY_TOLERANCE,
    )
    check(
        "correlation_threshold is exactly portfolio_sharpe * correlation_to_portfolio",
        abs(result.correlation_threshold - threshold) < IDENTITY_TOLERANCE,
    )
    check(
        "passes is exactly (candidate_sharpe > correlation_threshold)",
        result.passes == bool(sr_i > threshold),
    )


def test_compensation_test_boundary_is_not_arbitrarily_classified() -> None:
    """``passes`` is a strict inequality, so a candidate sitting exactly on
    the threshold must read as NOT compensated -- an off-by-one here (``>=``
    instead of ``>``) would silently reclassify the boundary case, which the
    marginal-improvement derivation (PLAN.md) does not support: at equality
    the derivative of Sharpe at eps=0 is exactly zero, i.e. no improvement.

    Engineered to land exactly on the boundary rather than merely close to
    it: a candidate built as the baseline portfolio's OWN return series (the
    weighted sum ``compensation_test`` itself computes internally) has
    ``correlation_to_portfolio == 1.0`` and ``candidate_sharpe ==
    portfolio_sharpe`` exactly, so ``correlation_threshold`` and
    ``candidate_sharpe`` coincide to floating-point precision -- this is the
    real ``compensation_test`` function under test, not a hand-built
    ``CompensationTest`` standing in for it.
    """
    returns = make_fake_returns(tickers=config.UNIVERSE).copy()
    baseline = incremental.baseline_erc(returns, config.UNIVERSE)
    weights = pd.Series(np.asarray(baseline.weights, dtype="float64"), index=config.UNIVERSE)
    returns["MIRROR"] = returns[list(config.UNIVERSE)].mul(weights, axis=1).sum(axis=1)

    result = incremental.compensation_test(returns, "MIRROR", baseline.weights, config.UNIVERSE)

    check(
        "a candidate identical to the portfolio correlates with it at exactly 1.0",
        abs(result.correlation_to_portfolio - 1.0) < 1e-12,
    )
    check(
        "such a candidate's Sharpe sits exactly on correlation_threshold",
        abs(result.candidate_sharpe - result.correlation_threshold) < 1e-9,
    )
    check(
        "sitting exactly on the threshold does not pass (strict inequality, not >=)",
        result.passes is False,
    )


# ---------------------------------------------------------------------------
# The 5-asset re-solve's limiting case
# ---------------------------------------------------------------------------
def test_five_asset_resolve_splits_a_duplicated_sleeve() -> None:
    """A candidate identical to sleeve j must land on j's exact weight.

    The 5x5 covariance built from ``[duplicate, *UNIVERSE]`` is invariant
    under swapping the duplicate and sleeve j's positions (their rows/columns
    are numerically identical, since the return series are), and the
    ``naive_inv_vol`` warm start solve_erc_with_status uses is a function of
    the diagonal only -- also identical for the two -- so the whole solve
    path stays symmetric under that swap as long as it converges on the
    first attempt (no asymmetric random restart). Measured across all four
    sleeves the observed gap is 0 to 5.6e-17 (floating-point noise from
    solving, not from the construction), so this is held to a bar six orders
    tighter than would be needed to merely detect an unequal split.
    """
    for duplicated in config.UNIVERSE:
        returns = make_fake_returns(tickers=config.UNIVERSE).copy()
        returns["DUP"] = returns[duplicated]

        solution = incremental.resolve_five_asset_erc(returns, "DUP", config.UNIVERSE)
        weights = pd.Series(solution.weights, index=["DUP", *config.UNIVERSE])

        check(
            f"5-asset re-solve converges with a duplicate of {duplicated}",
            solution.converged,
        )
        check(
            f"the duplicate's weight matches {duplicated}'s weight to {SYMMETRY_TOLERANCE:.0e}",
            abs(weights["DUP"] - weights[duplicated]) < SYMMETRY_TOLERANCE,
        )

        cov = estimate_covariance(returns[["DUP", *config.UNIVERSE]])
        rc_fractions = risk_contributions(weights.to_numpy(), cov) / portfolio_volatility(
            weights.to_numpy(), cov
        )
        check(
            f"every asset's risk-contribution fraction sits at 1/5 (duplicate of {duplicated})",
            bool(np.allclose(rc_fractions, 1.0 / 5.0, atol=1e-4)),
        )


# ---------------------------------------------------------------------------
# Correlation-table correctness
# ---------------------------------------------------------------------------
def test_correlation_to_sleeves_matches_independent_computation() -> None:
    returns = _candidate_and_universe_returns()
    baseline = incremental.baseline_erc(returns, config.UNIVERSE)
    result = incremental.compensation_test(returns, CANDIDATE, baseline.weights, config.UNIVERSE)

    # A per-column loop with plain numpy, rather than the vectorised
    # ``DataFrame.corrwith`` src/incremental.py uses -- a different code path
    # computing the same quantity, so a bug in the vectorised call would not
    # be repeated here.
    candidate_series = returns[CANDIDATE].to_numpy()
    for sleeve in config.UNIVERSE:
        independent = float(np.corrcoef(returns[sleeve].to_numpy(), candidate_series)[0, 1])
        check(
            f"correlation_to_sleeves[{sleeve}] matches an independent per-column computation",
            abs(result.correlation_to_sleeves[sleeve] - independent) < IDENTITY_TOLERANCE,
        )

    check(
        "correlation_to_sleeves is indexed by the universe, in order",
        list(result.correlation_to_sleeves.index) == list(config.UNIVERSE),
    )


def test_correlation_to_an_identical_sleeve_is_exactly_one() -> None:
    """The boundary case: a candidate built as an exact copy of a sleeve must
    correlate with it at exactly 1.0, not merely close to it.
    """
    returns = make_fake_returns(tickers=config.UNIVERSE).copy()
    returns["DUP"] = returns[config.EQUITY_TICKER]
    baseline = incremental.baseline_erc(returns, config.UNIVERSE)
    result = incremental.compensation_test(returns, "DUP", baseline.weights, config.UNIVERSE)

    check(
        "a duplicated sleeve's correlation to itself is exactly 1.0",
        abs(result.correlation_to_sleeves[config.EQUITY_TICKER] - 1.0) < 1e-12,
    )


# ---------------------------------------------------------------------------
# Input validation and the offline end-to-end path
# ---------------------------------------------------------------------------
def test_analyze_position_rejects_a_candidate_already_in_universe() -> None:
    expect_raises(
        "analyze_position rejects a candidate that is already a sleeve",
        ValueError,
        lambda: incremental.analyze_position(config.EQUITY_TICKER),
    )


def test_analyze_position_rejects_min_history_above_lookback() -> None:
    expect_raises(
        "analyze_position rejects min_history_days > lookback_days",
        ValueError,
        lambda: incremental.analyze_position(
            CANDIDATE, lookback_days=100, min_history_days=200
        ),
    )


def test_analyze_position_end_to_end_offline() -> None:
    """The full step 37-38 pipeline, wired together, with no network access.

    ``fetch_prices`` is monkeypatched to the fixture price generator -- the
    same technique ``src.incremental``'s own self-check uses to attack the
    MIN_HISTORY_DAYS guard deterministically -- so this exercises
    ``analyze_position``'s actual wiring (every field of ``IncrementalReport``
    populated and internally consistent) without touching yfinance.
    """
    real_fetch_prices = incremental.fetch_prices

    def _fake_fetch_prices(tickers, start=None, end=None, refresh=False):
        return make_fake_prices(tickers=tuple(tickers))

    incremental.fetch_prices = _fake_fetch_prices
    try:
        report = incremental.analyze_position(CANDIDATE)
    finally:
        incremental.fetch_prices = real_fetch_prices

    check("offline report resolves the requested candidate", report.candidate == CANDIDATE)
    check(
        "offline report's baseline weights sum to 1.0",
        abs(report.baseline_weights.sum() - 1.0) < IDENTITY_TOLERANCE,
    )
    check(
        "offline report's 5-asset weights sum to 1.0",
        abs(report.five_asset_weights.sum() - 1.0) < IDENTITY_TOLERANCE,
    )
    check(
        "offline report's weight_diff is indexed by the universe only",
        list(report.weight_diff.index) == list(config.UNIVERSE),
    )
    check(
        "offline report carries a CompensationTest",
        isinstance(report.compensation, incremental.CompensationTest),
    )
    check(
        "offline report's window used the full target lookback "
        "(the fixture has ample history)",
        report.n_observations == config.INCREMENTAL_LOOKBACK_DAYS,
    )


def test_fetch_window_raises_on_thin_candidate_history() -> None:
    """The MIN_HISTORY_DAYS guard, re-verified from this suite rather than
    only from ``src.incremental``'s own self-check -- a future refactor could
    change the self-check without anyone re-running it, but this file runs
    every time.
    """
    real_fetch_prices = incremental.fetch_prices

    def _thin_fetch_prices(tickers, start=None, end=None, refresh=False):
        n = config.INCREMENTAL_MIN_HISTORY_DAYS // 2
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
        rng = np.random.default_rng(0)
        levels = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=(n, len(tickers))), axis=0)
        return pd.DataFrame(levels, index=idx, columns=list(tickers))

    incremental.fetch_prices = _thin_fetch_prices
    try:
        expect_raises(
            "analyze_position raises DataUnavailableError on a too-thin candidate",
            DataUnavailableError,
            lambda: incremental.analyze_position(CANDIDATE),
        )
    finally:
        incremental.fetch_prices = real_fetch_prices


if __name__ == "__main__":
    test_perturbation_weights_sum_to_one()
    test_perturbation_risk_contribution_fractions_sum_to_one()

    test_compensation_test_formula_regression()
    test_compensation_test_boundary_is_not_arbitrarily_classified()

    test_five_asset_resolve_splits_a_duplicated_sleeve()

    test_correlation_to_sleeves_matches_independent_computation()
    test_correlation_to_an_identical_sleeve_is_exactly_one()

    test_analyze_position_rejects_a_candidate_already_in_universe()
    test_analyze_position_rejects_min_history_above_lookback()
    test_analyze_position_end_to_end_offline()
    test_fetch_window_raises_on_thin_candidate_history()

    print(f"\n{_checks} checks passed")
