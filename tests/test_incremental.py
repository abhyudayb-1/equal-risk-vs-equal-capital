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

Five things are under test (the fifth, step 43, is the realized-risk block --
see its section below), one function each:

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

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src import config, incremental, metrics  # noqa: E402
from src.covariance import estimate_covariance  # noqa: E402
from src.data import DataUnavailableError, daily_returns  # noqa: E402
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


# ---------------------------------------------------------------------------
# Realized risk, before vs after (step 43)
# ---------------------------------------------------------------------------
# Every expected value below is derived by hand from the definition, in plain
# arithmetic, not by calling the function under test a second way.
METRIC_TOLERANCE = 1e-12


def test_historical_var_and_cvar_hand_derived() -> None:
    """r_k = (k - 51) / 1000 for k = 1..100, so the sorted returns are
    -0.050, -0.049, ..., +0.049. The 5% quantile sits at position 0.05 * 99 =
    4.95 of the sorted array, i.e. 95% of the way from sorted[4] = -0.046 to
    sorted[5] = -0.045: VaR = -0.046 + 0.95 * 0.001 = -0.04505. The tail at or
    below it is exactly sorted[0..4] = -0.050 .. -0.046, mean -0.048.
    """
    r = pd.Series((np.arange(1, 101) - 51) / 1000.0)
    check(
        "historical VaR 95% equals the hand-interpolated quantile -0.04505",
        abs(incremental.historical_var(r, 0.95) - (-0.04505)) < METRIC_TOLERANCE,
    )
    check(
        "historical CVaR 95% equals the mean of the five worst returns, -0.048",
        abs(incremental.historical_cvar(r, 0.95) - (-0.048)) < METRIC_TOLERANCE,
    )
    check(
        "CVaR is never better than VaR",
        incremental.historical_cvar(r, 0.95) <= incremental.historical_var(r, 0.95),
    )
    check(
        "VaR on fewer than two observations is NaN, not a number from nothing",
        np.isnan(incremental.historical_var(pd.Series([0.01]))),
    )
    expect_raises(
        "a confidence outside (0, 1) is rejected",
        ValueError,
        lambda: incremental.historical_var(r, 1.0),
    )


def test_downside_deviation_and_sortino_hand_derived() -> None:
    """r = [+1%, -2%, +3%, -1%] at rf = 0: the shortfalls are [0, -.02, 0, -.01],
    mean square (0.0004 + 0.0001) / 4 = 0.000125 over ALL four periods, and
    the mean return is 0.0025.
    """
    r = pd.Series([0.01, -0.02, 0.03, -0.01])
    per_period_downside = math.sqrt((0.0004 + 0.0001) / 4.0)
    annual = math.sqrt(252)
    check(
        "annualised downside deviation matches sqrt(mean shortfall^2) * sqrt(252)",
        abs(incremental.downside_deviation(r, 252, 0.0) - per_period_downside * annual)
        < METRIC_TOLERANCE,
    )
    check(
        "Sortino matches mean excess / downside deviation * sqrt(252)",
        abs(incremental.sortino_ratio(r, 252, 0.0) - 0.0025 / per_period_downside * annual)
        < METRIC_TOLERANCE,
    )

    no_downside = pd.Series([0.001, 0.002, 0.003, 0.002])
    check(
        "a series with no losing period has zero downside deviation",
        incremental.downside_deviation(no_downside, 252, 0.0) == 0.0,
    )
    check(
        "...and an UNDEFINED Sortino (NaN), not a huge finite ratio",
        np.isnan(incremental.sortino_ratio(no_downside, 252, 0.0)),
    )


def test_drawdown_duration_and_calmar_hand_derived() -> None:
    """r = [+10%, -5%, -5%, +5%, +10%]: the curve is 1.1, 1.045, 0.99275,
    1.0424, 1.1466. Periods 2-4 sit below the 1.1 peak (three of them) and
    period 5 makes a new high, so the longest stretch is 3 and it recovered.
    The trough is 1.1 * 0.95^2, a drawdown of 1 - 0.9025 = 9.75% exactly.
    """
    r = pd.Series([0.10, -0.05, -0.05, 0.05, 0.10])
    days, still_open = incremental.drawdown_duration(r)
    check("longest drawdown lasts 3 periods on the hand-built path", days == 3)
    check("...and it has recovered by the end of the series", still_open is False)

    days, still_open = incremental.drawdown_duration(r.iloc[:4])
    check("cutting the series before the recovery keeps the 3-period stretch", days == 3)
    check("...and flags it as still open at the window end", still_open is True)

    days, still_open = incremental.drawdown_duration(pd.Series([-0.05, -0.05]))
    check(
        "a series that starts below its initial value is underwater from the first "
        "period (the peak is floored at the 1.0 starting value, as in metrics.py)",
        (days, still_open) == (2, True),
    )

    # ppy = 5 makes the five periods exactly one year, so CAGR is just growth - 1.
    growth = 1.10 * 0.95 * 0.95 * 1.05 * 1.10
    expected = (growth - 1.0) / (1.0 - 0.95 * 0.95)
    check(
        "Calmar equals CAGR over the 9.75% max drawdown",
        abs(incremental.calmar_ratio(r, 5) - expected) < METRIC_TOLERANCE,
    )
    check(
        "Calmar is NaN for a series that never draws down",
        np.isnan(incremental.calmar_ratio(pd.Series([0.001, 0.002, 0.003]), 252)),
    )


def _mixed_books_match_definition(returns: pd.DataFrame, baseline_weights: np.ndarray) -> bool:
    """True iff every ``+w%`` column equals ``(1-w) * baseline + w * candidate``,
    recomputed here from the definition. Factored out so the mutation test
    below can run the same predicate against a broken ``_mixed_weights``.
    """
    books = incremental.simulate_mixed_books(
        returns, CANDIDATE, baseline_weights, config.UNIVERSE
    )
    baseline_series = returns[list(config.UNIVERSE)].to_numpy() @ baseline_weights
    candidate_series = returns[CANDIDATE].to_numpy()
    for w in config.INCREMENTAL_CANDIDATE_WEIGHTS:
        expected = (1.0 - w) * baseline_series + w * candidate_series
        if not np.allclose(books[f"+{w * 100:g}%"].to_numpy(), expected, atol=METRIC_TOLERANCE):
            return False
    return True


def test_simulate_mixed_books_identities() -> None:
    returns = _candidate_and_universe_returns()
    baseline = incremental.baseline_erc(returns, config.UNIVERSE).weights
    books = incremental.simulate_mixed_books(returns, CANDIDATE, baseline, config.UNIVERSE)

    check(
        "one column per book, in order: baseline, each +w%, then the stock alone",
        list(books.columns) == ["baseline", "+1%", "+5%", "+10%", "stock alone"],
    )
    check(
        "every mixed book equals (1-w) * baseline + w * candidate, from the definition",
        _mixed_books_match_definition(returns, baseline),
    )
    check(
        "the stock-alone column is the candidate's raw returns, untouched",
        bool(np.array_equal(books["stock alone"].to_numpy(), returns[CANDIDATE].to_numpy())),
    )

    zero = incremental.simulate_mixed_books(
        returns, CANDIDATE, baseline, config.UNIVERSE, candidate_weights=(0.0,)
    )
    check(
        "a 0% position reproduces the baseline book exactly",
        bool(np.allclose(zero["+0%"].to_numpy(), zero["baseline"].to_numpy(), atol=METRIC_TOLERANCE)),
    )

    # Positive control for the check below: on real (different) returns the
    # mixes must NOT collapse onto the baseline, or "they are identical" proves
    # nothing about a candidate that genuinely is the baseline.
    check(
        "positive control: with a real candidate the +10% book differs from the baseline",
        float(np.abs(books["+10%"] - books["baseline"]).max()) > 1e-4,
    )
    clone = returns.copy()
    clone[CANDIDATE] = returns[list(config.UNIVERSE)].to_numpy() @ baseline
    clone_books = incremental.simulate_mixed_books(clone, CANDIDATE, baseline, config.UNIVERSE)
    check(
        "a candidate that IS the baseline book leaves every mixed book identical to it",
        all(
            np.allclose(clone_books[c].to_numpy(), clone_books["baseline"].to_numpy(),
                        atol=METRIC_TOLERANCE)
            for c in ("+1%", "+5%", "+10%")
        ),
    )

    expect_raises(
        "empty candidate_weights is rejected",
        ValueError,
        lambda: incremental.simulate_mixed_books(
            returns, CANDIDATE, baseline, config.UNIVERSE, candidate_weights=()
        ),
    )
    expect_raises(
        "duplicate candidate_weights are rejected (they would collide as columns)",
        ValueError,
        lambda: incremental.simulate_mixed_books(
            returns, CANDIDATE, baseline, config.UNIVERSE, candidate_weights=(0.05, 0.05)
        ),
    )
    expect_raises(
        "a baseline with the wrong number of weights is rejected",
        ValueError,
        lambda: incremental.simulate_mixed_books(
            returns, CANDIDATE, baseline[:3], config.UNIVERSE
        ),
    )


def test_mixed_book_definition_check_has_teeth() -> None:
    """Break ``_mixed_weights`` (drop the ``(1 - w)`` scaling, so the five
    weights sum to more than 1) and confirm the definition check that passed
    above now FAILS -- the house technique: a check that cannot fail proves
    nothing.
    """
    returns = _candidate_and_universe_returns()
    baseline = incremental.baseline_erc(returns, config.UNIVERSE).weights

    real = incremental._mixed_weights
    incremental._mixed_weights = lambda b, w: np.concatenate(([w], np.asarray(b, dtype="float64")))
    try:
        broken_passes = _mixed_books_match_definition(returns, baseline)
    finally:
        incremental._mixed_weights = real
    check("with the (1 - w) scaling removed the definition check fails", not broken_passes)
    check("...and it passes again once the real function is restored",
          _mixed_books_match_definition(returns, baseline))


def test_realized_section_in_the_position_report() -> None:
    real_fetch_prices = incremental.fetch_prices
    incremental.fetch_prices = lambda tickers, start=None, end=None, refresh=False: make_fake_prices(
        tickers=tuple(tickers)
    )
    try:
        report = incremental.analyze_position(CANDIDATE)
    finally:
        incremental.fetch_prices = real_fetch_prices

    realized = report.realized
    check(
        "the report's realized table has one row per book",
        list(realized.index) == ["baseline", "+1%", "+5%", "+10%", "stock alone"],
    )
    numeric_keys = [key for _, key, _ in incremental._REALIZED_ROWS]
    check(
        "every reported metric is finite for every book on a well-behaved window",
        bool(np.isfinite(realized[numeric_keys].to_numpy(dtype="float64")).all()),
    )
    # The window analyze_position itself analysed: prices -> daily returns ->
    # the last lookback_days rows (make_fake_prices is deterministic).
    window = daily_returns(make_fake_prices(tickers=(CANDIDATE, *config.UNIVERSE))).iloc[
        -config.INCREMENTAL_LOOKBACK_DAYS:
    ]
    baseline_series = window[list(config.UNIVERSE)].to_numpy() @ report.baseline_solution.weights
    check(
        "the table's max drawdown agrees with metrics.max_drawdown on the same series",
        abs(
            realized.at["baseline", "max_drawdown"]
            - metrics.max_drawdown(pd.Series(baseline_series))
        )
        < METRIC_TOLERANCE,
    )

    text = incremental.format_position_report(report)
    check("the printed report has the realized before/after section",
          "Realized risk, before vs after" in text)
    check("...with a change-vs-baseline block", "Change vs baseline" in text)
    check("...and the in-sample caveat", "in-sample replay" in text)

    check("an undefined metric prints as -- not as a number",
          incremental._format_metric(float("nan"), "ratio") == "--")
    check("an undefined change prints as -- too",
          incremental._format_delta(float("nan"), "pct") == "--")
    check("a percentage-point change carries its sign",
          incremental._format_delta(-0.0123, "pct") == "-1.23pp")


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

    test_historical_var_and_cvar_hand_derived()
    test_downside_deviation_and_sortino_hand_derived()
    test_drawdown_duration_and_calmar_hand_derived()
    test_simulate_mixed_books_identities()
    test_mixed_book_definition_check_has_teeth()
    test_realized_section_in_the_position_report()

    print(f"\n{_checks} checks passed")
