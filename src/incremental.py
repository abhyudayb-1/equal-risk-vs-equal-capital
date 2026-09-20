"""Incremental position analysis (Phase 6, steps 37-38): what happens if the
existing four-sleeve risk-parity book adds one more stock.

A single stock is not a risk-parity sleeve -- it carries idiosyncratic/sector
risk and no independent structural risk premium the way an asset class does,
so it cannot "compete" against SPY/TLT/DBC/TIP for a slot in the
equal-risk-contribution book. The question this module answers is narrower and
is the one a risk desk actually asks when someone proposes a new position:
**given the book we already hold, what does adding this one stock do to it?**
See PLAN.md's "Phase 6" section for the full derivation; this docstring only
carries what a reader needs to trust the numbers.

**Point-in-time, not a second backtest.** Phases 0-5 answer "how would this
strategy have performed since 2006." This module answers a different
question -- "what does adding this position look like *right now*" -- on a
trailing window ending today (``INCREMENTAL_LOOKBACK_DAYS``, ~3 years) fed
through the same ``covariance.estimate_covariance`` the core engine already
uses. No new estimator, no new statistical assumption: the trade this phase
makes is a shorter window in exchange for working on any ticker, including
ones with far less than the 2006-to-present history the core universe has.
The report always says which kind of analysis it is.

**Three pieces of math, all built from primitives ``covariance.py`` and
``riskparity.py`` already validate:**

1. :func:`perturbation_scan` -- scale the baseline 4-asset ERC weights by
   ``(1 - w)`` and add the candidate at ``w`` for each of
   ``INCREMENTAL_CANDIDATE_WEIGHTS``, reading the candidate's own risk
   contribution and the portfolio vol at each. A perturbation, not a
   re-optimisation: "what if I just add this much," which is how a position
   actually gets sized.
2. :func:`resolve_five_asset_erc` -- the unconstrained 5-asset
   ``solve_erc_with_status`` re-solve, diffed against the baseline. A
   different question from (1): what would equal risk contribution actually
   assign the stock, and how much does it reshuffle the other four.
3. The compensation test, ``SR_i > SR_p * rho(i, p)`` -- step 38, not yet in
   this file.

Run a self-check from the repo root (this one, unlike the seven Phase 0-5
module self-checks, touches the network for its live-ticker demonstration --
there is no way to demonstrate "analyse a real stock" without it)::

    .venv/bin/python -m src.incremental
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    INCREMENTAL_CANDIDATE_WEIGHTS,
    INCREMENTAL_LOOKBACK_DAYS,
    INCREMENTAL_MIN_HISTORY_DAYS,
    UNIVERSE,
)
from .covariance import estimate_covariance
from .data import DataUnavailableError, daily_returns, fetch_prices
from .riskparity import ERCSolution, portfolio_volatility, risk_contributions, solve_erc_with_status

log = logging.getLogger(__name__)

# Calendar-days-per-trading-day ratio used to size the fetch window, padded
# generously above the raw weekday ratio (7/5 = 1.4) so the ~9-10 NYSE holidays
# a year cannot leave the fetch short of the trading days actually wanted --
# the window is trimmed to the exact row count wanted afterward regardless, so
# overshooting here costs nothing but a slightly larger download.
_CALENDAR_DAYS_PER_TRADING_DAY: float = 1.6
_CALENDAR_BUFFER_DAYS: int = 20


def _window_start(trading_days: int) -> str:
    """A calendar start date comfortably covering ``trading_days`` of trading."""
    calendar_days = math.ceil(trading_days * _CALENDAR_DAYS_PER_TRADING_DAY) + _CALENDAR_BUFFER_DAYS
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=calendar_days)
    return start.strftime("%Y-%m-%d")


def _fetch_window(
    candidate: str,
    universe: tuple[str, ...],
    lookback_days: int,
    min_history_days: int,
    refresh: bool,
) -> pd.DataFrame:
    """Daily returns for ``[candidate, *universe]`` on the trailing window.

    Fetches candidate and universe **together**, so the on-disk price cache
    (shared with the core engine's ``fetch_prices(UNIVERSE)`` calls) only ever
    gains the candidate's column rather than being overwritten down to it --
    ``cache.write_frame`` replaces the whole file, and a call for the candidate
    alone would silently wipe out the cached SPY/TLT/DBC/TIP history the next
    ``run_backtest.py`` run expects to find.

    Tries the full ``lookback_days`` window first, which is the common case:
    most tickers anyone would ask about have traded longer than the ~3-year
    lookback. ``fetch_prices`` fails loudly when a ticker's history starts
    materially later than the requested window (``data._validate``'s
    late-start guard) -- that is also exactly the signature of a recent IPO,
    so on that specific failure the fetch is retried against the shorter
    ``min_history_days`` window instead of surfacing the first attempt's
    error. A candidate younger than the ideal lookback but with enough history
    to trust a covariance estimate still gets analysed, on a window sized to
    what it actually has -- which is the point of ``INCREMENTAL_MIN_HISTORY_DAYS``
    existing as a *floor* distinct from ``INCREMENTAL_LOOKBACK_DAYS``, the
    *target*.
    """
    tickers = (candidate, *universe)
    try:
        prices = fetch_prices(tickers, start=_window_start(lookback_days), refresh=refresh)
    except DataUnavailableError as exc:
        log.info(
            "%s does not cover the full %d-day lookback (%s); retrying against "
            "the %d-day minimum-history window.",
            candidate, lookback_days, exc, min_history_days,
        )
        try:
            prices = fetch_prices(
                tickers, start=_window_start(min_history_days), refresh=refresh
            )
        except DataUnavailableError as exc2:
            raise DataUnavailableError(
                f"{candidate} does not have the {min_history_days} trading days of "
                f"recent history INCREMENTAL_MIN_HISTORY_DAYS requires: {exc2}"
            ) from exc2

    returns = daily_returns(prices)
    if len(returns) < min_history_days:
        raise DataUnavailableError(
            f"{candidate} has only {len(returns)} trading days of history "
            f"overlapping the universe's, fewer than "
            f"INCREMENTAL_MIN_HISTORY_DAYS={min_history_days}; too thin to trust "
            f"a trailing-window covariance estimate"
        )

    window = returns.iloc[-lookback_days:]
    if len(window) < lookback_days:
        log.warning(
            "%s: using a %d-day window, shorter than the %d-day target lookback "
            "-- this is a point-in-time snapshot on whatever history the "
            "candidate has, not the full INCREMENTAL_LOOKBACK_DAYS estimate.",
            candidate, len(window), lookback_days,
        )
    return window


# ---------------------------------------------------------------------------
# Step 37 -- baseline snapshot, perturbation scan, the 5-asset re-solve
# ---------------------------------------------------------------------------
def baseline_erc(returns: pd.DataFrame, universe: tuple[str, ...] = UNIVERSE) -> ERCSolution:
    """The existing book's equal-risk-contribution weights on this window.

    Reuses ``covariance.estimate_covariance`` and ``riskparity.solve_erc_with_status``
    unchanged -- the same primitives the core engine solves at every rebalance,
    just on a shorter, more recent window instead of everything at or before a
    backtest date.
    """
    cov = estimate_covariance(returns[list(universe)])
    return solve_erc_with_status(cov)


def perturbation_scan(
    returns: pd.DataFrame,
    candidate: str,
    baseline_weights: np.ndarray,
    universe: tuple[str, ...] = UNIVERSE,
    candidate_weights: tuple[float, ...] = INCREMENTAL_CANDIDATE_WEIGHTS,
) -> pd.DataFrame:
    """Add the candidate at each fixed weight and read what it costs and does.

    For each ``w`` in ``candidate_weights``, the baseline weights are scaled by
    ``(1 - w)`` and the candidate is inserted at ``w``: capital weights for the
    existing four sleeves shrink proportionally rather than the candidate
    displacing any one of them, so the five weights sum to
    ``w + (1-w) * sum(baseline_weights) == 1`` exactly (``baseline_weights``
    already sums to 1, from :func:`baseline_erc`).

    This is a perturbation, not a re-optimisation -- :func:`resolve_five_asset_erc`
    is the re-optimisation, and the two answer different questions on purpose
    (see the module docstring).
    """
    columns = [candidate, *universe]
    cov = estimate_covariance(returns[columns])

    rows = []
    for w in candidate_weights:
        weights = np.concatenate(([w], np.asarray(baseline_weights, dtype="float64") * (1.0 - w)))
        sigma_p = portfolio_volatility(weights, cov)
        rc_fractions = risk_contributions(weights, cov) / sigma_p
        row = {"candidate_weight": w, "portfolio_vol": sigma_p}
        for name, weight_value, rc_value in zip(columns, weights, rc_fractions):
            row[f"weight_{name}"] = weight_value
            row[f"rc_fraction_{name}"] = rc_value
        rows.append(row)
    return pd.DataFrame(rows).set_index("candidate_weight")


def resolve_five_asset_erc(
    returns: pd.DataFrame, candidate: str, universe: tuple[str, ...] = UNIVERSE
) -> ERCSolution:
    """The unconstrained equal-risk answer with the candidate free to compete.

    Different question from :func:`perturbation_scan`: not "what if I add
    ``w``" but "what would equal risk contribution actually assign this
    stock, and how much does it reshuffle the other four." A candidate the
    solver assigns 3% at equilibrium and one it assigns 25% are very
    different findings before return ever enters the picture.
    """
    cov = estimate_covariance(returns[[candidate, *universe]])
    return solve_erc_with_status(cov)


@dataclass(frozen=True)
class IncrementalReport:
    """Everything one run of the incremental analysis produces.

    ``baseline_solution`` and ``five_asset_solution`` carry their own
    ``converged`` flags (``riskparity.ERCSolution``) -- a caller (the CLI,
    step 40) has to be able to say whether the equal-risk answer it is about
    to print actually converged, not just report the weights.

    ``weight_diff`` compares the 5-asset re-solve's weight on each existing
    sleeve against that sleeve's baseline (4-asset) weight -- how much the
    book reshuffles when the candidate is allowed to compete on equal terms.
    The candidate itself has no baseline counterpart to diff against, so it is
    not in this series; its 5-asset weight is in ``five_asset_weights``.
    """

    candidate: str
    universe: tuple[str, ...]
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_observations: int
    lookback_days: int
    min_history_days: int
    baseline_weights: pd.Series
    baseline_solution: ERCSolution
    perturbation: pd.DataFrame
    five_asset_weights: pd.Series
    five_asset_solution: ERCSolution
    weight_diff: pd.Series


def analyze_position(
    candidate: str,
    universe: tuple[str, ...] = UNIVERSE,
    lookback_days: int = INCREMENTAL_LOOKBACK_DAYS,
    min_history_days: int = INCREMENTAL_MIN_HISTORY_DAYS,
    candidate_weights: tuple[float, ...] = INCREMENTAL_CANDIDATE_WEIGHTS,
    refresh: bool = False,
) -> IncrementalReport:
    """Fetch, solve and scan: the full step 37 pipeline for one candidate.

    Raises
    ------
    ValueError
        If ``candidate`` is already a member of ``universe`` -- "what happens
        if we add this stock" is not well posed for a stock already in the
        book.
    DataUnavailableError
        If the candidate cannot be fetched at all, or has fewer than
        ``min_history_days`` of history overlapping the universe's.
    """
    candidate = candidate.strip().upper()
    universe = tuple(t.strip().upper() for t in universe)
    if candidate in universe:
        raise ValueError(
            f"{candidate} is already a sleeve in {universe}; the incremental "
            "analysis asks what happens when it is ADDED to the book, which is "
            "not well posed for a stock already in it"
        )
    if min_history_days > lookback_days:
        raise ValueError(
            f"min_history_days ({min_history_days}) cannot exceed lookback_days "
            f"({lookback_days}) -- the floor must be at or below the target"
        )

    returns = _fetch_window(candidate, universe, lookback_days, min_history_days, refresh)

    baseline = baseline_erc(returns, universe)
    baseline_weights = pd.Series(baseline.weights, index=universe, name="baseline_weight")

    scan = perturbation_scan(returns, candidate, baseline.weights, universe, candidate_weights)

    five = resolve_five_asset_erc(returns, candidate, universe)
    five_weights = pd.Series(five.weights, index=[candidate, *universe], name="five_asset_weight")
    weight_diff = (five_weights[list(universe)] - baseline_weights).rename("weight_diff")

    return IncrementalReport(
        candidate=candidate,
        universe=universe,
        window_start=returns.index[0],
        window_end=returns.index[-1],
        n_observations=len(returns),
        lookback_days=lookback_days,
        min_history_days=min_history_days,
        baseline_weights=baseline_weights,
        baseline_solution=baseline,
        perturbation=scan,
        five_asset_weights=five_weights,
        five_asset_solution=five,
        weight_diff=weight_diff,
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # --- INCREMENTAL_MIN_HISTORY_DAYS, monkeypatched rather than found -------
    # The same house technique tests/test_lookahead.py uses to attack a guard
    # deterministically: a real recent IPO thin enough to trip this goes stale
    # the moment it trades one more day, so the guard is exercised against a
    # synthetic short history instead of a ticker that happens to qualify today.
    print("MIN_HISTORY_DAYS guard (monkeypatched short synthetic history)")
    module = sys.modules[__name__]
    real_fetch_prices = module.fetch_prices

    def _thin_fetch_prices(tickers, start=None, end=None, refresh=False):
        n = INCREMENTAL_MIN_HISTORY_DAYS // 2
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
        rng = np.random.default_rng(0)
        levels = 100.0 * np.cumprod(
            1.0 + rng.normal(0.0004, 0.01, size=(n, len(tickers))), axis=0
        )
        return pd.DataFrame(levels, index=idx, columns=list(tickers))

    module.fetch_prices = _thin_fetch_prices
    try:
        analyze_position("ZZZZ")
        print("  NO ERROR RAISED -- regression!")
    except DataUnavailableError as exc:
        print(f"  raised DataUnavailableError, as it must: {str(exc)[:110]}...")
    finally:
        module.fetch_prices = real_fetch_prices

    # --- the live path: a real ticker, real data, real network --------------
    print("\nLive run against a real ticker (network required)")
    DEMO_TICKER = "AAPL"
    try:
        report = analyze_position(DEMO_TICKER)
    except DataUnavailableError as exc:
        print(f"  could not fetch live data ({exc}); skipping the live demonstration.")
    else:
        print(f"  candidate         : {report.candidate}")
        print(f"  window            : {report.window_start.date()} -> "
              f"{report.window_end.date()} ({report.n_observations} obs, target "
              f"{report.lookback_days})")
        print(f"  baseline weights  : "
              f"{report.baseline_weights.round(4).to_dict()} "
              f"(converged={report.baseline_solution.converged})")
        print(f"  5-asset weights   : "
              f"{report.five_asset_weights.round(4).to_dict()} "
              f"(converged={report.five_asset_solution.converged})")
        print(f"  weight diff (4->5): {report.weight_diff.round(4).to_dict()}")
        print("  perturbation scan :")
        print(
            report.perturbation[
                ["portfolio_vol", f"weight_{DEMO_TICKER}", f"rc_fraction_{DEMO_TICKER}"]
            ].round(4).to_string()
        )

        # sum-to-one, at every stage -- the cheapest possible regression check
        print(f"\n  baseline sums to 1  : {abs(report.baseline_weights.sum() - 1.0):.2e}")
        print(f"  5-asset sums to 1   : {abs(report.five_asset_weights.sum() - 1.0):.2e}")
        for w in report.perturbation.index:
            weight_cols = [c for c in report.perturbation.columns if c.startswith("weight_")]
            total = report.perturbation.loc[w, weight_cols].sum()
            print(f"  scan[{w}] sums to 1  : {abs(total - 1.0):.2e}")
