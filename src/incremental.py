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
3. :func:`compensation_test` -- ``SR_i > SR_p * rho(i, p)``, a known result
   (Treynor-Black / Elton-Gruber active-portfolio theory) rather than
   something invented here: adding asset ``i`` at an infinitesimal weight
   improves portfolio ``p``'s Sharpe ratio iff its own Sharpe exceeds the
   portfolio's Sharpe times its correlation to the *portfolio's returns* (not
   to any one sleeve). It is a **marginal** test -- whether the direction of
   adding the candidate helps, not what weight to add it at -- which is
   exactly complementary to (1) and (2). The candidate's correlation to each
   of the four sleeves individually ships alongside it, since that decomposes
   *why* the portfolio correlation came out where it did.

Run a self-check from the repo root (this one, unlike the seven Phase 0-5
module self-checks, touches the network for its live-ticker demonstration --
there is no way to demonstrate "analyse a real stock" without it)::

    .venv/bin/python -m src.incremental
"""

from __future__ import annotations

import logging
import math
import textwrap
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    ANNUAL_RISK_FREE_RATE,
    INCREMENTAL_CANDIDATE_WEIGHTS,
    INCREMENTAL_LOOKBACK_DAYS,
    INCREMENTAL_MIN_HISTORY_DAYS,
    SUMMARY_LINE_WIDTH,
    TRADING_DAYS_PER_YEAR,
    UNIVERSE,
)
from .covariance import estimate_covariance
from .data import DataUnavailableError, daily_returns, fetch_prices
from .metrics import sharpe_ratio
from .riskparity import ERCSolution, portfolio_volatility, risk_contributions, solve_erc_with_status

log = logging.getLogger(__name__)

# Padding above the requested trading-day count, in BUSINESS days, so the
# ~9-10 NYSE holidays a year cannot leave the fetch short of what was wanted.
# The window is trimmed to the exact row count wanted afterward regardless
# (see _fetch_window), so overshooting here costs nothing but a slightly
# larger download.
#
# Measured, not guessed: a 756-trading-day request (~3 years) with a 30-day
# buffer returned only 754 real rows from a live fetch -- short by 2, because
# 3 years of NYSE holidays (~28-30) plus the one row daily_returns drops ate
# the whole buffer with nothing left over. 40 returned 763 (7-day margin); 45
# is used here for headroom against a year with a slightly heavier holiday
# calendar, still comfortably short of a wasteful over-fetch.
_BUSINESS_DAY_HOLIDAY_BUFFER: int = 45


def _window_start(trading_days: int) -> str:
    """A calendar start date comfortably covering ``trading_days`` of trading.

    Built from ``pd.bdate_range`` -- exact business-day (Mon-Fri) arithmetic,
    the same primitive ``src/data.py`` already uses for its own calendar work
    -- rather than a hand-tuned calendar-day ratio. Business days still
    overcount true NYSE trading days by the holiday calendar, hence the
    buffer on top; padding rather than exactness is fine because the caller
    trims to the exact row count it wants afterward.
    """
    start = pd.bdate_range(
        end=pd.Timestamp.today().normalize(), periods=trading_days + _BUSINESS_DAY_HOLIDAY_BUFFER
    )[0]
    return start.strftime("%Y-%m-%d")


def _fetch_window(
    candidate: str,
    universe: tuple[str, ...],
    lookback_days: int,
    min_history_days: int,
    refresh: bool,
) -> pd.DataFrame:
    """Daily returns for ``[candidate, *universe]`` on the trailing window.

    Fetches candidate and universe **together**, in one ``fetch_prices`` call,
    rather than the candidate alone: a candidate-only call would write the
    candidate's single column as the *entire* on-disk price cache --
    ``cache.write_frame`` replaces the whole file rather than merging into it
    -- wiping out the cached SPY/TLT/DBC/TIP history ``run_backtest.py``
    expects to find. The combined call keeps the universe's columns present.

    **What this does not fully avoid, and why it is still safe.** A fresh
    combined download still writes the cache at *this* call's short window,
    not the core engine's full ``DATA_START_DATE``-to-present range -- so a
    candidate not already cached alongside the universe can temporarily
    shrink the shared cache's covered history for SPY/TLT/DBC/TIP too, not
    only add the candidate. This is bounded and self-healing rather than
    silent corruption: ``run_backtest.py`` always requests
    ``fetch_prices(UNIVERSE, start=DATA_START_DATE)``, which checks the
    cache's coverage before trusting it (see ``data._cached_prices``) and
    re-downloads the full history itself if the cache falls short -- the
    engine never uses truncated universe data, it just occasionally pays for
    one extra download. Avoiding this outright would mean either writing
    ``data.py``'s cache merge-aware, which PLAN.md's Phase 6 design
    deliberately avoids, or requesting ``DATA_START_DATE`` here too, which
    would force every candidate younger than 2006 -- most of them -- into
    the ``min_history_days`` retry below even when they comfortably cover
    the full target lookback. Between those two costs, the bounded,
    self-healing one is the smaller.

    Tries the full ``lookback_days`` window first, which is the common case:
    most tickers anyone would ask about have traded longer than the ~3-year
    lookback. ``fetch_prices`` fails loudly when a ticker's history starts
    materially later than the requested window (``data._validate``'s
    late-start guard) -- that is also exactly the signature of a recent IPO,
    so on that specific failure the fetch is retried against the shorter
    ``min_history_days`` window instead of surfacing the first attempt's
    error. Note this retry is sized to the *floor*, not to however much
    history the candidate actually has: a candidate with, say, 500 trading
    days (above the floor, below the target) is still analysed on a
    ``min_history_days``-sized window rather than its full 500 -- a
    deliberate simplicity trade-off (finding the candidate's true available
    history without it would need a search, not a single retry), not a bug.
    Either way the candidate is still analysed rather than rejected, which is
    the point of ``INCREMENTAL_MIN_HISTORY_DAYS`` existing as a *floor*
    distinct from ``INCREMENTAL_LOOKBACK_DAYS``, the *target*.
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
            # Both underlying errors are surfaced verbatim rather than
            # collapsed into a single "insufficient history" diagnosis: that
            # framing is only correct when the cause really is a recent IPO.
            # data._validate raises the same DataUnavailableError for a
            # typo'd ticker, a data gap, or non-positive prices, none of
            # which a shorter retry window fixes -- asserting "not enough
            # history" over one of those would send a reader chasing the
            # wrong flag.
            raise DataUnavailableError(
                f"{candidate}: could not fetch enough recent history after two "
                f"attempts. Full {lookback_days}-day lookback: {exc}. Retry at "
                f"the {min_history_days}-day floor: {exc2}"
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
    solution = solve_erc_with_status(cov)
    if not solution.converged:
        # solve_erc_with_status already warns on its own non-convergence; this
        # adds the one thing it cannot know -- which analysis stage this was --
        # so a caller reading logs (rather than IncrementalReport's converged
        # flags) can still tell the baseline snapshot apart from the 5-asset
        # re-solve below.
        log.warning(
            "baseline_erc did not converge on %s (deviation %.3e)",
            universe, solution.max_rc_deviation,
        )
    return solution


def _five_asset_covariance(
    returns: pd.DataFrame, candidate: str, universe: tuple[str, ...]
) -> np.ndarray:
    """The 5x5 covariance :func:`perturbation_scan` and :func:`resolve_five_asset_erc`
    both need -- factored out so :func:`analyze_position` can compute it once
    and hand it to both rather than paying the EWMA/shrinkage/PSD-repair cost
    twice on an identical input.
    """
    return estimate_covariance(returns[[candidate, *universe]])


def perturbation_scan(
    returns: pd.DataFrame,
    candidate: str,
    baseline_weights: np.ndarray,
    universe: tuple[str, ...] = UNIVERSE,
    candidate_weights: tuple[float, ...] = INCREMENTAL_CANDIDATE_WEIGHTS,
    cov: np.ndarray | None = None,
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

    ``cov``, if given, must be the 5x5 covariance over ``[candidate, *universe]``
    in that column order -- pass it when the caller (:func:`analyze_position`)
    already computed it for :func:`resolve_five_asset_erc`, to avoid deriving
    the same estimate twice. Left as ``None`` it is computed here, so calling
    this function on its own (as the tests and the ``__main__`` self-check do)
    needs nothing extra.
    """
    if not candidate_weights:
        # Without this, pd.DataFrame([]).set_index("candidate_weight") below
        # raises a bare KeyError -- correct that the column doesn't exist on
        # an empty frame, but it names the wrong thing: the actual mistake is
        # an empty candidate_weights, not a missing column. The CLI's argparse
        # already rejects an empty --weights, so this only matters calling
        # perturbation_scan/analyze_position directly as a library.
        raise ValueError("candidate_weights must contain at least one value")

    columns = [candidate, *universe]
    if cov is None:
        cov = _five_asset_covariance(returns, candidate, universe)

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
    returns: pd.DataFrame,
    candidate: str,
    universe: tuple[str, ...] = UNIVERSE,
    cov: np.ndarray | None = None,
) -> ERCSolution:
    """The unconstrained equal-risk answer with the candidate free to compete.

    Different question from :func:`perturbation_scan`: not "what if I add
    ``w``" but "what would equal risk contribution actually assign this
    stock, and how much does it reshuffle the other four." A candidate the
    solver assigns 3% at equilibrium and one it assigns 25% are very
    different findings before return ever enters the picture.

    ``cov``, if given, must be the 5x5 covariance over ``[candidate, *universe]``
    -- see :func:`perturbation_scan`'s matching parameter.
    """
    if cov is None:
        cov = _five_asset_covariance(returns, candidate, universe)
    solution = solve_erc_with_status(cov)
    if not solution.converged:
        log.warning(
            "resolve_five_asset_erc did not converge for candidate %s against "
            "%s (deviation %.3e)",
            candidate, universe, solution.max_rc_deviation,
        )
    return solution


# ---------------------------------------------------------------------------
# Step 38 -- the compensation test
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CompensationTest:
    """Treynor-Black / Elton-Gruber: is the candidate's added risk paid for.

    ``passes`` is ``candidate_sharpe > portfolio_sharpe * correlation_to_portfolio``
    -- derived by writing the blended return as ``(1-eps)*r_p + eps*r_i``,
    differentiating ``Sharpe(eps)`` at ``eps = 0``, and reducing the sign of the
    derivative to exactly this inequality (PLAN.md's Phase 6 section has the
    full derivation). It is a **marginal** test: it says whether increasing the
    candidate's weight from 0 helps, not what the optimal weight is -- read
    alongside :func:`perturbation_scan`, which answers that complementary
    question. A candidate with a mediocre or even negative own Sharpe can still
    pass this if ``correlation_to_portfolio`` is low or negative; a candidate
    with an excellent own Sharpe can still fail it if it is highly correlated
    with the book -- correlation, not the candidate's own return, is the real
    driver of the sign.

    Both Sharpe ratios are computed at ``config.ANNUAL_RISK_FREE_RATE`` (0.0 by
    default project-wide), matching every other Sharpe/return-to-vol ratio in
    this project so the test is internally consistent with the rest of the
    write-up rather than mixing conventions.
    """

    candidate_sharpe: float
    portfolio_sharpe: float
    correlation_to_portfolio: float
    correlation_threshold: float
    passes: bool
    correlation_to_sleeves: pd.Series


def compensation_test(
    returns: pd.DataFrame,
    candidate: str,
    baseline_weights: pd.Series | np.ndarray,
    universe: tuple[str, ...] = UNIVERSE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    annual_risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
) -> CompensationTest:
    """Run the compensation test for ``candidate`` against the baseline book.

    ``portfolio_returns`` holds the baseline ERC weights **constant** over the
    whole window rather than replaying a rebalance sequence -- this is a
    point-in-time snapshot's Sharpe on its current weights, not a backtest, so
    there is no decision-timing question for a one-day lag to resolve (unlike
    ``overlay.py``'s ``_in_force_before``, which exists specifically for a
    walk-forward sequence of *changing* decisions).

    ``correlation_to_portfolio`` is ``rho(i, p)`` in the formula above: the
    candidate's correlation to the *portfolio's* returns, not to any one
    sleeve. ``correlation_to_sleeves`` is the per-sleeve breakdown that
    decomposes why that number came out where it did -- e.g. a candidate
    uncorrelated with SPY but correlated with DBC is not diversifying the
    equity risk it looks like it is diversifying.

    ``baseline_weights`` as a ``pd.Series`` is aligned by **label** (its own
    index), not by position -- a plain ``np.ndarray`` is still read
    positionally, matched to ``universe`` in order. Every call site in this
    module passes the ndarray ERC solutions carry, so the label path is
    exercised only by a caller supplying its own Series; without it, a Series
    whose index order differed from ``universe`` would silently relabel
    weights onto the wrong sleeves instead of raising.
    """
    if isinstance(baseline_weights, pd.Series):
        weights = baseline_weights.reindex(list(universe))
        if weights.isna().any():
            # reindex() produces NaN both for a label absent from the input's
            # own index and for a label that WAS present but already held a
            # NaN value -- these are different mistakes and need different
            # messages, or "missing an entry for" reports an empty list when
            # the real problem is a NaN value on a label that is present.
            missing = [t for t in universe if t not in baseline_weights.index]
            if missing:
                raise ValueError(
                    f"baseline_weights is missing an entry for {missing}; its "
                    f"index must cover every ticker in universe {universe}"
                )
            nan_valued = [t for t in universe if pd.isna(weights[t])]
            raise ValueError(
                f"baseline_weights has a NaN value for {nan_valued}; every "
                f"entry must be a finite weight"
            )
    else:
        weights = pd.Series(np.asarray(baseline_weights, dtype="float64"), index=list(universe))
    sleeve_returns = returns[list(universe)]
    portfolio_returns = sleeve_returns.mul(weights, axis=1).sum(axis=1)
    candidate_returns = returns[candidate]

    candidate_sharpe = sharpe_ratio(candidate_returns, periods_per_year, annual_risk_free_rate)
    portfolio_sharpe = sharpe_ratio(portfolio_returns, periods_per_year, annual_risk_free_rate)
    correlation_to_portfolio = float(candidate_returns.corr(portfolio_returns))
    threshold = portfolio_sharpe * correlation_to_portfolio

    return CompensationTest(
        candidate_sharpe=candidate_sharpe,
        portfolio_sharpe=portfolio_sharpe,
        correlation_to_portfolio=correlation_to_portfolio,
        correlation_threshold=threshold,
        passes=bool(candidate_sharpe > threshold),
        # rename_axis(None) strips whatever the price frame's column index was
        # named -- yfinance's raw download names it "Ticker", a cache round
        # trip through cache.read_frame does not, so without this the report
        # printed a stray "Ticker" header on a fresh download and not on a
        # cached one. Cosmetic, but the report should read the same either way.
        correlation_to_sleeves=(
            sleeve_returns.corrwith(candidate_returns).rename("correlation").rename_axis(None)
        ),
    )


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

    ``compensation`` is the marginal test (:class:`CompensationTest`): does
    adding the candidate at an infinitesimal weight improve the baseline
    portfolio's Sharpe ratio. Complementary to ``perturbation``, not a
    substitute for it -- one says whether the direction helps, the other says
    what a specific amount does.
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
    compensation: CompensationTest


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

    # Computed once and handed to both: perturbation_scan and
    # resolve_five_asset_erc both need the identical 5x5 covariance, and
    # estimate_covariance's EWMA/shrinkage/PSD-repair pipeline is not free.
    five_asset_cov = _five_asset_covariance(returns, candidate, universe)
    scan = perturbation_scan(
        returns, candidate, baseline.weights, universe, candidate_weights, cov=five_asset_cov
    )

    five = resolve_five_asset_erc(returns, candidate, universe, cov=five_asset_cov)
    five_weights = pd.Series(five.weights, index=[candidate, *universe], name="five_asset_weight")
    weight_diff = (five_weights[list(universe)] - baseline_weights).rename("weight_diff")

    compensation = compensation_test(returns, candidate, baseline.weights, universe)

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
        compensation=compensation,
    )


# ---------------------------------------------------------------------------
# Step 40 -- the report analyze_position.py prints
# ---------------------------------------------------------------------------
# Stated in the report every time it is printed, not just in PLAN.md, because
# PLAN.md is not what a CLI user reads. Order matches PLAN.md's "Explicit
# caveats the tool must state every time it runs".
_CAVEATS: tuple[str, ...] = (
    "Point-in-time, not historical: this is a snapshot on the trailing window "
    "above, as of the run date. It is not a claim about how the position would "
    "have performed since 2006, and a re-run next month will read differently "
    "as the window moves.",
    "A marginal test, not a sizing recommendation: the compensation test says "
    "whether the DIRECTION of adding the candidate helps, not what weight to "
    "add it at. Read it alongside the perturbation scan above, which is for "
    "that question.",
    "Still a single stock: passing the compensation test does not manufacture "
    "a structural risk premium the way holding an asset class does. It answers "
    "whether this improves THIS portfolio's risk-adjusted return given the "
    "correlation structure right now -- a narrower and more honest question "
    "than \"should I own this stock.\"",
)


def _convergence_line(solution: ERCSolution) -> str:
    """The "! converged: ..." line shared by the baseline and 5-asset blocks."""
    marker = " " if solution.converged else "!"
    return (
        f"{marker} converged: {solution.converged} "
        f"(max risk-contribution deviation {solution.max_rc_deviation:.2e})"
    )


def format_position_report(report: IncrementalReport) -> str:
    """Human-readable account of one :func:`analyze_position` run.

    In the style of ``backtest.format_run_summary``: a solver that did not
    converge gets a leading ``!`` rather than being one number among many, a
    table for each of the three pieces of math, and the caveats PLAN.md
    requires printed every time -- a docstring nobody reading the CLI's
    output will ever see is not "stating them."
    """
    lines = [
        f"{report.candidate} added to the {'/'.join(report.universe)} book "
        "-- incremental position analysis"
    ]
    window_note = (
        ""
        if report.n_observations == report.lookback_days
        else f", SHORTER than the {report.lookback_days}-day target -- thin candidate history"
    )
    lines.append(
        f"  window: {report.window_start.date()} -> {report.window_end.date()} "
        f"({report.n_observations} obs{window_note})"
    )

    lines.append("\nBaseline -- existing 4-asset ERC book on this window")
    lines.append(_convergence_line(report.baseline_solution))
    lines.append(textwrap.indent(report.baseline_weights.round(4).to_frame("weight").to_string(), "  "))

    lines.append(f"\n5-asset re-solve -- {report.candidate} free to compete for equal risk")
    lines.append(_convergence_line(report.five_asset_solution))
    five_table = pd.DataFrame(
        {
            "weight": report.five_asset_weights,
            "vs baseline": report.weight_diff.reindex(report.five_asset_weights.index),
        }
    )
    lines.append(textwrap.indent(five_table.round(4).to_string(na_rep="--"), "  "))

    lines.append("\nPerturbation scan -- fixed candidate weights, not a re-solve")
    scan_columns = [
        "portfolio_vol",
        f"weight_{report.candidate}",
        f"rc_fraction_{report.candidate}",
    ]
    lines.append(textwrap.indent(report.perturbation[scan_columns].round(4).to_string(), "  "))

    comp = report.compensation
    lines.append("\nCompensation test -- SR_i > SR_p * rho(i, p)")
    lines.append(
        f"  SR_i={comp.candidate_sharpe:+.3f}  SR_p={comp.portfolio_sharpe:+.3f}  "
        f"rho(i,p)={comp.correlation_to_portfolio:+.3f}  "
        f"threshold={comp.correlation_threshold:+.3f}"
    )
    # metrics.sharpe_ratio returns NaN on a near-zero-variance return series
    # (_ZERO_VARIANCE_EPS), which would otherwise read as a silent "FAILS" --
    # `NaN > x` is False in Python/numpy, so comp.passes gives no sign
    # anything was actually undefined. Solver non-convergence gets its own
    # explicit "!" marker elsewhere in this report; a NaN Sharpe deserves the
    # same visibility rather than being indistinguishable from a real fail.
    undefined = any(
        np.isnan(v)
        for v in (comp.candidate_sharpe, comp.portfolio_sharpe, comp.correlation_threshold)
    )
    if undefined:
        verdict = (
            "UNDEFINED -- a Sharpe ratio is NaN (a near-zero-variance return "
            "series), so the test cannot be evaluated"
        )
    else:
        verdict = (
            "PASSES -- adding a small amount would improve risk-adjusted return"
            if comp.passes
            else "FAILS -- adding a small amount would not improve risk-adjusted return"
        )
    lines.append(f"  {verdict}")
    lines.append("  correlation to each sleeve:")
    lines.append(textwrap.indent(comp.correlation_to_sleeves.round(4).to_string(), "    "))

    lines.append("\nWhat this is, and is not:")
    for i, caveat in enumerate(_CAVEATS, start=1):
        lines.extend(
            textwrap.wrap(
                f"{i}. {caveat}",
                width=SUMMARY_LINE_WIDTH,
                subsequent_indent="   ",
            )
        )

    return "\n".join(lines)


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

    # --- step 38's "done when": a hand-derivable two-asset synthetic case ---
    # Built exactly, not merely approximately -- a Gram-Schmidt construction
    # gives a candidate series with the REQUESTED sample correlation and
    # sample mean/std to floating-point precision, not a value a large-enough
    # random draw happens to land near. That sidesteps the house rule "never
    # assert on a single simulated path" (CLAUDE.md) at the root: there is no
    # sampling noise here to be unlucky about, so one seed is enough.
    print("\ncompensation test: hand-derivable two-asset synthetic cases")

    def _exact_series(n: int, mean_target: float, std_target: float, seed: int) -> np.ndarray:
        """A series with sample mean/std (ddof=1) equal to the targets, exactly."""
        raw = np.random.default_rng(seed).normal(size=n)
        raw = raw - raw.mean()
        raw = raw / raw.std(ddof=1)
        return mean_target + std_target * raw

    def _exact_corr_partner(
        base: np.ndarray, rho: float, mean_target: float, std_target: float, seed: int
    ) -> np.ndarray:
        """A series with EXACT sample correlation ``rho`` to ``base`` and the
        requested sample mean/std, via Gram-Schmidt: an independent draw is
        orthogonalised against ``base`` then blended back at the angle that
        gives the requested correlation. Affine rescaling afterward sets the
        mean/std without touching the correlation (Pearson's r is invariant to
        a positive affine transform of either series).
        """
        base_centered = base - base.mean()
        u = base_centered / np.linalg.norm(base_centered)
        raw = np.random.default_rng(seed).normal(size=len(base))
        raw_centered = raw - raw.mean()
        orthogonal = raw_centered - (raw_centered @ u) * u
        v = orthogonal / np.linalg.norm(orthogonal)
        z = rho * u + math.sqrt(max(0.0, 1.0 - rho * rho)) * v
        z = z / z.std(ddof=1)
        return mean_target + std_target * z

    n_days = 2000
    portfolio_returns = pd.Series(
        _exact_series(n_days, 0.0006, 0.01, seed=1),
        index=pd.bdate_range("2020-01-01", periods=n_days),
        name="portfolio",
    )
    sr_p = sharpe_ratio(portfolio_returns, TRADING_DAYS_PER_YEAR)
    print(f"  baseline portfolio Sharpe (SR_p) = {sr_p:.4f}")

    # Each scenario's expected pass/fail is derived by hand from the formula
    # BEFORE running it: at rho=+1 the test reduces to SR_i > SR_p; at rho=0
    # the threshold is 0 regardless of SR_p, so any positive SR_i passes; at
    # rho=-1 it reduces to SR_i > -SR_p, so a candidate can pass with a
    # NEGATIVE own Sharpe as long as it is not too negative -- the case that
    # makes "correlation, not the candidate's own return, is the real driver"
    # concrete rather than asserted.
    scenarios = [
        ("rho=+1, SR_i far above SR_p", +1.0, 0.0012, 0.01, True),
        ("rho=+1, SR_i far below SR_p", +1.0, 0.0002, 0.01, False),
        ("rho=0,  SR_i clearly positive", 0.0, 0.0003, 0.01, True),
        ("rho=-1, SR_i<0 but above -SR_p", -1.0, -0.0002, 0.01, True),
        ("rho=-1, SR_i far below -SR_p", -1.0, -0.0010, 0.01, False),
    ]
    all_ok = True
    for label, rho, mu_i, sigma_i, expected_pass in scenarios:
        candidate_returns = pd.Series(
            _exact_corr_partner(portfolio_returns.to_numpy(), rho, mu_i, sigma_i, seed=2),
            index=portfolio_returns.index,
            name="candidate",
        )
        sr_i = sharpe_ratio(candidate_returns, TRADING_DAYS_PER_YEAR)
        measured_rho = float(candidate_returns.corr(portfolio_returns))
        threshold = sr_p * measured_rho
        passes = bool(sr_i > threshold)
        ok = passes == expected_pass
        all_ok &= ok
        print(
            f"  {label:32s}: SR_i={sr_i:+.4f} rho={measured_rho:+.4f} "
            f"threshold={threshold:+.4f} passes={passes} "
            f"({'ok' if ok else 'MISMATCH -- regression!'})"
        )
    print(f"  all scenarios matched their hand-derived expectation: {bool(all_ok)}")

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
        comp = report.compensation
        print(f"\n  compensation test : SR_i={comp.candidate_sharpe:+.4f} "
              f"SR_p={comp.portfolio_sharpe:+.4f} "
              f"rho(i,p)={comp.correlation_to_portfolio:+.4f} "
              f"threshold={comp.correlation_threshold:+.4f} passes={comp.passes}")
        print(f"  correlation to sleeves: "
              f"{comp.correlation_to_sleeves.round(4).to_dict()}")

        # sum-to-one, at every stage -- the cheapest possible regression check
        print(f"\n  baseline sums to 1  : {abs(report.baseline_weights.sum() - 1.0):.2e}")
        print(f"  5-asset sums to 1   : {abs(report.five_asset_weights.sum() - 1.0):.2e}")
        for w in report.perturbation.index:
            weight_cols = [c for c in report.perturbation.columns if c.startswith("weight_")]
            total = report.perturbation.loc[w, weight_cols].sum()
            print(f"  scan[{w}] sums to 1  : {abs(total - 1.0):.2e}")
