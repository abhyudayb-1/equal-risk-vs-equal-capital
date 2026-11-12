"""Equal risk against equal capital -- the comparison this project exists to make
.

Two construction rules are being compared over one multi-asset book: **equal risk
contribution (ERC)** and **equal capital (1/N)**. The question is what equalising
risk buys, and what it costs in volatility and return.

**Why a volatility-matched column is not optional.** ERC run unlevered is a
lower-volatility book than 1/N -- 6.85% against 8.25% on the live sample -- so
their *returns* are not comparable and a table that puts them side by side on
CAGR is comparing two different amounts of risk. The vol-targeting overlay is
what fixes that: re-run the engine with ``target_vol`` set to 1/N's own realised
volatility and the two books are (approximately, see below) the same size, so the
return difference means something. The financing that levering costs is then part
of the answer rather than a detail, which is why the financed and financing-free
versions are both reported.

**The matched column is a retrospective diagnostic, not a strategy anyone could
have run.** Its target is 1/N's realised volatility over the *whole* evaluation
window, a number known only at the end of it. Nothing inside the engine looks
ahead -- each month's leverage still uses only trailing data -- but the single
constant it aims at was chosen with hindsight. It answers "at the same realised
risk as 1/N, which rule earned more?", which is a fair question to ask of history
and not a forecast. The labels say "(hindsight)" so that no table can present it
as a tradable rule. A target chosen from a calibration period before the window
would make it tradable, at the cost of not matching 1/N's risk.

**What this module does and does not own.** Every function here is a pure
function of return series -- no network, no disk, no argument parsing. It does
call :func:`src.backtest.run_backtest` a second time for the matched column,
because that is also a pure function and because ``backtest.py`` owns the overlay
alignment that a hand-rolled ``leverage = target / vol`` would get wrong (see the
alignment notes in ``overlay.py``). The data fetching
and the printing-to-a-terminal belong to ``compare_rules.py``.

Metric rows come from :mod:`src.riskmetrics` and the pairwise blocks from
:mod:`src.headtohead`, so nothing statistical is re-derived here.

**Two measurement facts that shape the output, both measured rather than
assumed**:

1. **The overlay overshoots its target by roughly 13%.** On the live sample
   (data through 2026-11-06) a 10% target realised 11.34% and an 8.25% target
   realised 9.30%. So the matched column is
   *approximately* matched, and :class:`RuleComparison` carries ``target_vol``
   and ``realized_vol`` side by side so a report can print both. Describing it
   as exactly matched would be wrong.
2. **A different ``target_vol`` does not change the ERC weights.** The target and
   the cap feed the leverage path only, never the ERC solve, so the matched
   column is one extra engine run rather than a redesign. :func:`compare_rules`
   asserts it, because the whole construction depends on it.

Run a self-check from the repo root (offline, fixtures only, writes nothing)::

    .venv/bin/python -m src.comparison
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import DEFAULT_STRATEGY_NAME, run_backtest
from .benchmarks import equal_weight, max_sharpe, naive_risk_parity, sixty_forty
from .config import (
    ANNUAL_RISK_FREE_RATE,
    COV_WARMUP_TRADING_DAYS,
    LEVERAGE_CAP,
    MIN_MATCHABLE_VOL,
    SUMMARY_LINE_WIDTH,
    TARGET_VOL,
    TRADING_DAYS_PER_YEAR,
    UNIVERSE,
)
from .covariance import estimate_covariance
from .headtohead import crisis_table, format_crisis_table, format_relative_table, relative_metrics, tail_block
from .metrics import summarize
from .overlay import apply_overlay, unlevered_returns
from .riskmetrics import format_metric_table, realized_risk_table

log = logging.getLogger(__name__)

# The column names, in the order a report prints them. The reference rows sit
# between the two protagonists and the levered variants on purpose: 1/sigma is
# what says how much of ERC's edge needs the solver at all, and 60/40 is what
# says whether any of this beats the institutional default.
EQUAL_CAPITAL = "1/N (equal weight)"
ERC_UNLEVERED = "ERC unlevered"
NAIVE_INV_VOL = "Naive 1/vol"
SIXTY_FORTY = "60/40"
MAX_SHARPE = "Max Sharpe"
ERC_TARGET_FINANCED = "ERC @ target, financed"
ERC_TARGET_FREE = "ERC @ target, financing omitted"
ERC_MATCHED_FINANCED = "ERC @ 1/N vol (hindsight), financed"
ERC_MATCHED_FREE = "ERC @ 1/N vol (hindsight), financing omitted"

COLUMN_ORDER: tuple[str, ...] = (
    ERC_UNLEVERED,
    NAIVE_INV_VOL,
    EQUAL_CAPITAL,
    MAX_SHARPE,
    SIXTY_FORTY,
    ERC_TARGET_FINANCED,
    ERC_TARGET_FREE,
    ERC_MATCHED_FINANCED,
    ERC_MATCHED_FREE,
)

# Rows that are NOT strategies anyone can run. They exist because the difference
# between each and its financed twin is the cost of leverage, which is one of the
# three findings -- but a reader who takes them for a result has been misled, so
# every report marks them.
COUNTERFACTUAL: frozenset[str] = frozenset({ERC_TARGET_FREE, ERC_MATCHED_FREE})


class ComparisonError(ValueError):
    """The comparison cannot be built from the inputs given."""


@dataclass(frozen=True)
class RuleComparison:
    """Everything one equal-risk-vs-equal-capital comparison produces.

    ``series`` is the aligned daily returns every table was computed from, kept
    so a caller or a test can recompute any of this independently. ``summary``
    is :func:`src.riskmetrics.realized_risk_table`'s frame, one row per column
    of ``series``.

    ``target_vol`` is what the matched column *asked* the overlay for and
    ``realized_vol`` is what it actually delivered; they differ by roughly 13%
    because a trailing estimator cannot hit a target exactly. Both are carried
    so a report can print the gap rather than imply there isn't one.

    ``rule_results`` is the ``BacktestResult`` each distinct-weight rule actually
    produced, keyed the same way every table here is. It is carried rather than
    discarded because ``allocation`` and ``concentration`` are *summaries* of it:
    a chart that wants the risk shares through time or a sweep that
    wants the weights to measure turnover from needs the frames, and
    re-running five rules to get them back would be both slow and a second
    chance to run them on different inputs.

    ``matched_leverage`` is the matched run's own leverage path, carried because
    it is not recoverable from anything else here -- the ``rule_results`` entry
    for ERC is the *published-target* run, which holds the same weights at a
    different size. A trading-cost estimate for the matched column needs it.

    ``matched_target_overridden`` says whether ``matched_target_vol`` is 1/N's
    own realized volatility or a value the caller supplied. When it is an
    override, the ``ERC @ 1/N vol`` rows are **not** matched to 1/N and every
    printed form says so -- a column whose label lies is worse than no column.
    """

    data_through: str
    start: str
    n_observations: int
    n_rebalances: int
    target_vol: float
    matched_target_vol: float
    matched_realized_vol: float
    matched_target_overridden: bool
    matched_leverage: pd.Series
    equal_capital_vol: float
    rule_results: dict
    series: pd.DataFrame
    summary: pd.DataFrame
    relative: pd.DataFrame
    tail: pd.DataFrame
    crisis: pd.DataFrame
    allocation: pd.DataFrame
    concentration: pd.DataFrame
    effective_bets: pd.DataFrame
    weights_unchanged: bool
    meta: dict


def matched_target_vol(
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """The volatility an ERC book must be levered to in order to match ``benchmark_returns``.

    Just that series' own realised annualised volatility -- over every date it
    is given, so when it is given the whole evaluation window the result is
    known only in hindsight. That is the matched column's design, not an
    accident: see the module docstring, and never present the column built from
    it as a strategy that could have been run. It is a separate function because
    the number is reported next to the result and because a test needs to be
    able to ask for it without building a whole comparison.
    """
    if not isinstance(benchmark_returns, pd.Series):
        raise ComparisonError("benchmark_returns must be a pandas Series of daily returns")
    vol = float(summarize(benchmark_returns, periods_per_year)["annual_volatility"])
    # Against MIN_MATCHABLE_VOL rather than against 0.0. A constant return
    # series is degenerate but its measured volatility is float noise -- 1.7e-18,
    # not 0.0, because the constant is not exactly representable -- so a bare
    # `vol <= 0` test passes it and the overlay is handed a target it can only
    # answer by sitting on its leverage floor. See config.MIN_MATCHABLE_VOL.
    if not np.isfinite(vol) or vol <= MIN_MATCHABLE_VOL:
        raise ComparisonError(
            f"benchmark volatility is {vol!r}, at or below the {MIN_MATCHABLE_VOL:g} floor; "
            "there is no meaningful target to lever an ERC book to"
        )
    return vol


def build_series(
    sleeve_returns: pd.DataFrame,
    engine: object,
    matched: object,
    benchmarks: dict,
    risk_free: pd.Series,
) -> pd.DataFrame:
    """One daily return column per comparison row, on the dates all of them exist.

    Every column is taken from a ``BacktestResult`` the caller already built, or
    derived from one with ``overlay``'s own functions -- nothing here re-accrues
    returns by hand, which is the mistake that produces a plausible curve and a
    wrong one.
    """
    index = engine.returns.index
    for name, result in {"engine": engine, "matched": matched, **benchmarks}.items():
        if not result.returns.index.equals(index):
            raise ComparisonError(
                f"{name}'s return calendar differs from the engine's; "
                "a comparison across different calendars is not a fair one"
            )

    sleeves = sleeve_returns.loc[index]
    rf = risk_free.reindex(index)
    if rf.isna().any():
        raise ComparisonError("risk_free does not cover every return date")

    # The unlevered ERC book: the engine's own weights with no overlay at all.
    unlevered = unlevered_returns(sleeves, engine.weights)

    # The financing-free counterfactuals are re-accrued as held books at a zero
    # rate, not derived from the financed series by the one-period identity: in
    # a held book the loan's interest also moves the leverage between
    # rebalances, so "the same book, borrowing free" has its own leverage path.
    def financing_free(result) -> pd.Series:
        return apply_overlay(sleeves, result.weights, result.leverage, 0.0)["strategy"]

    columns = {
        ERC_UNLEVERED: unlevered,
        NAIVE_INV_VOL: benchmarks[NAIVE_INV_VOL].returns,
        EQUAL_CAPITAL: benchmarks[EQUAL_CAPITAL].returns,
        MAX_SHARPE: benchmarks[MAX_SHARPE].returns,
        SIXTY_FORTY: benchmarks[SIXTY_FORTY].returns,
        ERC_TARGET_FINANCED: engine.returns,
        ERC_TARGET_FREE: financing_free(engine),
        ERC_MATCHED_FINANCED: matched.returns,
        ERC_MATCHED_FREE: financing_free(matched),
    }
    return pd.DataFrame({name: columns[name] for name in COLUMN_ORDER}, index=index)


# ---------------------------------------------------------------------------
# Risk allocation -- where each rule actually puts its risk
# ---------------------------------------------------------------------------
# The return table says what each rule earned. This says what each rule was
# DOING, which is the part a capital-weight pie chart hides: ERC holds equal
# risk by construction and lets capital move, 1/N holds capital still and lets
# risk move wherever volatility takes it.
#
# Only the rules with distinct weights get a block. The levered ERC columns are
# deliberately excluded: they hold the SAME weights as ERC unlevered -- that is
# the whole premise of the matched column -- so listing them would repeat one
# rule four times and imply four allocations where there is one.
ALLOCATION_RULES: tuple[str, ...] = (ERC_UNLEVERED, EQUAL_CAPITAL, NAIVE_INV_VOL, SIXTY_FORTY, MAX_SHARPE)


def effective_bets(risk_contributions: pd.DataFrame) -> pd.Series:
    """Effective number of risk shares per rebalance: ``1 / sum(RC_i**2)``.

    The reciprocal Herfindahl index of the estimated risk shares. ``N`` when every
    sleeve carries ``1/N`` of the risk, ``1`` when one sleeve carries all of it.
    It measures how *evenly the estimated risk is split* across sleeves, and
    nothing more.

    **It is not a count of independent bets.** The risk shares already fold the
    correlations in -- each is ``w_i * (cov @ w)_i`` -- but an even split of
    variance among correlated sleeves is still a split among correlated sleeves:
    four equal shares from four assets that move together are not four
    independent sources of return. Counting independent bets needs the
    covariance's factor structure (e.g. the dispersion of risk across principal
    components), which this statistic does not look at. The name is the
    literature's; the reading is "how concentrated", not "how many independent".

    **The usual ``[1, N]`` bound holds only while every share is non-negative,
    and 1/N does not guarantee that.** A sleeve that hedges the rest can carry a
    *negative* risk contribution; the shares still sum to 1, but ``sum(RC**2)``
    can then exceed 1 and push this below 1. Measured on the live sample: 1/N
    produced a negative share at **12 of 237** rebalances and the lowest
    effective bets reached was **1.84**, so the bound was never actually
    breached here -- but it is not guaranteed, and a check that asserts
    ``>= 1`` is asserting something about this data rather than about the
    formula. ERC cannot hit the case at all: equal positive shares by
    construction.
    """
    if not isinstance(risk_contributions, pd.DataFrame) or risk_contributions.empty:
        raise ComparisonError("risk_contributions must be a non-empty DataFrame of risk shares")
    squared = (risk_contributions.astype("float64") ** 2).sum(axis=1)
    if (squared <= 0).any():
        raise ComparisonError("a rebalance has zero total squared risk share; shares cannot all be 0")
    return (1.0 / squared).rename("effective_bets")


def allocation_table(results: dict) -> pd.DataFrame:
    """Mean capital weight beside mean risk share, per sleeve, per rule.

    The single table this whole project is an argument about. Rows are sleeves,
    columns are ``(rule, capital)`` and ``(rule, risk)`` pairs. Both come from
    the ``BacktestResult`` each rule already produced -- ``benchmarks.fixed_weight``
    recomputes 1/N's risk shares from the engine's own covariance at every
    rebalance, so nothing is estimated here.
    """
    columns = {}
    for name, result in results.items():
        columns[(name, "capital")] = result.weights.mean()
        columns[(name, "risk")] = result.risk_contributions.mean()
    table = pd.DataFrame(columns)
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    return table


def concentration_table(results: dict) -> pd.DataFrame:
    """One row per rule: how concentrated its risk is, and how much it moves.

    ``effective_bets_mean`` is the mean of the per-rebalance values, **not**
    ``1 / mean(sum(RC**2))``. The two are different numbers and the first is the
    one that reads as "the average number of bets": because ``1/x`` is convex,
    aggregating the other way round understates it (measured on 1/N: 2.92
    against 2.83). An earlier draft of this project's write-up quoted the 2.83
    and called it the average; it was the wrong aggregation, not a different
    rounding.
    """
    rows = []
    for name, result in results.items():
        rc = result.risk_contributions
        bets = effective_bets(rc)
        widest = (rc.max() - rc.min()).idxmax()
        rows.append(
            {
                "name": name,
                "effective_bets_mean": float(bets.mean()),
                "effective_bets_min": float(bets.min()),
                "effective_bets_max": float(bets.max()),
                "worst_sleeve_risk_share": float(rc.max(axis=1).max()),
                "widest_sleeve": widest,
                "widest_range_low": float(rc[widest].min()),
                "widest_range_high": float(rc[widest].max()),
                "negative_share_rebalances": int((rc < 0).any(axis=1).sum()),
                "n_rebalances": int(len(rc)),
            }
        )
    return pd.DataFrame(rows).set_index("name").rename_axis(None)


def compare_rules(
    sleeve_returns: pd.DataFrame,
    risk_free: pd.Series,
    target_vol: float = TARGET_VOL,
    match_vol: float | None = None,
    leverage_cap: tuple[float, float] = LEVERAGE_CAP,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    annual_risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
    warmup_trading_days: int = COV_WARMUP_TRADING_DAYS,
) -> RuleComparison:
    """Build the whole comparison from daily sleeve returns and a financing rate.

    Runs the engine twice: once at ``target_vol`` (the published strategy) and
    once at 1/N's realised volatility (the matched column). The second run is an
    engine run on purpose -- ``backtest.py`` owns the one-day alignment between a
    leverage decision and the return it earns, and reproducing it here by hand is
    the bug this project is shaped to prevent.

    ``match_vol`` replaces 1/N's realised volatility as the matched column's
    target. It exists to ask "what if the match point were somewhere else" --
    60/40's volatility, say, or a round number -- and it makes the ``ERC @ 1/N
    vol`` label untrue, so :class:`RuleComparison` records that it was used and
    every printed form says so. 1/N's own volatility is measured and reported
    either way, because that is what the override is being compared against.

    Raises
    ------
    ComparisonError
        If the inputs cannot produce a fair comparison: a calendar mismatch, a
        risk-free series with gaps, or a benchmark with no volatility to match.
    """
    if not isinstance(sleeve_returns, pd.DataFrame) or sleeve_returns.empty:
        raise ComparisonError("sleeve_returns must be a non-empty DataFrame of daily returns")
    if not isinstance(sleeve_returns.index, pd.DatetimeIndex):
        raise ComparisonError("sleeve_returns must carry a DatetimeIndex")

    log.info("Running the engine at the published %.1f%% target...", target_vol * 100)
    engine = run_backtest(
        sleeve_returns,
        risk_free=risk_free,
        target_vol=target_vol,
        leverage_cap=leverage_cap,
        periods_per_year=periods_per_year,
        warmup_trading_days=warmup_trading_days,
    )

    benchmarks = {
        EQUAL_CAPITAL: equal_weight(
            sleeve_returns,
            cov_estimator=estimate_covariance,
            strategy=EQUAL_CAPITAL,
            warmup_trading_days=warmup_trading_days,
        ),
        NAIVE_INV_VOL: naive_risk_parity(
            sleeve_returns,
            cov_estimator=estimate_covariance,
            strategy=NAIVE_INV_VOL,
            warmup_trading_days=warmup_trading_days,
        ),
        SIXTY_FORTY: sixty_forty(
            sleeve_returns,
            cov_estimator=estimate_covariance,
            strategy=SIXTY_FORTY,
            warmup_trading_days=warmup_trading_days,
        ),
        MAX_SHARPE: max_sharpe(
            sleeve_returns,
            cov_estimator=estimate_covariance,
            annual_risk_free_rate=annual_risk_free_rate,
            strategy=MAX_SHARPE,
            warmup_trading_days=warmup_trading_days,
        ),
    }
    # Every rule takes the warm-up together. The benchmarks used to be built at
    # the config default whatever the engine was given, so a non-default warm-up
    # put them on a different calendar and the volatility match below died with
    # a bare KeyError. Checked here, by name, so a calendar mismatch from any
    # other cause is a ComparisonError that says which rule disagrees.
    for name, result in benchmarks.items():
        if not result.returns.index.equals(engine.returns.index):
            raise ComparisonError(
                f"{name}'s return calendar differs from the engine's; "
                "a comparison across different calendars is not a fair one"
            )
    for result in (engine, *benchmarks.values()):
        result.validate()

    # The matched column's target is 1/N's realised volatility over the SAME
    # window the comparison is measured on, which is the engine's return index.
    index = engine.returns.index
    equal_capital_vol = matched_target_vol(
        benchmarks[EQUAL_CAPITAL].returns.loc[index], periods_per_year
    )
    overridden = match_vol is not None
    if overridden:
        if not np.isfinite(match_vol) or match_vol <= MIN_MATCHABLE_VOL:
            raise ComparisonError(
                f"match_vol must be a volatility above {MIN_MATCHABLE_VOL:g}, got {match_vol!r}"
            )
        log.warning(
            "Matched column overridden to %.3f%%; 1/N's own realised volatility is %.3f%%. "
            "The 'ERC @ 1/N vol' rows are not matched to 1/N on this run.",
            match_vol * 100, equal_capital_vol * 100,
        )
    match_target = float(match_vol) if overridden else equal_capital_vol
    log.info("Re-running the engine at %.3f%% volatility...", match_target * 100)
    matched = run_backtest(
        sleeve_returns,
        risk_free=risk_free,
        target_vol=match_target,
        leverage_cap=leverage_cap,
        periods_per_year=periods_per_year,
        warmup_trading_days=warmup_trading_days,
        strategy=f"{DEFAULT_STRATEGY_NAME} @ 1/N vol",
    )
    matched.validate()

    # The construction rests on this: the target feeds the leverage path only,
    # never the ERC solve. If it ever stops being true the matched column is no
    # longer "the same book at a different size" and the comparison is invalid.
    weights_unchanged = bool(matched.weights.equals(engine.weights))
    if not weights_unchanged:
        raise ComparisonError(
            "re-running the engine at a different target_vol changed the ERC weights. "
            "The matched column is only meaningful if the target scales the SAME book, "
            "so either backtest.py now feeds the target into the solve or the two runs "
            "saw different data. Do not report this comparison until that is resolved."
        )

    series = build_series(sleeve_returns, engine, matched, benchmarks, risk_free)
    summary = realized_risk_table(
        series, periods_per_year=periods_per_year, annual_risk_free_rate=annual_risk_free_rate
    )
    matched_realized_vol = float(summary.at[ERC_MATCHED_FINANCED, "annual_volatility"])

    # Every rule is measured AGAINST 1/N, because that is the thing the project
    # asks whether equalising risk beats.
    others = [c for c in COLUMN_ORDER if c != EQUAL_CAPITAL]
    relative = pd.DataFrame(
        {
            name: relative_metrics(series[name], series[EQUAL_CAPITAL], periods_per_year)
            for name in others
        }
    )
    tail = pd.DataFrame(
        {name: tail_block(series[name], series[EQUAL_CAPITAL]) for name in others}
    ).T
    crisis = crisis_table(
        series[ERC_UNLEVERED], series[EQUAL_CAPITAL], periods_per_year=periods_per_year
    )

    # Risk allocation. ERC's own block comes from the UNLEVERED engine
    # result, because the levered columns hold identical weights -- the matched
    # column is the same book at a different size, not a different allocation.
    allocation_results = {
        ERC_UNLEVERED: engine,
        EQUAL_CAPITAL: benchmarks[EQUAL_CAPITAL],
        NAIVE_INV_VOL: benchmarks[NAIVE_INV_VOL],
        SIXTY_FORTY: benchmarks[SIXTY_FORTY],
        MAX_SHARPE: benchmarks[MAX_SHARPE],
    }
    allocation = allocation_table(allocation_results)
    concentration = concentration_table(allocation_results)
    bets = pd.DataFrame(
        {name: effective_bets(r.risk_contributions) for name, r in allocation_results.items()}
    )

    return RuleComparison(
        data_through=str(index[-1].date()),
        start=str(index[0].date()),
        n_observations=len(index),
        n_rebalances=len(engine.weights),
        target_vol=target_vol,
        matched_target_vol=match_target,
        matched_realized_vol=matched_realized_vol,
        matched_target_overridden=overridden,
        matched_leverage=matched.leverage,
        equal_capital_vol=equal_capital_vol,
        rule_results=allocation_results,
        series=series,
        summary=summary,
        relative=relative,
        tail=tail,
        crisis=crisis,
        allocation=allocation,
        concentration=concentration,
        effective_bets=bets,
        weights_unchanged=weights_unchanged,
        meta={
            "engine": engine.meta,
            "matched": matched.meta,
            "benchmarks": {name: r.meta for name, r in benchmarks.items()},
        },
    )


# ---------------------------------------------------------------------------
# The printed report
# ---------------------------------------------------------------------------
# Stated on every run, not only in the README: a reader reads the output.
_CAVEATS: tuple[str, ...] = (
    "Ratios are the annualised ARITHMETIC mean return over volatility at a zero risk-free "
    "rate: not CAGR / volatility (CAGR is lower) and not a true Sharpe. The same convention "
    "is applied to every row, so the ranking is fair even though the level is not.",
    "The matched column is APPROXIMATELY matched. A trailing volatility estimator cannot hit a "
    "target exactly, and this overlay overshoots by roughly 13%, so the realised volatility "
    "printed above the table is the one to read -- not the target.",
    "The matched column is a HINDSIGHT diagnostic, not a strategy anyone could have run. Its "
    "target is 1/N's realised volatility over the whole window, known only at the end. Each "
    "month's leverage uses trailing data only, but the level it aims at was chosen after the "
    "fact. Read it as 'at 1/N's realised risk, which rule earned more', not as a forecast.",
    "Two rows are counterfactual and are marked: the financing-omitted ones. Nobody can borrow "
    "at zero. They are reported only because the difference from their financed twin IS the cost "
    "of leverage, which is one of the three findings.",
    "No transaction costs, bid-ask spread or borrow spread over the bill rate, on any row, so "
    "every level here is optimistic. src/sensitivity.py measures what that is worth, and the "
    "obvious version of the objection is wrong: ERC re-optimises every month, but a 1/N book "
    "drifts off 25% and has to be traded back every month too, so the costs do not fall nearly "
    "as unevenly as 'equal capital barely trades' implies. Read the breakeven there.",
    "One data date per table. These numbers are measured through the date printed above and will "
    "not match a run on a different date.",
)


def format_allocation_table(allocation: pd.DataFrame) -> str:
    """:func:`allocation_table`'s frame as capital-beside-risk percentages."""
    return allocation.map(lambda v: f"{v:.1%}").to_string()


def format_concentration_table(concentration: pd.DataFrame) -> str:
    """:func:`concentration_table`'s frame, one labelled line per rule."""
    rows = {}
    for name in concentration.index:
        c = concentration.loc[name]
        rows[name] = {
            "effective bets (mean)": f"{c['effective_bets_mean']:.2f}",
            "   min - max": f"{c['effective_bets_min']:.2f} - {c['effective_bets_max']:.2f}",
            "worst sleeve's risk share": f"{c['worst_sleeve_risk_share']:.1%}",
            "widest-moving sleeve": str(c["widest_sleeve"]),
            "   its risk share range": f"{c['widest_range_low']:.1%} to {c['widest_range_high']:.1%}",
            "rebalances w/ a negative share": (
                f"{int(c['negative_share_rebalances'])} of {int(c['n_rebalances'])}"
            ),
        }
    return pd.DataFrame(rows).to_string()


def format_comparison_report(comp: RuleComparison) -> str:
    """Human-readable account of one :func:`compare_rules` run."""
    match_source = "an OVERRIDDEN" if comp.matched_target_overridden else "1/N's realised"
    lines = [
        "Equal risk vs equal capital -- multi-asset construction comparison",
        f"  window        : {comp.start} -> {comp.data_through} "
        f"({comp.n_observations} return days, {comp.n_rebalances} rebalances)",
        f"  universe      : {'/'.join(UNIVERSE)}",
        f"  published tgt : {comp.target_vol:.1%} volatility",
        f"  matched column: levered toward {match_source} {comp.matched_target_vol:.2%}, "
        f"and actually realised {comp.matched_realized_vol:.2%} "
        f"({comp.matched_realized_vol / comp.matched_target_vol - 1:+.1%} overshoot)",
        f"  same weights  : {'yes' if comp.weights_unchanged else 'NO -- see the error'} "
        "(the target scales the book, it does not re-solve it)",
    ]
    if comp.matched_target_overridden:
        # The row labels say "@ 1/N vol" and on this run they do not mean it.
        # Said here, where a reader meets the number, rather than only in the
        # caveats at the bottom.
        lines.append(
            f"! the match point was overridden: 1/N's own realised volatility is "
            f"{comp.equal_capital_vol:.2%}, so the '{ERC_MATCHED_FINANCED}' and "
            f"'{ERC_MATCHED_FREE}' rows are NOT matched to 1/N on this run"
        )

    health = comp.meta["engine"].get("run_health", "unknown")
    lines.append(f"  engine health : {health}" if health == "clean" else f"! engine health : {health}")

    lines.append("\nRisk and return, every rule on the same calendar")
    lines.append(textwrap.indent(format_metric_table(comp.summary), "  "))
    lines.append(
        "  VaR / CVaR / worst day / max drawdown: more negative = worse. "
        "(open) = still underwater at the window end."
    )
    counterfactual = [c for c in comp.summary.index if c in COUNTERFACTUAL]
    if counterfactual:
        lines.append(f"  ! counterfactual, not runnable: {', '.join(counterfactual)}")

    lines.append("\nWhere each rule puts its risk -- mean capital weight beside mean risk share")
    lines.append(textwrap.indent(format_allocation_table(comp.allocation), "  "))
    lines.append(
        "  Equal capital is flat by definition and its RISK is not; equal risk is flat by"
    )
    lines.append(
        "  construction and its CAPITAL is not. That divergence is the whole argument."
    )

    lines.append("\nHow concentrated that risk is (effective bets = 1 / sum of squared risk shares)")
    lines.append(textwrap.indent(format_concentration_table(comp.concentration), "  "))
    lines.append(
        f"  {len(UNIVERSE)}.00 means every sleeve carries an equal share of the risk; "
        "1.00 means one sleeve carries all of it."
    )
    _negatives = comp.concentration["negative_share_rebalances"]
    if (_negatives > 0).any():
        _named = ", ".join(str(n) for n in _negatives[_negatives > 0].index)
        lines.append(
            f"  ! {_named} produced a NEGATIVE risk share at some rebalances -- a sleeve"
        )
        lines.append(
            "    hedging the rest. The shares still sum to 1, but effective bets is then no"
        )
        lines.append(
            "    longer bounded below by 1.00, so read it as a measure rather than a count."
        )
    lines.append(
        "  It measures how evenly the ESTIMATED risk is split, not how many independent"
    )
    lines.append(
        "  bets the book holds: equal shares among correlated sleeves are still correlated."
    )
    lines.append(f"\nRelative to {EQUAL_CAPITAL} (daily returns)")
    lines.append(textwrap.indent(format_relative_table(comp.relative, "1/N"), "  "))
    lines.append("  Down capture below 100% = lost less than 1/N on 1/N's down days.")

    lines.append(f"\nCrisis windows (ERC unlevered against {EQUAL_CAPITAL}, measured from the peak close)")
    lines.append(textwrap.indent(format_crisis_table(comp.crisis, "1/N"), "  "))

    lines.append("\nWhat this is, and is not:")
    for i, caveat in enumerate(_CAVEATS, start=1):
        lines.extend(
            textwrap.wrap(f"{i}. {caveat}", width=SUMMARY_LINE_WIDTH, subsequent_indent="   ")
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    # Imported here, not at module scope: src/ must never depend on tests/.
    # Offline and write-free -- fixtures in, a report on stdout, nothing on disk.
    from tests.fixtures import make_fake_returns, make_fake_risk_free

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    fake_returns = make_fake_returns(tickers=UNIVERSE)
    fake_rf = make_fake_risk_free().reindex(fake_returns.index).ffill().bfill()
    comparison = compare_rules(fake_returns, fake_rf)

    print(format_comparison_report(comparison))
    print()
    checks = 0

    def check(label: str, condition: bool) -> None:
        global checks
        if not condition:
            raise AssertionError(f"FAILED: {label}")
        checks += 1
        print(f"  ok   {label}")

    check("every comparison column is present, in order",
          list(comparison.series.columns) == list(COLUMN_ORDER))
    check("the summary has one row per column",
          list(comparison.summary.index) == list(COLUMN_ORDER))
    check("the target did not change the ERC weights", comparison.weights_unchanged)
    check("the matched target equals 1/N's own realised volatility",
          abs(comparison.matched_target_vol
              - matched_target_vol(comparison.series[EQUAL_CAPITAL])) < 1e-12)
    check("the matched column's realised vol is reported separately from its target",
          comparison.matched_realized_vol != comparison.matched_target_vol)
    check("financing is a cost, never a gain: the financed row earns less than its free twin",
          comparison.summary.at[ERC_MATCHED_FINANCED, "cagr"]
          <= comparison.summary.at[ERC_MATCHED_FREE, "cagr"])
    check("...and the same holds at the published target",
          comparison.summary.at[ERC_TARGET_FINANCED, "cagr"]
          <= comparison.summary.at[ERC_TARGET_FREE, "cagr"])
    check("both financing-free rows are marked counterfactual",
          COUNTERFACTUAL == {ERC_TARGET_FREE, ERC_MATCHED_FREE})
    check("1/N is measured against itself nowhere in the relative block",
          EQUAL_CAPITAL not in comparison.relative.columns)
    check("every other column has relative metrics against 1/N",
          len(comparison.relative.columns) == len(COLUMN_ORDER) - 1)
    check("a rule compared with itself would give beta 1 (sanity on the primitive)",
          abs(relative_metrics(comparison.series[EQUAL_CAPITAL],
                               comparison.series[EQUAL_CAPITAL])["beta"] - 1.0) < 1e-9)

    # --- risk allocation ------------------------------------------
    n = len(UNIVERSE)
    check("the allocation block covers exactly the rules with distinct weights",
          sorted({r for r, _ in comparison.allocation.columns}) == sorted(ALLOCATION_RULES))
    check("each rule's mean capital weights sum to 1",
          all(abs(comparison.allocation[(r, "capital")].sum() - 1.0) < 1e-9
              for r in ALLOCATION_RULES))
    check("each rule's mean risk shares sum to 1",
          all(abs(comparison.allocation[(r, "risk")].sum() - 1.0) < 1e-9
              for r in ALLOCATION_RULES))
    check("equal CAPITAL really is flat for 1/N",
          abs(comparison.allocation[(EQUAL_CAPITAL, "capital")] - 1.0 / n).max() < 1e-12)
    check("...while its RISK is not -- that divergence is the thesis",
          abs(comparison.allocation[(EQUAL_CAPITAL, "risk")] - 1.0 / n).max() > 0.01)
    check("equal RISK really is flat for ERC",
          abs(comparison.allocation[(ERC_UNLEVERED, "risk")] - 1.0 / n).max() < 1e-6)
    check("...while its CAPITAL is not",
          abs(comparison.allocation[(ERC_UNLEVERED, "capital")] - 1.0 / n).max() > 0.01)

    # The two limiting cases the formula has to honour, hand-built so neither
    # depends on the fixture. These are what make the measured numbers readable.
    equal = pd.DataFrame([[1.0 / n] * n], columns=list(UNIVERSE))
    one_sleeve = pd.DataFrame([[1.0] + [0.0] * (n - 1)], columns=list(UNIVERSE))
    check(f"effective bets is exactly {n} when every risk share is equal",
          abs(float(effective_bets(equal).iloc[0]) - n) < 1e-12)
    check("effective bets is exactly 1 when one sleeve carries all the risk",
          abs(float(effective_bets(one_sleeve).iloc[0]) - 1.0) < 1e-12)
    check("a negative risk share can push effective bets BELOW 1 -- the [1, N] bound "
          "is not guaranteed, so the docstring says so",
          float(effective_bets(pd.DataFrame([[1.5, -0.5, 0.0, 0.0]],
                                            columns=list(UNIVERSE))).iloc[0]) < 1.0)
    try:
        effective_bets(pd.DataFrame())
        raise AssertionError("FAILED: an empty frame did not raise")
    except ComparisonError:
        check("an empty risk-contribution frame raises rather than returning nothing", True)

    check("ERC takes exactly N effective bets at every rebalance, by construction",
          abs(comparison.effective_bets[ERC_UNLEVERED] - n).max() < 1e-6)
    check("1/N takes fewer, and the concentration table reports the mean of the "
          "per-rebalance values rather than 1 / the mean concentration",
          comparison.concentration.at[EQUAL_CAPITAL, "effective_bets_mean"] < n
          and abs(comparison.concentration.at[EQUAL_CAPITAL, "effective_bets_mean"]
                  - comparison.effective_bets[EQUAL_CAPITAL].mean()) < 1e-12)
    check("the worst single-sleeve risk share is reported and exceeds 1/N for 1/N",
          comparison.concentration.at[EQUAL_CAPITAL, "worst_sleeve_risk_share"] > 1.0 / n)

    print(f"\n{checks} checks passed")
    print("\nFixture data: shape, not magnitude. Nothing was written to disk.")
