"""Equal risk against equal capital -- the comparison this project exists to make
(TASKS.md step 53).

Two construction rules are being compared over one multi-asset book: **equal risk
contribution (ERC)** and **equal capital (1/N)**. The question is what equalising
risk buys, and what it costs in volatility and return.

**Why a volatility-matched column is not optional.** ERC run unlevered is a
lower-volatility book than 1/N -- 6.9% against 8.4% on the live sample -- so
their *returns* are not comparable and a table that puts them side by side on
CAGR is comparing two different amounts of risk. The vol-targeting overlay is
what fixes that: re-run the engine with ``target_vol`` set to 1/N's own realised
volatility and the two books are (approximately, see below) the same size, so the
return difference means something. The financing that levering costs is then part
of the answer rather than a detail, which is why the financed and financing-free
versions are both reported.

**What this module does and does not own.** Every function here is a pure
function of return series -- no network, no disk, no argument parsing. It does
call :func:`src.backtest.run_backtest` a second time for the matched column,
because that is also a pure function and because ``backtest.py`` owns the overlay
alignment that a hand-rolled ``leverage = target / vol`` would get wrong (see
AGENTS.md/CLAUDE.md under ``backtest.py`` and ``overlay.py``). The data fetching
and the printing-to-a-terminal belong to the CLI, step 56.

Metric rows come from :mod:`src.riskmetrics` and the pairwise blocks from
:mod:`src.headtohead`, both of which were salvaged out of removed products for
exactly this; nothing statistical is re-derived here.

**Two measurement facts that shape the output, both measured rather than
assumed** (PLAN.md's Phase 8 section records them):

1. **The overlay overshoots its target by roughly 12.5%.** A 10% target realised
   11.25%; an 8.355% target realised 9.40%. So the matched column is
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
    SUMMARY_LINE_WIDTH,
    TARGET_VOL,
    TRADING_DAYS_PER_YEAR,
    UNIVERSE,
)
from .covariance import estimate_covariance
from .headtohead import crisis_table, format_crisis_table, format_relative_table, relative_metrics, tail_block
from .metrics import summarize
from .overlay import financed_returns, unlevered_returns
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
ERC_MATCHED_FINANCED = "ERC @ 1/N vol, financed"
ERC_MATCHED_FREE = "ERC @ 1/N vol, financing omitted"

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
    ``realized_vol`` is what it actually delivered; they differ by roughly 12.5%
    because a trailing estimator cannot hit a target exactly. Both are carried
    so a report can print the gap rather than imply there isn't one.
    """

    data_through: str
    start: str
    n_observations: int
    n_rebalances: int
    target_vol: float
    matched_target_vol: float
    matched_realized_vol: float
    equal_capital_vol: float
    series: pd.DataFrame
    summary: pd.DataFrame
    relative: pd.DataFrame
    tail: pd.DataFrame
    crisis: pd.DataFrame
    weights_unchanged: bool
    meta: dict


def matched_target_vol(
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """The volatility an ERC book must be levered to in order to match ``benchmark_returns``.

    Just that series' own realised annualised volatility. It is a separate
    function because the number is reported next to the result and because a
    test needs to be able to ask for it without building a whole comparison.
    """
    if not isinstance(benchmark_returns, pd.Series):
        raise ComparisonError("benchmark_returns must be a pandas Series of daily returns")
    vol = float(summarize(benchmark_returns, periods_per_year)["annual_volatility"])
    if not np.isfinite(vol) or vol <= 0.0:
        raise ComparisonError(
            f"benchmark volatility is {vol!r}; there is no target to lever an ERC book to"
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
    matched_unlevered = unlevered_returns(sleeves, matched.weights)
    zero = pd.Series(0.0, index=index)

    columns = {
        ERC_UNLEVERED: unlevered,
        NAIVE_INV_VOL: benchmarks[NAIVE_INV_VOL].returns,
        EQUAL_CAPITAL: benchmarks[EQUAL_CAPITAL].returns,
        MAX_SHARPE: benchmarks[MAX_SHARPE].returns,
        SIXTY_FORTY: benchmarks[SIXTY_FORTY].returns,
        ERC_TARGET_FINANCED: engine.returns,
        ERC_TARGET_FREE: financed_returns(unlevered, engine.leverage, zero),
        ERC_MATCHED_FINANCED: matched.returns,
        ERC_MATCHED_FREE: financed_returns(matched_unlevered, matched.leverage, zero),
    }
    return pd.DataFrame({name: columns[name] for name in COLUMN_ORDER}, index=index)


def compare_rules(
    sleeve_returns: pd.DataFrame,
    risk_free: pd.Series,
    target_vol: float = TARGET_VOL,
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
            sleeve_returns, cov_estimator=estimate_covariance, strategy=EQUAL_CAPITAL
        ),
        NAIVE_INV_VOL: naive_risk_parity(
            sleeve_returns, cov_estimator=estimate_covariance, strategy=NAIVE_INV_VOL
        ),
        SIXTY_FORTY: sixty_forty(
            sleeve_returns, cov_estimator=estimate_covariance, strategy=SIXTY_FORTY
        ),
        MAX_SHARPE: max_sharpe(
            sleeve_returns,
            cov_estimator=estimate_covariance,
            annual_risk_free_rate=annual_risk_free_rate,
            strategy=MAX_SHARPE,
        ),
    }
    for result in (engine, *benchmarks.values()):
        result.validate()

    # The matched column's target is 1/N's realised volatility over the SAME
    # window the comparison is measured on, which is the engine's return index.
    index = engine.returns.index
    equal_capital_vol = matched_target_vol(
        benchmarks[EQUAL_CAPITAL].returns.loc[index], periods_per_year
    )
    log.info("Re-running the engine at 1/N's realised %.3f%% volatility...", equal_capital_vol * 100)
    matched = run_backtest(
        sleeve_returns,
        risk_free=risk_free,
        target_vol=equal_capital_vol,
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

    return RuleComparison(
        data_through=str(index[-1].date()),
        start=str(index[0].date()),
        n_observations=len(index),
        n_rebalances=len(engine.weights),
        target_vol=target_vol,
        matched_target_vol=equal_capital_vol,
        matched_realized_vol=matched_realized_vol,
        equal_capital_vol=equal_capital_vol,
        series=series,
        summary=summary,
        relative=relative,
        tail=tail,
        crisis=crisis,
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
# Stated on every run, not only in PLAN.md: a reader reads the output.
_CAVEATS: tuple[str, ...] = (
    "Ratios are return-to-volatility at a zero risk-free rate, not a true Sharpe. The same "
    "convention is applied to every row, so the ranking is fair even though the level is not.",
    "The matched column is APPROXIMATELY matched. A trailing volatility estimator cannot hit a "
    "target exactly, and this overlay overshoots by roughly 12.5%, so the realised volatility "
    "printed above the table is the one to read -- not the target.",
    "Two rows are counterfactual and are marked: the financing-omitted ones. Nobody can borrow "
    "at zero. They are reported only because the difference from their financed twin IS the cost "
    "of leverage, which is one of the three findings.",
    "No transaction costs, bid-ask spread or borrow spread over the bill rate, on any row. ERC "
    "re-optimises every month and 1/N barely trades, so costs would not fall evenly -- they would "
    "hurt ERC more. Step 58 measures that; until it does, these levels are optimistic for ERC.",
    "One data date per table. These numbers are measured through the date printed above and will "
    "not match a run on a different date.",
)


def format_comparison_report(comp: RuleComparison) -> str:
    """Human-readable account of one :func:`compare_rules` run."""
    lines = [
        "Equal risk vs equal capital -- multi-asset construction comparison",
        f"  window        : {comp.start} -> {comp.data_through} "
        f"({comp.n_observations} return days, {comp.n_rebalances} rebalances)",
        f"  universe      : {'/'.join(UNIVERSE)}",
        f"  published tgt : {comp.target_vol:.1%} volatility",
        f"  matched column: levered toward 1/N's realised {comp.matched_target_vol:.2%}, "
        f"and actually realised {comp.matched_realized_vol:.2%} "
        f"({comp.matched_realized_vol / comp.matched_target_vol - 1:+.1%} overshoot)",
        f"  same weights  : {'yes' if comp.weights_unchanged else 'NO -- see the error'} "
        "(the target scales the book, it does not re-solve it)",
    ]

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

    print(f"\n{checks} checks passed")
    print("\nFixture data: shape, not magnitude. Nothing was written to disk.")
