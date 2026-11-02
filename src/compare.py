"""Head-to-head: the risk-parity strategy against buy-and-hold of any ticker (step 44).

``run_backtest.py`` compares the strategy only against four fixed, unlevered
benchmarks. This module answers the plain version a reader asks first: **how
did the strategy do against simply buying and holding this ticker, and what was
the difference in risk?**

It compares, on the dates both series exist (the "overlap window"):

* the strategy -- the engine's daily returns, levered to the vol target and
  **net of financing**, exactly what ``run_backtest.py`` reports;
* the same equal-risk book **unlevered** -- the ticker is unlevered, so this is
  the like-for-like risk comparison and the levered column is not;
* the ticker's total-return daily returns.

Everything here is a pure function of return series, so it runs identically on
fixtures and on live data. The engine run and the data fetching live in
``compare_ticker.py``. The strategy is run over its whole 2006+ history first and
*then* cut to the overlap window, so its weights and leverage at any date never
depend on when the ticker happened to start trading.

Almost nothing here is specific to a ticker, and what is not has moved out. The
metric rows come from :mod:`src.riskmetrics` (``risk_summary`` and its table
formatter), and the relative block -- correlation, beta, tracking error, up/down
capture, what the strategy did on the benchmark's worst days, the crisis-window
table and both printed forms -- from :mod:`src.headtohead`, which compares any
two return series on a shared index. What remains this module's own is the part
that *is* about a ticker: aligning a live price series onto the strategy's
calendar, the two-tier trading-calendar guard, and the report prose.

Run a self-check from the repo root (offline, fixtures only, writes nothing)::

    .venv/bin/python -m src.compare
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import DEFAULT_STRATEGY_NAME
from .config import (
    ANNUAL_RISK_FREE_RATE,
    COMPARE_CALENDAR_WARN_BELOW,
    COMPARE_MIN_CALENDAR_MATCH,
    COMPARE_MIN_OVERLAP_DAYS,
    LEVERAGE_CAP,
    SUMMARY_LINE_WIDTH,
    TARGET_VOL,
    TRADING_DAYS_PER_YEAR,
    UNIVERSE,
    VAR_CONFIDENCE,
)
from .data import DataUnavailableError
from .headtohead import (
    crisis_table,
    format_crisis_table,
    format_relative_table,
    relative_metrics,
    tail_block,
)
from .riskmetrics import format_metric_table, realized_risk_table

log = logging.getLogger(__name__)

STRATEGY_LABEL = DEFAULT_STRATEGY_NAME
UNLEVERED_LABEL = "Risk parity (unlevered)"


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def _require_series(name: str, series: pd.Series) -> None:
    if not isinstance(series, pd.Series):
        raise ValueError(f"{name} must be a pandas Series of daily returns")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must be indexed by date")
    if not series.index.is_monotonic_increasing or series.index.has_duplicates:
        raise ValueError(f"{name}'s index must be strictly increasing (sorted, no duplicates)")
    if not np.isfinite(series.to_numpy(dtype="float64")).all():
        raise ValueError(f"{name} contains NaN or inf")


def _calendar_match(strategy_index: pd.DatetimeIndex, candidate_index: pd.DatetimeIndex) -> float:
    """Share of the ticker's trading days, inside the strategy's date range, that
    are also strategy trading days.

    ~1.0 for a US listing. Meaningfully below it means a different exchange
    calendar, where intersecting the two would pair a multi-day return on one
    side with a single day's on the other. Measured on the ticker's days rather
    than the strategy's so a young ticker does not look like a mismatch merely
    because the strategy has years of history it does not.
    """
    lo, hi = strategy_index[0], strategy_index[-1]
    in_range = candidate_index[(candidate_index >= lo) & (candidate_index <= hi)]
    if len(in_range) == 0:
        return 0.0
    return len(in_range.intersection(strategy_index)) / len(in_range)


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TickerComparison:
    """Everything one head-to-head run produces.

    ``series`` is the aligned daily returns the other tables were computed from,
    kept so a caller (or a test) can recompute anything here independently.
    ``summary`` is :func:`src.riskmetrics.realized_risk_table`'s frame, one row
    per column of ``series``. ``relative`` and ``tail`` are indexed by the
    risk-parity columns only; the ticker is the thing they are measured against.
    """

    candidate: str
    strategy_name: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_observations: int
    calendar_match: float
    series: pd.DataFrame
    summary: pd.DataFrame
    relative: pd.DataFrame
    tail: pd.DataFrame
    crisis: pd.DataFrame
    calendar_warning: str | None = None


def _calendar_warning(candidate: str, match: float) -> str:
    return (
        f"only {match:.1%} of {candidate}'s trading days are also strategy trading days. "
        f"That is a different exchange's calendar: on days one market was shut and the "
        f"other open a multi-day return is paired with a one-day one, and a market that "
        f"closes at a different time of day reacts to US news on a different date. "
        f"Correlation, beta, capture and the tail block are likely UNDERSTATED; treat "
        f"them as indicative, not exact."
    )


def compare_to_ticker(
    strategy: pd.Series,
    candidate_returns: pd.Series,
    candidate: str,
    unlevered: pd.Series | None = None,
    strategy_name: str = STRATEGY_LABEL,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    annual_risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
    min_overlap_days: int = COMPARE_MIN_OVERLAP_DAYS,
    min_calendar_match: float = COMPARE_MIN_CALENDAR_MATCH,
    warn_calendar_below: float = COMPARE_CALENDAR_WARN_BELOW,
    confidence: float = VAR_CONFIDENCE,
) -> TickerComparison:
    """Compare the strategy's daily returns with a ticker's, on their overlap.

    Parameters
    ----------
    strategy:
        The engine's daily returns (levered, net of financing).
    candidate_returns:
        The ticker's total-return daily returns.
    unlevered:
        The same book without leverage, if available. Reported as its own column
        and its own relative metrics -- the like-for-like comparison, since the
        ticker is not levered.

    Raises
    ------
    DataUnavailableError
        If the ticker's trading calendar does not match the strategy's (see
        :func:`_calendar_match`), or the overlap is shorter than
        ``min_overlap_days``. Both are properties of the data, not of the call.

    Two tiers on the calendar, deliberately. Below ``min_calendar_match`` (crypto,
    which trades weekends) it raises: the series cannot be aligned. Between that
    and ``warn_calendar_below`` (a foreign weekday exchange) it runs and sets
    ``calendar_warning``, which the report prints: the comparison is usable but
    its relative metrics are biased, and silence there would be the same failure
    the refusal exists to prevent, one notch quieter.
    ValueError
        If an input is not a clean, dated, finite return series, or ``candidate``
        collides with a strategy column name.
    """
    candidate = candidate.strip().upper()
    _require_series("strategy", strategy)
    _require_series("candidate_returns", candidate_returns)
    if unlevered is not None:
        _require_series("unlevered", unlevered)

    strategy_columns = {strategy_name: strategy}
    if unlevered is not None:
        strategy_columns[UNLEVERED_LABEL] = unlevered
    if candidate in strategy_columns:
        raise ValueError(f"the ticker label {candidate!r} collides with a strategy column")

    match = _calendar_match(strategy.index, candidate_returns.index)
    if match < min_calendar_match:
        raise DataUnavailableError(
            f"only {match:.0%} of {candidate}'s trading days are also strategy trading "
            f"days (need {min_calendar_match:.0%}). That is what a listing on another "
            "exchange's calendar looks like, and comparing the two would pair multi-day "
            "returns with single-day ones."
        )

    common = strategy.index.intersection(candidate_returns.index)
    for series in strategy_columns.values():
        common = common.intersection(series.index)
    if len(common) < min_overlap_days:
        raise DataUnavailableError(
            f"{candidate} overlaps the strategy on only {len(common)} trading days, "
            f"fewer than COMPARE_MIN_OVERLAP_DAYS={min_overlap_days}; too few for a "
            "drawdown or a tail estimate to mean anything"
        )

    aligned = pd.DataFrame(
        {name: series.loc[common] for name, series in strategy_columns.items()}
        | {candidate: candidate_returns.loc[common]},
        index=common,
    )

    summary = realized_risk_table(aligned, periods_per_year, annual_risk_free_rate)

    relative = pd.DataFrame(
        {
            name: relative_metrics(aligned[name], aligned[candidate], periods_per_year)
            for name in strategy_columns
        }
    )
    tail = pd.DataFrame(
        {name: tail_block(aligned[name], aligned[candidate], confidence) for name in strategy_columns}
    ).T
    crisis = crisis_table(aligned[strategy_name], aligned[candidate], periods_per_year=periods_per_year)

    return TickerComparison(
        candidate=candidate,
        strategy_name=strategy_name,
        window_start=common[0],
        window_end=common[-1],
        n_observations=len(common),
        calendar_match=match,
        series=aligned,
        summary=summary,
        relative=relative,
        tail=tail,
        crisis=crisis,
        calendar_warning=(
            _calendar_warning(candidate, match) if match < warn_calendar_below else None
        ),
    )


# ---------------------------------------------------------------------------
# The report compare_ticker.py prints
# ---------------------------------------------------------------------------
# Stated in the report every run, not only in PLAN.md: a CLI user reads the
# output, not the design document.
_CAVEATS: tuple[str, ...] = (
    "Overlap window only: both sides are measured on the dates the ticker exists, "
    "not over the strategy's full 2006-to-today history, so these numbers will not "
    "match run_backtest.py's headline results.",
    "Not like-for-like on leverage: the strategy is levered to the vol target and "
    "charged financing at DTB3; the ticker is not levered at all. The unlevered "
    "column and the scale-free ratios (Sharpe, Sortino, Calmar) are the fair read.",
    "Chosen with hindsight: a ticker someone asks about today is, more often than "
    "not, one that did well or is well known, so it flatters the ticker against a "
    "diversified strategy that never got to pick it.",
    "No transaction costs, bid-ask spread or borrow spread over the bill rate, on "
    "either side. Levered results in particular are still optimistic.",
    "Short windows make the tail measures noisy: a 5% tail is 5% of the observation "
    "count above, and a max drawdown is a single episode.",
)


def format_comparison_report(comp: TickerComparison) -> str:
    """Human-readable account of one :func:`compare_to_ticker` run."""
    lines = [
        f"{comp.strategy_name} vs {comp.candidate} buy-and-hold -- head-to-head comparison",
        f"  overlap window: {comp.window_start.date()} -> {comp.window_end.date()} "
        f"({comp.n_observations} obs; {comp.calendar_match:.1%} of {comp.candidate}'s "
        "trading days are strategy days)",
        f"  strategy: {len(UNIVERSE)}-sleeve equal-risk-contribution book "
        f"({'/'.join(UNIVERSE)}), levered to {TARGET_VOL:.0%} vol "
        f"(cap {LEVERAGE_CAP[0]}-{LEVERAGE_CAP[1]}x), financed at DTB3; run over its full "
        "history, then cut to this window",
        f"  {comp.candidate}: bought and held, total return (dividends reinvested)",
    ]
    if comp.calendar_warning:
        lines.extend(
            textwrap.wrap(
                f"! calendar warning: {comp.calendar_warning}",
                width=SUMMARY_LINE_WIDTH,
                subsequent_indent="    ",
            )
        )

    lines.append("\nRisk and return on the overlap window")
    lines.append(textwrap.indent(format_metric_table(comp.summary), "  "))
    lines.append(
        "  VaR / CVaR / worst day / max drawdown: more negative = worse. "
        "(open) = still underwater at window end."
    )

    lines.append(f"\nRelative to {comp.candidate} (daily returns)")
    lines.append(textwrap.indent(format_relative_table(comp.relative, comp.candidate), "  "))
    lines.append("  Down capture below 100% = lost less than the ticker on its down days.")

    tail_days = int(comp.tail["n_days"].iloc[0])
    tail_mean = comp.tail["benchmark_mean"].iloc[0]
    lines.append(
        f"\nOn {comp.candidate}'s worst {1 - VAR_CONFIDENCE:.0%} of days "
        f"({tail_days} days, {comp.candidate} averaged "
        f"{'--' if pd.isna(tail_mean) else format(tail_mean, '+.2%')})"
    )
    for name in comp.tail.index:
        mean = comp.tail.at[name, "strategy_mean"]
        share = comp.tail.at[name, "strategy_beat_share"]
        mean_text = "--" if pd.isna(mean) else format(mean, "+.2%")
        share_text = "--" if pd.isna(share) else format(share, ".0%")
        lines.append(
            f"  {name:<28} averaged {mean_text}, and beat {comp.candidate} on {share_text} of them"
        )

    lines.append("\nCrisis windows (measured from the peak close)")
    lines.append(textwrap.indent(format_crisis_table(comp.crisis, comp.candidate), "  "))
    if not comp.crisis["covered"].all():
        lines.append(
            f"  n/a = the window is not fully inside the overlap ({comp.candidate} "
            "or the strategy was not yet trading)."
        )

    lines.append("\nWhat this is, and is not:")
    for i, caveat in enumerate(_CAVEATS, start=1):
        lines.extend(
            textwrap.wrap(f"{i}. {caveat}", width=SUMMARY_LINE_WIDTH, subsequent_indent="   ")
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    # Imported here, not at module scope: src/ must never depend on tests/.
    # Offline and write-free: fixtures in, a report on stdout, nothing on disk.
    from tests.fixtures import make_fake_result, make_fake_returns

    from .config import GOLD_TICKER
    from .overlay import unlevered_returns

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    fake_returns = make_fake_returns(tickers=(GOLD_TICKER, *UNIVERSE))
    fake_result = make_fake_result()
    book_returns = fake_returns[list(UNIVERSE)].loc[fake_result.returns.index]

    comparison = compare_to_ticker(
        fake_result.returns,
        fake_returns[GOLD_TICKER],
        GOLD_TICKER,
        unlevered=unlevered_returns(book_returns, fake_result.weights),
    )
    print(format_comparison_report(comparison))
    print("\nFixture data: shape, not magnitude. Nothing was written to disk.")
