"""Head-to-head: the risk-parity strategy against buy-and-hold of any ticker (step 44).

Two other tools answer nearby questions and neither is this one.
``run_backtest.py`` compares the strategy only against four fixed, unlevered
benchmarks; ``analyze_position.py`` asks what *adding* a stock does to the book.
This module answers the plain version a reader asks first: **how did the
strategy do against simply buying and holding this ticker, and what was the
difference in risk?**

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

The metrics come from :mod:`src.incremental` (``risk_summary`` and its table
formatter), so the two reports print the same rows the same way. What is new here
is the *relative* block -- correlation, beta, tracking error, up/down capture and
what the strategy did on the ticker's worst days -- and the crisis-window table.

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
    COMPARE_MIN_CALENDAR_MATCH,
    COMPARE_MIN_OVERLAP_DAYS,
    CRISIS_WINDOWS,
    LEVERAGE_CAP,
    SUMMARY_LINE_WIDTH,
    TARGET_VOL,
    TRADING_DAYS_PER_YEAR,
    UNIVERSE,
    VAR_CONFIDENCE,
)
from .data import DataUnavailableError
from .incremental import format_metric_table, historical_var, realized_risk_table
from .metrics import _ZERO_VARIANCE_EPS, resolve_window, summarize

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
# Relative metrics
# ---------------------------------------------------------------------------
def relative_metrics(
    strategy: pd.Series,
    ticker: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """How ``strategy`` moves against ``ticker``, on identically-indexed daily returns.

    * ``correlation`` -- Pearson, daily.
    * ``beta`` -- ``cov(strategy, ticker) / var(ticker)``: the strategy's move per
      unit of ticker move. NaN if the ticker has no variance.
    * ``tracking_error`` -- annualised std of the daily difference.
    * ``up_capture`` / ``down_capture`` -- the strategy's mean return on the days
      the ticker rose (fell), as a multiple of the ticker's own mean on those
      days. Arithmetic daily means. A down capture below 1 means the strategy lost
      less than the ticker on the ticker's down days; a negative one means it
      gained.
    """
    if not strategy.index.equals(ticker.index):
        raise ValueError("relative_metrics needs both series on the same index")
    s = strategy.astype("float64")
    t = ticker.astype("float64")

    if len(t) < 2 or t.std(ddof=1) < _ZERO_VARIANCE_EPS:
        beta = float("nan")
        correlation = float("nan")
    else:
        beta = float(s.cov(t) / t.var(ddof=1))
        correlation = float(s.corr(t))
    tracking_error = (
        float((s - t).std(ddof=1) * np.sqrt(periods_per_year)) if len(t) >= 2 else float("nan")
    )

    up, down = t > 0, t < 0
    up_capture = float(s[up].mean() / t[up].mean()) if up.any() else float("nan")
    down_capture = float(s[down].mean() / t[down].mean()) if down.any() else float("nan")
    return {
        "correlation": correlation,
        "beta": beta,
        "tracking_error": tracking_error,
        "up_capture": up_capture,
        "down_capture": down_capture,
    }


def tail_block(
    strategy: pd.Series,
    ticker: pd.Series,
    confidence: float = VAR_CONFIDENCE,
) -> dict[str, float]:
    """What the strategy did on the ticker's worst days.

    The days are those at or below the ticker's own historical VaR at
    ``confidence`` (the worst 5% by default). ``strategy_beat_share`` is the
    share of those days on which the strategy returned more than the ticker --
    the direct test of "does the diversifier hold up when the ticker falls",
    which correlation and beta average over the whole sample and can hide.
    """
    if not strategy.index.equals(ticker.index):
        raise ValueError("tail_block needs both series on the same index")
    var = historical_var(ticker, confidence)
    if np.isnan(var):
        return {
            "n_days": 0,
            "ticker_mean": float("nan"),
            "strategy_mean": float("nan"),
            "strategy_beat_share": float("nan"),
        }
    mask = ticker <= var
    return {
        "n_days": int(mask.sum()),
        "ticker_mean": float(ticker[mask].mean()),
        "strategy_mean": float(strategy[mask].mean()),
        "strategy_beat_share": float((strategy[mask] > ticker[mask]).mean()),
    }


# ---------------------------------------------------------------------------
# Crisis windows
# ---------------------------------------------------------------------------
def crisis_table(
    strategy: pd.Series,
    ticker: pd.Series,
    windows: dict[str, tuple[str, str]] | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Total return and max drawdown through each crisis window, strategy vs ticker.

    Each window is measured **from the peak close**: the returns kept are those
    stamped strictly *after* the window's start date, because the start date is
    the peak itself and its own return happened on the way up. This is the same
    rule ``metrics.slice_to_window`` applies to a ``BacktestResult`` (see its
    docstring for why); it is repeated here because these are bare series.

    A window is ``covered`` only if the series reach both ends of it. The two
    series share one index here, so it is the *later* starter that decides -- a
    ticker that listed after a window's peak gets ``NaN``, not a partial window
    that reads like a full one.
    """
    windows = CRISIS_WINDOWS if windows is None else windows
    if not strategy.index.equals(ticker.index):
        raise ValueError("crisis_table needs both series on the same index")

    rows = []
    for name in windows:
        start, end = resolve_window((windows[name][0], windows[name][1]))
        covered = bool(strategy.index[0] <= start and (end is None or strategy.index[-1] >= end))
        row: dict[str, object] = {"window": name, "covered": covered}
        for label, series in (("strategy", strategy), ("ticker", ticker)):
            if covered:
                keep = series.index > start
                if end is not None:
                    keep &= series.index <= end
                stats = summarize(series[keep], periods_per_year, name=label)
                row[f"{label}_return"] = stats["total_return"]
                row[f"{label}_max_drawdown"] = stats["max_drawdown"]
            else:
                row[f"{label}_return"] = float("nan")
                row[f"{label}_max_drawdown"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("window")


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TickerComparison:
    """Everything one head-to-head run produces.

    ``series`` is the aligned daily returns the other tables were computed from,
    kept so a caller (or a test) can recompute anything here independently.
    ``summary`` is :func:`src.incremental.realized_risk_table`'s frame, one row
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
    )


# ---------------------------------------------------------------------------
# The report compare_ticker.py prints
# ---------------------------------------------------------------------------
# Stated in the report every run, not only in PLAN.md, for the same reason the
# incremental report does: a CLI user reads the output, not the design document.
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


def _relative_table(comp: TickerComparison) -> str:
    rows = {
        "correlation": ("correlation", "ratio"),
        f"beta to {comp.candidate}": ("beta", "ratio"),
        "tracking error (annual)": ("tracking_error", "pct"),
        "up capture": ("up_capture", "share"),
        "down capture": ("down_capture", "share"),
    }
    cells: dict[str, dict[str, str]] = {}
    for label, (key, kind) in rows.items():
        cells[label] = {}
        for column in comp.relative.columns:
            value = comp.relative.at[key, column]
            if pd.isna(value):
                text = "--"
            elif kind == "pct":
                text = f"{value:.2%}"
            elif kind == "share":
                text = f"{value:.0%}"
            else:
                text = f"{value:.3f}"
            cells[label][column] = text
    return pd.DataFrame(cells).T.to_string()


def _crisis_text(comp: TickerComparison) -> str:
    def pct(value: float) -> str:
        return "n/a" if pd.isna(value) else f"{value:.2%}"

    table = pd.DataFrame(
        {
            "strategy return": comp.crisis["strategy_return"].map(pct),
            "strategy max DD": comp.crisis["strategy_max_drawdown"].map(pct),
            f"{comp.candidate} return": comp.crisis["ticker_return"].map(pct),
            f"{comp.candidate} max DD": comp.crisis["ticker_max_drawdown"].map(pct),
        }
    )
    return table.rename_axis(None).to_string()


def format_comparison_report(comp: TickerComparison) -> str:
    """Human-readable account of one :func:`compare_to_ticker` run."""
    lines = [
        f"{comp.strategy_name} vs {comp.candidate} buy-and-hold -- head-to-head comparison",
        f"  overlap window: {comp.window_start.date()} -> {comp.window_end.date()} "
        f"({comp.n_observations} obs; {comp.calendar_match:.0%} of {comp.candidate}'s "
        "trading days are strategy days)",
        f"  strategy: {len(UNIVERSE)}-sleeve equal-risk-contribution book "
        f"({'/'.join(UNIVERSE)}), levered to {TARGET_VOL:.0%} vol "
        f"(cap {LEVERAGE_CAP[0]}-{LEVERAGE_CAP[1]}x), financed at DTB3; run over its full "
        "history, then cut to this window",
        f"  {comp.candidate}: bought and held, total return (dividends reinvested)",
    ]

    lines.append("\nRisk and return on the overlap window")
    lines.append(textwrap.indent(format_metric_table(comp.summary), "  "))
    lines.append(
        "  VaR / CVaR / worst day / max drawdown: more negative = worse. "
        "(open) = still underwater at window end."
    )

    lines.append(f"\nRelative to {comp.candidate} (daily returns)")
    lines.append(textwrap.indent(_relative_table(comp), "  "))
    lines.append("  Down capture below 100% = lost less than the ticker on its down days.")

    tail_days = int(comp.tail["n_days"].iloc[0])
    tail_mean = comp.tail["ticker_mean"].iloc[0]
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
    lines.append(textwrap.indent(_crisis_text(comp), "  "))
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
