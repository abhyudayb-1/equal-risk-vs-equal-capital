"""Head-to-head comparison of two daily return series, on one shared index.

Every function takes a ``strategy`` series and a ``benchmark`` series indexed
by the same dates and answers "how did the first move against the second", so
this is the module for any pairwise comparison the project wants: an equal-risk
book against an equal-capital one, a levered book against its unlevered self.

Consumed by :mod:`src.comparison`, which measures every construction rule
against 1/N: :func:`relative_metrics` and :func:`tail_block` per rule,
:func:`crisis_table` for the per-window comparison, and both formatters for the
printed report. Changing a signature here changes that report.

The parameters are named for their *roles*, not their contents: ``strategy`` is
the thing being judged and ``benchmark`` is the thing it is judged against. The
pair is deliberately **not** symmetric -- up and down capture are measured on
the benchmark's up and down days, and the tail block is measured on the
benchmark's worst days -- so passing them the other way round answers a
different (also valid) question. The dataclass, the data fetching and the
report prose that wrap these live with their caller.

What is here:

* :func:`relative_metrics` -- correlation, beta, tracking error and up/down
  capture, each with an exact definition in its docstring, because "beta" and
  "capture" are both names with more than one convention behind them.
* :func:`tail_block` -- what the strategy did on the benchmark's worst days,
  which correlation and beta average over the whole sample and can hide.
* :func:`crisis_table` -- total return and max drawdown through each configured
  crisis window, measured **from the peak close**, with a window the series do
  not fully cover reported as ``NaN`` rather than as a partial window.
* :func:`format_relative_table` / :func:`format_crisis_table` -- the printed
  forms, taking a frame and a label so any report can print identical tables.
  Undefined values render ``--`` and an uncovered crisis window ``n/a``; the
  metric rows themselves are ``riskmetrics.format_metric_table``'s job.

Every function is a pure function of return series, so this module runs
identically on fixtures and on live data and needs no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CRISIS_WINDOWS,
    TRADING_DAYS_PER_YEAR,
    VAR_CONFIDENCE,
)
from .metrics import _ZERO_VARIANCE_EPS, resolve_window, summarize
from .riskmetrics import historical_var


# ---------------------------------------------------------------------------
# Relative metrics
# ---------------------------------------------------------------------------
def relative_metrics(
    strategy: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """How ``strategy`` moves against ``benchmark``, on identically-indexed daily returns.

    * ``correlation`` -- Pearson, daily.
    * ``beta`` -- ``cov(strategy, benchmark) / var(benchmark)``: the strategy's move per
      unit of benchmark move. NaN if the benchmark has no variance.
    * ``tracking_error`` -- annualised std of the daily difference.
    * ``up_capture`` / ``down_capture`` -- the strategy's mean return on the days
      the benchmark rose (fell), as a multiple of the benchmark's own mean on those
      days. Arithmetic daily means. A down capture below 1 means the strategy lost
      less than the benchmark on the benchmark's down days; a negative one means it
      gained.
    """
    if not strategy.index.equals(benchmark.index):
        raise ValueError("relative_metrics needs both series on the same index")
    s = strategy.astype("float64")
    t = benchmark.astype("float64")

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
    benchmark: pd.Series,
    confidence: float = VAR_CONFIDENCE,
) -> dict[str, float]:
    """What the strategy did on the benchmark's worst days.

    The days are those at or below the benchmark's own historical VaR at
    ``confidence`` (the worst 5% by default). ``strategy_beat_share`` is the
    share of those days on which the strategy returned more than the benchmark --
    the direct test of "does the diversifier hold up when the benchmark falls",
    which correlation and beta average over the whole sample and can hide.
    """
    if not strategy.index.equals(benchmark.index):
        raise ValueError("tail_block needs both series on the same index")
    var = historical_var(benchmark, confidence)
    if np.isnan(var):
        return {
            "n_days": 0,
            "benchmark_mean": float("nan"),
            "strategy_mean": float("nan"),
            "strategy_beat_share": float("nan"),
        }
    mask = benchmark <= var
    return {
        "n_days": int(mask.sum()),
        "benchmark_mean": float(benchmark[mask].mean()),
        "strategy_mean": float(strategy[mask].mean()),
        "strategy_beat_share": float((strategy[mask] > benchmark[mask]).mean()),
    }


# ---------------------------------------------------------------------------
# Crisis windows
# ---------------------------------------------------------------------------
def crisis_table(
    strategy: pd.Series,
    benchmark: pd.Series,
    windows: dict[str, tuple[str, str]] | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Total return and max drawdown through each crisis window, strategy vs benchmark.

    Each window is measured **from the peak close**: the returns kept are those
    stamped strictly *after* the window's start date, because the start date is
    the peak itself and its own return happened on the way up. This is the same
    rule ``metrics.slice_to_window`` applies to a ``BacktestResult`` (see its
    docstring for why); it is repeated here because these are bare series.

    A window is ``covered`` only if the series reach both ends of it. The two
    series share one index here, so whichever of them starts later is what
    decides: if either begins after a window's peak, that window is reported as
    ``NaN``, never as a partial window that reads like a full one.
    """
    windows = CRISIS_WINDOWS if windows is None else windows
    if not strategy.index.equals(benchmark.index):
        raise ValueError("crisis_table needs both series on the same index")

    rows = []
    for name in windows:
        start, end = resolve_window((windows[name][0], windows[name][1]))
        covered = bool(strategy.index[0] <= start and (end is None or strategy.index[-1] >= end))
        row: dict[str, object] = {"window": name, "covered": covered}
        for label, series in (("strategy", strategy), ("benchmark", benchmark)):
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
# Printed forms
# ---------------------------------------------------------------------------
# Taking a frame and a label rather than a report object, so any caller can
# print these tables and two reports cannot show the same numbers under
# different labels or formats -- the same reason
# riskmetrics.format_metric_table exists.
def format_relative_table(relative: pd.DataFrame, benchmark_label: str) -> str:
    """:func:`relative_metrics`' rows as a table, one column per strategy.

    ``relative`` is indexed by the metric keys :func:`relative_metrics` returns,
    with one column per series being compared against ``benchmark_label``.
    Captures print as whole percentages (they are multiples, read as "kept 78%
    of the move"), tracking error as a percentage, the rest to three decimals.
    An undefined value prints ``--`` rather than a zero.
    """
    rows = {
        "correlation": ("correlation", "ratio"),
        f"beta to {benchmark_label}": ("beta", "ratio"),
        "tracking error (annual)": ("tracking_error", "pct"),
        "up capture": ("up_capture", "share"),
        "down capture": ("down_capture", "share"),
    }
    cells: dict[str, dict[str, str]] = {}
    for label, (key, kind) in rows.items():
        cells[label] = {}
        for column in relative.columns:
            value = relative.at[key, column]
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


def format_crisis_table(crisis: pd.DataFrame, benchmark_label: str) -> str:
    """:func:`crisis_table`'s frame as a table, one row per crisis window.

    An uncovered window prints ``n/a`` in every cell -- never ``0.00%``, which
    would read as "came through it flat" instead of "was not trading yet".
    """
    def pct(value: float) -> str:
        return "n/a" if pd.isna(value) else f"{value:.2%}"

    table = pd.DataFrame(
        {
            "strategy return": crisis["strategy_return"].map(pct),
            "strategy max DD": crisis["strategy_max_drawdown"].map(pct),
            f"{benchmark_label} return": crisis["benchmark_return"].map(pct),
            f"{benchmark_label} max DD": crisis["benchmark_max_drawdown"].map(pct),
        }
    )
    return table.rename_axis(None).to_string()
