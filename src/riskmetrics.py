"""Realized risk and return metrics for a periodic return series.

Every function here is a pure function of a return series, so any comparison
of two books -- an equal-risk allocation against an equal-capital one, a
levered book against its unlevered self -- can use the same rows.

Consumed by :mod:`src.comparison`, the equal-risk-against-equal-capital
comparison, via :func:`realized_risk_table` and
:func:`format_metric_table`; ``src/headtohead.py`` also uses
:func:`historical_var`. Changing a signature here changes that report.

**Why these live here and not in ``src/metrics.py``.** ``metrics.py`` is
ported from ``abhyudayb-1/sp500-factor-backtester`` and kept close to the
original. Everything below
*follows* its conventions -- ``dropna`` and float64 on the way in, NaN on
input too short to mean anything, ``periods_per_year`` required wherever a
number is annualised, and the ``_ZERO_VARIANCE_EPS`` guard so a degenerate
series reads as undefined rather than as a ratio of ~1e16 -- but it is new
code, so it sits in its own module beside the port rather than inside it.
Like ``metrics.py`` it carries no ``__main__`` self-check; its coverage is
``tests/test_riskmetrics.py``, whose expected values are all derived by hand
from the definitions rather than by calling these functions a second way.

What is here:

* :func:`historical_var` / :func:`historical_cvar` -- empirical tail loss at
  ``config.VAR_CONFIDENCE``, making no normality assumption.
* :func:`downside_deviation` / :func:`sortino_ratio` -- risk counted only
  below the risk-free threshold.
* :func:`calmar_ratio` / :func:`drawdown_duration` -- drawdown depth against
  growth, and how long the longest stretch underwater lasted, flagged when it
  is still open at the end of the window.
* :func:`risk_summary` -- ``metrics.summarize``'s row extended with all of the
  above. :func:`realized_risk_table` applies it to one column per book.
* :func:`format_metric_table` -- the printed table, shared by every report
  that prints these rows so two reports cannot describe the same metric under
  different labels or formats.

Undefined metrics are NaN throughout and print as ``--``. Nothing here
substitutes a large finite number for an infinite ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ANNUAL_RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    VAR_CONFIDENCE,
)
from .metrics import (
    _ZERO_VARIANCE_EPS,
    _drawdown_series,
    cagr,
    max_drawdown,
    summarize,
)


def _clean(returns: pd.Series) -> pd.Series:
    return returns.dropna().astype("float64")


def _check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")


def historical_var(returns: pd.Series, confidence: float = VAR_CONFIDENCE) -> float:
    """Historical value-at-risk: the ``1 - confidence`` quantile of the returns.

    Returned as a **negative return**, the same sign convention as
    ``metrics.max_drawdown`` -- "-0.02" is a 2% one-period loss that ``confidence``
    of periods did not exceed. Empirical, not parametric: it makes no normality
    assumption, which matters for a single stock's fat tails. The price of that
    is sample size -- see ``config.VAR_CONFIDENCE`` for why 95% and not 99%.
    """
    _check_confidence(confidence)
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    return float(np.quantile(r.to_numpy(), 1.0 - confidence))


def historical_cvar(returns: pd.Series, confidence: float = VAR_CONFIDENCE) -> float:
    """Expected shortfall: the mean of the returns at or below :func:`historical_var`.

    The tail is never empty -- the VaR quantile is never below the sample
    minimum, so at least the worst observation is always in it.
    """
    r = _clean(returns)
    var = historical_var(r, confidence)
    if np.isnan(var):
        return float("nan")
    return float(r[r <= var].mean())


def _period_risk_free(periods_per_year: int, annual_risk_free_rate: float) -> float:
    """De-compounded per-period rate -- the same conversion ``sharpe_ratio`` makes."""
    return (1.0 + annual_risk_free_rate) ** (1.0 / periods_per_year) - 1.0


def _period_downside_deviation(r: pd.Series, mar: float) -> float:
    """Root-mean-square of the shortfall below ``mar``, over ALL periods.

    The denominator is the full sample, not just the losing periods: a series
    that rarely falls short is *less* risky, and dividing by the count of
    losers instead would make it look worse.
    """
    shortfall = np.minimum(r.to_numpy() - mar, 0.0)
    return float(np.sqrt(np.mean(shortfall**2)))


def downside_deviation(
    returns: pd.Series,
    periods_per_year: int,
    annual_risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
) -> float:
    """Annualised downside deviation, with the risk-free rate as the threshold."""
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    mar = _period_risk_free(periods_per_year, annual_risk_free_rate)
    return _period_downside_deviation(r, mar) * float(np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    periods_per_year: int,
    annual_risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
) -> float:
    """Annualised Sortino: mean excess return over downside deviation.

    Same threshold as :func:`downside_deviation` and the same per-period
    risk-free rate ``sharpe_ratio`` subtracts, so at rf = 0 the two ratios
    differ only in which risk they divide by. NaN when there is no downside at
    all -- the ratio is infinite there, and reporting a huge finite number for
    it would be the same silent lie ``_ZERO_VARIANCE_EPS`` exists to prevent.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    mar = _period_risk_free(periods_per_year, annual_risk_free_rate)
    downside = _period_downside_deviation(r, mar)
    if not np.isfinite(downside) or downside < _ZERO_VARIANCE_EPS:
        return float("nan")
    return float((r.mean() - mar) / downside * np.sqrt(periods_per_year))


def calmar_ratio(returns: pd.Series, periods_per_year: int) -> float:
    """CAGR over the magnitude of the max drawdown; NaN if it never drew down."""
    worst = max_drawdown(returns)
    if np.isnan(worst) or worst > -_ZERO_VARIANCE_EPS:
        return float("nan")
    growth = cagr(returns, periods_per_year)
    if np.isnan(growth):
        return float("nan")
    return float(growth / abs(worst))


def drawdown_duration(returns: pd.Series) -> tuple[int, bool]:
    """(longest stretch below a prior peak in periods, whether it is still open).

    A period is "underwater" when its drawdown is below zero, measured the way
    ``metrics`` measures it -- from a running peak floored at the 1.0 starting
    value. The flag is True when the longest stretch runs to the last period
    of the series, i.e. it has not recovered by the end of the window and the
    number is a lower bound on how long it will really last. On a tie the
    earlier stretch wins.
    """
    dd = _drawdown_series(returns)
    if dd is None:
        return 0, False
    underwater = dd.to_numpy() < 0.0
    longest, run, open_at_end = 0, 0, False
    for i, below in enumerate(underwater):
        run = run + 1 if below else 0
        if run > longest:
            longest, open_at_end = run, i == len(underwater) - 1
    return longest, open_at_end


def risk_summary(
    returns: pd.Series,
    periods_per_year: int,
    name: str = "",
    annual_risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
    confidence: float = VAR_CONFIDENCE,
) -> dict:
    """One row of realized risk and return metrics for a periodic return series.

    ``metrics.summarize``'s row (return, vol, Sharpe, max drawdown and its
    dates) extended with the tail, downside and drawdown-duration measures.
    """
    r = _clean(returns)
    row = summarize(r, periods_per_year, name, annual_risk_free_rate)
    days, still_open = drawdown_duration(r)
    row.update(
        {
            "var": historical_var(r, confidence),
            "cvar": historical_cvar(r, confidence),
            "worst_day": float(r.min()) if len(r) else float("nan"),
            "downside_deviation": downside_deviation(r, periods_per_year, annual_risk_free_rate),
            "sortino_ratio": sortino_ratio(r, periods_per_year, annual_risk_free_rate),
            "calmar_ratio": calmar_ratio(r, periods_per_year),
            "longest_drawdown_days": days,
            "longest_drawdown_open": still_open,
            # pandas' bias-corrected sample skew / Fisher excess kurtosis; NaN
            # below 3 / 4 observations rather than a number from too little data.
            "skew": float(r.skew()),
            "excess_kurtosis": float(r.kurt()),
        }
    )
    return row


def realized_risk_table(
    book_returns: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    annual_risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
) -> pd.DataFrame:
    """One :func:`risk_summary` row per book, indexed by book name."""
    rows = [
        risk_summary(book_returns[book], periods_per_year, str(book), annual_risk_free_rate)
        for book in book_returns.columns
    ]
    return pd.DataFrame(rows).set_index("name").rename_axis(None)



# Rows of the realized before/after table: (label, key in the realized frame,
# kind). Risk measures first, return context last. "pct" values print as
# percentages, "ratio" as plain numbers, "days" as whole periods.
_REALIZED_ROWS: tuple[tuple[str, str, str], ...] = (
    ("annual volatility", "annual_volatility", "pct"),
    ("max drawdown", "max_drawdown", "pct"),
    ("longest drawdown (days)", "longest_drawdown_days", "days"),
    (f"VaR {VAR_CONFIDENCE:.0%} (daily)", "var", "pct"),
    (f"CVaR {VAR_CONFIDENCE:.0%} (daily)", "cvar", "pct"),
    ("worst day", "worst_day", "pct"),
    ("downside deviation", "downside_deviation", "pct"),
    ("Sortino", "sortino_ratio", "ratio"),
    ("Calmar", "calmar_ratio", "ratio"),
    ("skew", "skew", "ratio"),
    ("excess kurtosis", "excess_kurtosis", "ratio"),
    ("CAGR", "cagr", "pct"),
    ("total return", "total_return", "pct"),
    ("Sharpe", "sharpe_ratio", "ratio"),
)


def _format_metric(value: float, kind: str) -> str:
    """One table cell; NaN prints ``--`` so an undefined metric never reads as 0."""
    if pd.isna(value):
        return "--"
    if kind == "pct":
        return f"{value:.2%}"
    if kind == "days":
        return f"{int(value)}"
    return f"{value:.3f}"


def _format_delta(value: float, kind: str) -> str:
    if pd.isna(value):
        return "--"
    if kind == "pct":
        return f"{value * 100:+.2f}pp"
    if kind == "days":
        return f"{int(value):+d}"
    return f"{value:+.3f}"


def format_metric_table(realized: pd.DataFrame) -> str:
    """Metrics as rows, one column per series, from a :func:`realized_risk_table` frame.

    Shared by every report that prints these rows, so two reports cannot
    describe the same metric under different labels or in a different format.
    Undefined metrics print ``--`` and an unrecovered longest drawdown is
    flagged ``(open)``.
    """
    cells: dict[str, dict[str, str]] = {}
    for label, key, kind in _REALIZED_ROWS:
        cells[label] = {}
        for book in realized.index:
            text = _format_metric(realized.at[book, key], kind)
            if key == "longest_drawdown_days" and realized.at[book, "longest_drawdown_open"]:
                text += " (open)"
            cells[label][book] = text
    return pd.DataFrame(cells).T.to_string()
