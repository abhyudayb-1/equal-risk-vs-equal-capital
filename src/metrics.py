"""Performance and risk metrics for a periodic return series.

Ported from ``abhyudayb-1/sp500-factor-backtester`` with one deliberate change:
``periods_per_year`` is a **required argument** rather than a module constant.
That project rebalanced monthly and could hard-code 12; this one accrues
**daily** (252) but still reports some series monthly, so a module-level default
would silently annualise half the project wrong.

Two subtleties carried over from the original, both of which look like noise and
are not -- see ``_ZERO_VARIANCE_EPS`` and the peak floor in
:func:`_drawdown_series`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ANNUAL_RISK_FREE_RATE

# Floating-point variance of a constant series is ~1e-18, not exactly 0, so an
# `sd == 0` guard silently lets through a Sharpe of ~1e16. Periodic returns are
# O(1e-2), so anything below this really is zero variance.
_ZERO_VARIANCE_EPS = 1e-12


def _drawdown_series(returns: pd.Series) -> pd.Series | None:
    """Drawdown at each period, as a non-positive fraction.

    The running peak is floored at the 1.0 starting value. Without that floor,
    a portfolio whose FIRST period is negative reports a 0% drawdown -- the
    peak would be taken from the first period's *end* rather than from the
    initial capital, which understates every drawdown that begins at the start
    of the window. That matters more here than in the original project, because
    the crisis windows in ``config.CRISIS_WINDOWS`` are sliced to *start* at a
    market peak, so the first period is very often negative.
    """
    r = returns.dropna().astype("float64")
    if r.empty:
        return None
    curve = (1.0 + r).cumprod()
    running_peak = curve.cummax().clip(lower=1.0)
    return curve / running_peak - 1.0


def cumulative_curve(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    """Growth of ``start_value`` through a series of periodic returns."""
    r = returns.fillna(0.0).astype("float64")
    return start_value * (1.0 + r).cumprod()


def total_return(returns: pd.Series) -> float:
    r = returns.dropna().astype("float64")
    if r.empty:
        return float("nan")
    return float((1.0 + r).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int) -> float:
    r = returns.dropna().astype("float64")
    if r.empty:
        return float("nan")
    years = len(r) / periods_per_year
    if years <= 0:
        return float("nan")
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return float("nan")
    return growth ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int) -> float:
    r = returns.dropna().astype("float64")
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int,
    annual_risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
) -> float:
    """Annualized Sharpe ratio of periodic excess returns.

    The annual risk-free rate is de-compounded to a per-period rate and
    subtracted before the ratio is taken. Default rf is 0.0 (configurable in
    ``config.ANNUAL_RISK_FREE_RATE``), so by default this is a
    return-to-volatility ratio.
    """
    r = returns.dropna().astype("float64")
    if len(r) < 2:
        return float("nan")
    rf_period = (1.0 + annual_risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_period
    sd = excess.std(ddof=1)
    if not np.isfinite(sd) or sd < _ZERO_VARIANCE_EPS:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline of the cumulative curve, as a negative float."""
    dd = _drawdown_series(returns)
    if dd is None:
        return float("nan")
    return float(dd.min())


def max_drawdown_dates(
    returns: pd.Series,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """(peak date, trough date) of the maximum drawdown.

    A peak of ``None`` means the drawdown ran from the initial capital, i.e.
    the strategy never traded above its starting value before this trough.
    """
    dd = _drawdown_series(returns)
    if dd is None:
        return None, None
    r = returns.dropna().astype("float64")
    curve = (1.0 + r).cumprod()
    trough = dd.idxmin()
    before = curve.loc[:trough]
    if before.empty or before.max() < 1.0:
        return None, trough
    return before.idxmax(), trough


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with a positive return."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    return float((r > 0).mean())


def summarize(
    returns: pd.Series,
    periods_per_year: int,
    name: str = "",
    annual_risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
) -> dict:
    """One-row metric summary for a periodic return series."""
    peak, trough = max_drawdown_dates(returns)
    return {
        "name": name,
        "n_periods": int(returns.dropna().shape[0]),
        "total_return": total_return(returns),
        "cagr": cagr(returns, periods_per_year),
        "annual_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year, annual_risk_free_rate),
        "max_drawdown": max_drawdown(returns),
        "max_drawdown_peak": peak,
        "max_drawdown_trough": trough,
        "hit_rate": hit_rate(returns),
    }
