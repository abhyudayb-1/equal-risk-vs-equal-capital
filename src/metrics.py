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

import logging

import numpy as np
import pandas as pd

from .config import ANNUAL_RISK_FREE_RATE, CRISIS_WINDOWS, TRADING_DAYS_PER_YEAR
from .types import BacktestResult

log = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Crisis windows
# ---------------------------------------------------------------------------
# The three windows in ``config.CRISIS_WINDOWS`` are peak-to-trough, so a sliced
# series very often opens with a negative period. That is exactly the case the
# peak floor in :func:`_drawdown_series` was kept for -- without it, slicing to a
# crisis and asking for its drawdown would report the decline from the window's
# own first close rather than from the pre-crisis peak, understating every
# crisis in the table. The port's subtlety and this section's correctness are
# the same fact.

Window = str | tuple[str | pd.Timestamp | None, str | pd.Timestamp | None]


def resolve_window(window: Window) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    """Turn a window name or an explicit pair into concrete timestamps.

    ``window`` is either a key of ``config.CRISIS_WINDOWS`` or a ``(start, end)``
    pair. An ``end`` of ``None`` means "through the end of the sample", which is
    how ``config.REGIME_SHIFT_WINDOW`` expresses an episode that has not closed.

    An unknown name raises rather than falling through to something empty: a
    mistyped window that silently produced zero rows would put a plausible,
    wrong line in the crisis table.
    """
    if isinstance(window, str):
        if window not in CRISIS_WINDOWS:
            raise KeyError(
                f"unknown window {window!r}; config.CRISIS_WINDOWS has "
                f"{list(CRISIS_WINDOWS)}. Pass an explicit (start, end) pair for "
                "anything else."
            )
        window = CRISIS_WINDOWS[window]

    start_raw, end_raw = window
    if start_raw is None:
        raise ValueError("a window must have a start date")

    start = pd.Timestamp(start_raw)
    end = None if end_raw is None else pd.Timestamp(end_raw)
    if end is not None and end < start:
        raise ValueError(f"window ends {end.date()} before it starts {start.date()}")
    return start, end


def slice_dated(data: pd.Series | pd.DataFrame, window: Window):
    """Restrict a dated series or frame to one window, both endpoints inclusive.

    The building block under :func:`slice_to_window`, and the entry point for
    anything that is not a ``BacktestResult`` -- a rolling correlation series, a
    price frame, the diversification ratio in step 25.

    Raises if the window and the data do not overlap at all. A crisis row built
    from an empty slice is every metric as NaN, which reads as "the strategy was
    flat" rather than "this was never measured" -- so it fails here instead.

    Partial coverage is allowed but logged: the GFC window opens 2007-10-09 and
    ``config.DATA_START_DATE`` is 2006-02-03, so the real run covers it fully,
    but a shortened test series may not, and a half-covered crisis is worth
    knowing about before it is quoted.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("slice_to_window needs a DatetimeIndex")
    if not data.index.is_monotonic_increasing:
        data = data.sort_index()

    start, end = resolve_window(window)
    sliced = data.loc[start:end] if end is not None else data.loc[start:]

    if len(sliced) == 0:
        available = (
            f"{data.index[0].date()} to {data.index[-1].date()}" if len(data) else "empty"
        )
        raise ValueError(
            f"window {start.date()} to {'open' if end is None else end.date()} does not "
            f"overlap the data ({available})"
        )

    if sliced.index[0] > start:
        log.warning(
            "Window opens %s but the data starts %s; the first %d days of the window "
            "are missing.", start.date(), sliced.index[0].date(),
            (sliced.index[0] - start).days,
        )
    if end is not None and sliced.index[-1] < end:
        log.warning(
            "Window closes %s but the data ends %s; the last %d days of the window "
            "are missing.", end.date(), sliced.index[-1].date(),
            (end - sliced.index[-1]).days,
        )
    return sliced


def _as_window(start: Window | str | pd.Timestamp, end) -> Window:
    """Accept either ``("GFC (2008)")`` or ``(start_date, end_date)``.

    A bare string is read as a window *name* when ``config.CRISIS_WINDOWS`` has
    it and no ``end`` was given; otherwise it is parsed as a date. The two never
    collide in practice -- no window is named like a date -- and the name form is
    what keeps call sites from re-typing the dates that config already owns.
    """
    if end is None and isinstance(start, str) and start in CRISIS_WINDOWS:
        return start
    if end is None and isinstance(start, tuple):
        return start
    return (start, end)


def slice_to_window(
    result: BacktestResult,
    start: Window | str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
) -> BacktestResult:
    """Cut a ``BacktestResult`` down to one date range, contract intact.

    ``start`` is a ``config.CRISIS_WINDOWS`` key, an explicit date paired with
    ``end``, or a ``(start, end)`` tuple. An ``end`` of ``None`` runs to the end
    of the sample.

    The part that is not a plain slice
    ---------------------------------
    ``weights``, ``risk_contributions`` and ``leverage`` are indexed by *monthly
    rebalance date*; ``returns`` is *daily*. Slicing all four to the same window
    breaks the result, because the contract requires the first return not to
    precede the first allocation -- and a crisis that opens on 2007-10-09 has its
    first in-window rebalance at the end of that month, two weeks *after* the
    returns start.

    So the rebalance frames are anchored to the **last rebalance at or before the
    window opens**, not to the window itself. That allocation is the one actually
    in force on the window's first day, which is both what the contract needs and
    what any honest attribution of the window's return has to attribute it to.
    When the window opens before the very first rebalance there is no such
    allocation, and the returns are trimmed forward to where one exists instead.
    """
    window = _as_window(start, end)
    window_start, window_end = resolve_window(window)

    rebalances = result.weights.index
    at_or_before = rebalances[rebalances <= window_start]
    anchor = at_or_before[-1] if len(at_or_before) else None

    keep = rebalances >= (anchor if anchor is not None else window_start)
    if window_end is not None:
        keep &= rebalances <= window_end
    kept = rebalances[keep]

    if len(kept) == 0:
        raise ValueError(
            f"no rebalance falls in {window_start.date()} to "
            f"{'open' if window_end is None else window_end.date()}; the result "
            f"rebalances {rebalances[0].date()} to {rebalances[-1].date()}"
        )

    # With no prior allocation the window has to start where one begins, or the
    # returns would describe days no weights in this result explain.
    returns_from = window_start if anchor is not None else kept[0]
    returns = result.returns.loc[returns_from:window_end]
    if len(returns) == 0:
        raise ValueError(
            f"no daily returns in {returns_from.date()} to "
            f"{'open' if window_end is None else window_end.date()}"
        )

    sliced = BacktestResult(
        weights=result.weights.loc[kept],
        risk_contributions=result.risk_contributions.loc[kept],
        leverage=result.leverage.loc[kept],
        returns=returns,
        meta={
            **result.meta,
            "window": window if isinstance(window, str) else None,
            "window_start": window_start,
            "window_end": window_end,
            "anchor_rebalance": anchor,
            "sliced_from": result.strategy,
            # Decided here, where the *unsliced* series is still available. After
            # the cut there is no way to tell a window the data covered fully
            # from one it merely overlapped.
            "fully_covered": bool(
                result.returns.index[0] <= window_start
                and (window_end is None or result.returns.index[-1] >= window_end)
            ),
        },
    )
    sliced.validate()
    return sliced


def summarize_window(
    result: BacktestResult,
    window: Window | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    name: str | None = None,
    annual_risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
) -> dict:
    """One metric row for a ``BacktestResult``, optionally sliced to a window.

    Pass ``window`` to slice and summarise in one call, or omit it to summarise a
    result already cut by :func:`slice_to_window`.

    ``periods_per_year`` defaults to daily rather than being required as it is
    elsewhere in this module: the contract pins ``BacktestResult.returns`` to
    daily accrual, so the caller cannot get it wrong here. It stays overridable
    for a monthly-resampled series.

    The row is :func:`summarize`'s, plus the window's identity and the dates
    actually covered. Those are not decoration -- a crisis table that reports a
    drawdown without saying which days produced it cannot be checked, and
    partial coverage is invisible otherwise.

    ``cagr`` is reported but should be read with care over a short window:
    annualising the 24-day COVID window extrapolates a crash to a full year. Lead
    a crisis table with ``total_return`` and ``max_drawdown``.
    """
    if window is not None:
        result = slice_to_window(result, window)

    returns = result.returns
    declared_start = result.meta.get("window_start")
    declared_end = result.meta.get("window_end")

    if name is None:
        name = result.meta.get("window") or result.strategy

    row = summarize(returns, periods_per_year, name=name,
                    annual_risk_free_rate=annual_risk_free_rate)
    row.update(
        strategy=result.meta.get("sliced_from", result.strategy),
        window_start=declared_start,
        window_end=declared_end,
        covered_start=returns.index[0],
        covered_end=returns.index[-1],
        # True for an unsliced result: it covers its own span by definition.
        fully_covered=bool(result.meta.get("fully_covered", True)),
    )
    return row
