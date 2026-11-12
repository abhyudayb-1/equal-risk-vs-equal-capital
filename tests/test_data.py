"""Self-checks for the data layer.

Runs standalone, no framework, no network:

    .\\.venv\\Scripts\\python.exe tests\\test_data.py

Everything here uses synthetic frames on purpose. The point is to prove the
*machinery* -- windowing, the rebalance calendar, the look-ahead guard -- rather
than to assert anything about what yfinance returned today. Live-data checks
belong in a live run (run_backtest.py), not in a suite that has to pass offline.

Three of these exist because the corresponding bug was actually shipped:

* the on-disk cache ignored ``start``/``end`` entirely, so a warm cache returned
  the full history no matter what window was asked for;
* the risk-free series was truncated *before* being cached, so one late-start
  request poisoned every later early-start one;
* the last observed date of an unfinished month was treated as a month-end,
  putting a rebalance on the 10th and opening a stub period.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, data  # noqa: E402
from src.data import (  # noqa: E402
    DataUnavailableError,
    _is_true_month_end,
    _restrict,
    _validate,
    align_risk_free,
    asof_slice,
    daily_returns,
    month_end_rebalance_dates,
)

_checks = 0


def check(label: str, condition: bool) -> None:
    global _checks
    if not condition:
        raise AssertionError(f"FAILED: {label}")
    _checks += 1
    print(f"  ok   {label}")


def _prices(start: str = "2006-02-03", end: str = "2026-08-10") -> pd.DataFrame:
    index = pd.bdate_range(start, end, name="date")
    rng = np.random.default_rng(0)
    steps = 1.0 + rng.normal(0.0003, 0.01, (len(index), len(config.UNIVERSE)))
    return pd.DataFrame(
        100.0 * steps.cumprod(axis=0), index=index, columns=list(config.UNIVERSE)
    )


def test_windowing() -> None:
    print("windowing (the cache bug):")
    frame = _prices()

    trimmed = _restrict(frame, "2012-01-01", "2015-01-01")
    check("start is honoured", trimmed.index[0] >= pd.Timestamp("2012-01-01"))
    check("end is honoured", trimmed.index[-1] <= pd.Timestamp("2015-01-01"))
    check("open-ended request keeps the tail",
          _restrict(frame, "2012-01-01", None).index[-1] == frame.index[-1])
    check("full-range request is a no-op",
          len(_restrict(frame, "2006-02-03", None)) == len(frame))

    try:
        _restrict(frame, "2030-01-01", None)
        raise AssertionError("FAILED: empty window did not raise")
    except DataUnavailableError:
        check("an empty window raises instead of returning nothing", True)

    series = frame[config.EQUITY_TICKER]
    check("works on a Series too (the risk-free path)",
          _restrict(series, "2020-01-01", None).index[0] >= pd.Timestamp("2020-01-01"))


def test_rebalance_calendar() -> None:
    print("rebalance calendar:")
    frame = _prices(end="2026-08-10")  # mid-month, month still in progress
    dates = month_end_rebalance_dates(frame.index, config.COV_WARMUP_TRADING_DAYS)

    check("every rebalance date is a real trading day", bool(dates.isin(frame.index).all()))
    check("dates are sorted and unique",
          dates.is_monotonic_increasing and dates.is_unique)
    check(
        f"the in-progress final month is dropped (last = {dates[-1].date()})",
        dates[-1] < pd.Timestamp("2026-08-01"),
    )
    check("warmup is respected",
          int((frame.index <= dates[0]).sum()) >= config.COV_WARMUP_TRADING_DAYS)
    check("one rebalance per month, no gaps",
          len(dates) == len(pd.PeriodIndex(dates, freq="M").unique()))

    # A dataset ending exactly on a completed month-end must KEEP that date --
    # the naive "is there data next month?" test would wrongly drop it.
    december = month_end_rebalance_dates(_prices(end="2024-12-31").index, 0)
    check("a genuine 31-December month-end is kept",
          december[-1] == pd.Timestamp("2024-12-31"))

    check("2026-08-10 is not a month end", not _is_true_month_end(pd.Timestamp("2026-08-10")))
    check("2024-12-31 is a month end", _is_true_month_end(pd.Timestamp("2024-12-31")))
    check("a Friday before a weekend month-end counts",
          _is_true_month_end(pd.Timestamp("2024-08-30")))

    # Regression: "no BUSINESS days remain" is not "no TRADING days remain". When
    # the month's last business day is an exchange holiday, data ending on the
    # real last trading day looked like an unfinished month and its rebalance
    # was dropped.
    for day, holiday in (
        ("2024-03-28", "Good Friday 2024-03-29"),
        ("2018-03-29", "Good Friday 2018-03-30"),
        ("2021-05-28", "Memorial Day 2021-05-31"),
    ):
        check(f"{day} is a month end: only {holiday} follows it",
              _is_true_month_end(pd.Timestamp(day)))
    holiday_end = month_end_rebalance_dates(_prices(end="2024-03-28").index, 0)
    check("data ending the Thursday before a month-end Good Friday keeps that rebalance",
          holiday_end[-1] == pd.Timestamp("2024-03-28"))
    # The holidays must not make an ordinary mid-month day pass, nor the day
    # before a holiday that still has trading days after it in the month.
    check("the Wednesday before that Good Friday is still not a month end",
          not _is_true_month_end(pd.Timestamp("2024-03-27")))
    check("the Friday before a Memorial Day that is NOT the 31st is not a month end",
          not _is_true_month_end(pd.Timestamp("2024-05-24")))
    check("a Thursday with an ordinary Friday left in the month is not a month end",
          not _is_true_month_end(pd.Timestamp("2025-05-29")))

    check("opting out keeps the stub",
          month_end_rebalance_dates(frame.index, 0, drop_partial_final_month=False)[-1]
          == frame.index[-1])

    try:
        month_end_rebalance_dates(pd.DatetimeIndex([]), 0)
        raise AssertionError("FAILED: empty index did not raise")
    except DataUnavailableError:
        check("an empty index raises", True)


def test_returns_and_guard() -> None:
    print("returns and the look-ahead guard:")
    frame = _prices()
    returns = daily_returns(frame)

    check("one fewer row than prices", len(returns) == len(frame) - 1)
    check("no NaN survives", not returns.isna().to_numpy().any())
    check("first row is dropped, not zero-filled", returns.index[0] == frame.index[1])

    reconstructed = float((1 + returns[config.EQUITY_TICKER]).prod())
    actual = float(frame[config.EQUITY_TICKER].iloc[-1] / frame[config.EQUITY_TICKER].iloc[0])
    check("returns compound back to the price ratio",
          abs(reconstructed - actual) < 1e-6 * actual)

    date = returns.index[1000]
    sliced = asof_slice(returns, date)
    check("asof_slice includes the boundary date", sliced.index[-1] == date)
    check("asof_slice admits nothing after it", sliced.index.max() <= date)
    check("asof_slice works on a Series",
          asof_slice(returns[config.EQUITY_TICKER], date).index[-1] == date)

    # The guard must hold at every rebalance, not just a sampled one.
    dates = month_end_rebalance_dates(returns.index, config.COV_WARMUP_TRADING_DAYS)
    check("no rebalance can see the future",
          all(asof_slice(returns, d).index.max() <= d for d in dates))


def test_validation_raises() -> None:
    print("validation is loud:")
    frame = _prices()

    empty = frame.iloc[:0]
    try:
        daily_returns(empty)
        raise AssertionError("FAILED: empty price frame did not raise")
    except DataUnavailableError:
        check("empty price frame raises", True)

    late = frame.copy()
    late[config.COMMODITY_TICKER] = late[config.COMMODITY_TICKER].where(
        late.index >= pd.Timestamp("2008-01-01")
    )
    try:
        _validate(late, "2006-02-03")
        raise AssertionError("FAILED: a sleeve starting two years late did not raise")
    except DataUnavailableError:
        check("a sleeve whose history starts years late raises", True)

    negative = frame.copy()
    negative.iloc[100, 0] = -1.0
    try:
        _validate(negative, "2006-02-03")
        raise AssertionError("FAILED: negative price did not raise")
    except DataUnavailableError:
        check("a non-positive price raises", True)

    gapped = frame.copy()
    gapped.iloc[500:520, 1] = np.nan  # 20 trading days, well past the 5-day tolerance
    try:
        _validate(gapped, "2006-02-03")
        raise AssertionError("FAILED: a 20-day hole did not raise")
    except DataUnavailableError:
        check("an interior gap beyond the tolerance raises", True)

    short = frame.copy()
    short.iloc[500:502, 1] = np.nan  # 2 days, inside tolerance
    check("a short holiday gap is filled rather than fatal",
          not _validate(short, "2006-02-03").isna().to_numpy().any())


def test_risk_free_alignment() -> None:
    print("risk-free alignment:")
    prices = _prices()
    # FRED publishes on business days and skips holidays; imitate that.
    published = pd.bdate_range("2006-02-03", "2026-08-07", name="date")
    published = published[::1].drop(published[[10, 11, 12]])
    risk_free = pd.Series(0.05 / 252, index=published, name="risk_free")

    aligned = align_risk_free(risk_free, prices.index)
    check("covers the whole price calendar", len(aligned) == len(prices))
    check("no NaN after alignment", not aligned.isna().any())
    check("index matches the prices exactly", aligned.index.equals(prices.index))

    # Forward-fill only: a rate is known from its print date onward. A back-fill
    # would leak a future rate into a past decision.
    stepped = pd.Series(
        [0.01 / 252, 0.05 / 252],
        index=pd.DatetimeIndex(["2010-01-04", "2010-06-01"]),
        name="risk_free",
    )
    target = pd.bdate_range("2010-01-04", "2010-12-31")
    filled = align_risk_free(stepped, target)
    check("the old rate persists until the new one prints",
          float(filled.loc["2010-05-28"]) == 0.01 / 252)
    check("the new rate applies from its print date",
          float(filled.loc["2010-06-01"]) == 0.05 / 252)


def test_risk_free_cache_is_per_series() -> None:
    # Regression: every FRED series shared one cache file, so asking for a
    # second series while the first was still fresh returned the first one's
    # data under the second one's name. Offline: the cache and the HTTP call
    # are both replaced, so nothing here touches the disk or the network.
    print("risk-free cache:")
    store: dict[str, pd.DataFrame] = {}
    requested: list[str] = []

    class _Response:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    levels = {config.RISK_FREE_SERIES_ID: "5.00", "OTHER": "1.00"}

    def fake_get(url, **kwargs):
        series = next(s for s in levels if url.endswith(f"id={s}"))
        requested.append(series)
        return _Response(
            f"observation_date,{series}\n2010-01-04,{levels[series]}\n2010-01-05,{levels[series]}\n"
        )

    real = (data.cache.is_fresh, data.cache.read_frame, data.cache.write_frame,
            data.requests.get)
    data.cache.is_fresh = lambda name, max_age_hours: name in store
    data.cache.read_frame = lambda name: store.get(name)
    data.cache.write_frame = lambda name, frame: store.__setitem__(name, frame.copy())
    data.requests.get = fake_get
    try:
        default = data.fetch_risk_free(start="2010-01-04")
        other = data.fetch_risk_free(series_id="OTHER", start="2010-01-04")
        default_again = data.fetch_risk_free(start="2010-01-04")
        other_again = data.fetch_risk_free(series_id="OTHER", start="2010-01-04")
    finally:
        (data.cache.is_fresh, data.cache.read_frame, data.cache.write_frame,
         data.requests.get) = real

    check("a second series is fetched, not answered from the first one's cache",
          requested == [config.RISK_FREE_SERIES_ID, "OTHER"])
    check("...so the two series carry their own rates",
          float(other.iloc[0]) < float(default.iloc[0]))
    check("each series is then served from its own cache entry",
          default_again.equals(default) and other_again.equals(other))
    check("the default series keeps its existing cache file",
          data.RISK_FREE_CACHE_FILE in store and len(store) == 2)


if __name__ == "__main__":
    test_risk_free_cache_is_per_series()
    test_windowing()
    test_rebalance_calendar()
    test_returns_and_guard()
    test_validation_raises()
    test_risk_free_alignment()
    print(f"\n{_checks} checks passed")
