"""Self-checks for the data layer (TASKS.md steps 8-10).

Runs standalone, no framework, no network:

    .\\.venv\\Scripts\\python.exe tests\\test_data.py

Everything here uses synthetic frames on purpose. The point is to prove the
*machinery* -- windowing, the rebalance calendar, the look-ahead guard -- rather
than to assert anything about what yfinance returned today. Live-data checks
belong in the Phase 5 run, not in a suite that has to pass offline.

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
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import cache, config  # noqa: E402
from src import data as data_module  # noqa: E402
from src.data import (  # noqa: E402
    DataUnavailableError,
    _is_true_month_end,
    _restrict,
    _ticker_cache_name,
    _validate,
    align_risk_free,
    asof_slice,
    daily_returns,
    fetch_ticker_prices,
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


def _raw_download(ticker: str, start: str, end: str, multiindex: bool) -> pd.DataFrame:
    """What ``yf.download`` returns for one ticker, in either shape yfinance uses."""
    index = pd.bdate_range(start, end)
    rng = np.random.default_rng(1)
    close = 50.0 * (1.0 + rng.normal(0.0004, 0.012, len(index))).cumprod()
    frame = pd.DataFrame({"Open": close, "Close": close, "Volume": 1_000.0}, index=index)
    if multiindex:
        frame.columns = pd.MultiIndex.from_product([frame.columns, [ticker]])
    return frame


def test_fetch_ticker_prices() -> None:
    """The head-to-head's own fetch (step 44), against a monkeypatched yfinance.

    Two properties are why this function exists at all, and both are attacked
    here rather than assumed: it must accept a ticker that listed long after
    ``DATA_START_DATE`` (which ``fetch_prices``' late-start guard rejects), and
    it must never touch the shared ``prices_close.csv`` cache (which
    ``cache.write_frame`` would otherwise replace wholesale).
    """
    print("single-ticker fetch (step 44):")
    real_download = data_module.yf.download
    real_cache_dir = cache.CACHE_DIR
    calls: list[str] = []

    def make_download(raw: pd.DataFrame | None):
        def fake(tickers, **kwargs):
            calls.append(str(tickers))
            return raw

        return fake

    try:
        with tempfile.TemporaryDirectory() as directory:
            cache.CACHE_DIR = Path(directory)
            shared = cache.CACHE_DIR / data_module.PRICE_CACHE_FILE
            shared.write_bytes(b"date,SPY" + bytes([10]) + b"2020-01-01,1.0" + bytes([10]))  # a stand-in for the engine's cache
            shared_before = shared.read_bytes()

            # A ticker that listed in 2019: ~13 years after DATA_START_DATE.
            late_flat = _raw_download("NEWCO", "2019-06-03", "2024-06-28", multiindex=False)
            data_module.yf.download = make_download(late_flat)
            series = fetch_ticker_prices("newco")
            check("a ticker listed 13 years after DATA_START_DATE is accepted",
                  len(series) == len(late_flat) and series.index[0] == late_flat.index[0])
            check("...and comes back as a named, positive Series",
                  series.name == "NEWCO" and bool((series > 0).all()))

            as_frame = late_flat[["Close"]].rename(columns={"Close": "NEWCO"})
            as_frame.index.name = "date"
            try:
                _validate(as_frame, config.DATA_START_DATE)
                raise AssertionError("FAILED: fetch_prices' guard accepted a late-listed ticker")
            except DataUnavailableError:
                check("positive control: the same series is rejected by fetch_prices' "
                      "late-start guard", True)

            check("the shared prices_close.csv is untouched",
                  shared.read_bytes() == shared_before)
            check("the ticker is cached in its own file",
                  (cache.CACHE_DIR / _ticker_cache_name("NEWCO")).exists())

            # Second call is served from that file: yfinance must not be reached.
            calls.clear()
            data_module.yf.download = make_download(None)
            again = fetch_ticker_prices("NEWCO")
            check("a fresh cache hit does not download again", calls == [])
            check("...and returns the same values",
                  bool(np.allclose(again.to_numpy(), series.to_numpy())))
            data_module.yf.download = make_download(late_flat)
            fetch_ticker_prices("NEWCO", refresh=True)
            check("refresh=True bypasses the cache and downloads", len(calls) == 1)

            # yfinance's other output shape.
            data_module.yf.download = make_download(
                _raw_download("MULTI", "2015-01-05", "2024-06-28", multiindex=True)
            )
            multi = fetch_ticker_prices("MULTI")
            check("a MultiIndex (Close, ticker) download is flattened the same way",
                  multi.name == "MULTI" and len(multi) > 2000)

            check("the shared cache is still untouched after every call above",
                  shared.read_bytes() == shared_before)

            data_module.yf.download = make_download(pd.DataFrame())
            try:
                fetch_ticker_prices("GHOST", refresh=True)
                raise AssertionError("FAILED: an empty download did not raise")
            except DataUnavailableError:
                check("an empty download raises DataUnavailableError", True)

            gapped = _raw_download("HOLE", "2015-01-05", "2024-06-28", multiindex=False)
            gapped.iloc[500:520, gapped.columns.get_loc("Close")] = np.nan
            data_module.yf.download = make_download(gapped)
            try:
                fetch_ticker_prices("HOLE", refresh=True)
                raise AssertionError("FAILED: a 20-day hole did not raise")
            except DataUnavailableError:
                check("an interior gap beyond the tolerance still raises", True)

            try:
                fetch_ticker_prices("   ")
                raise AssertionError("FAILED: a blank ticker did not raise")
            except ValueError:
                check("a blank ticker raises ValueError", True)

            check("cache filenames are filesystem-safe for indices and share classes",
                  _ticker_cache_name("^GSPC") == "ticker__GSPC.csv"
                  and _ticker_cache_name("BRK-B") == "ticker_BRK-B.csv")
    finally:
        data_module.yf.download = real_download
        cache.CACHE_DIR = real_cache_dir


if __name__ == "__main__":
    test_windowing()
    test_rebalance_calendar()
    test_returns_and_guard()
    test_validation_raises()
    test_risk_free_alignment()
    test_fetch_ticker_prices()
    print(f"\n{_checks} checks passed")
