"""Market data: ETF prices from yfinance, cached on disk.

Keyless by design -- yfinance and FRED are both free and need no API key, which
keeps this project runnable by anyone who clones it.

The one thing to understand before changing anything here: prices are pulled
with ``auto_adjust=True``, so the close series is **total return** -- dividends
reinvested and splits applied. That is not cosmetic for this universe. TLT and
TIP distribute most of their return as coupons, so a price-only series would
understate bond returns badly and hand the risk-parity portfolio a systematically
wrong picture of the sleeve it leans on hardest.
"""

from __future__ import annotations

import io
import logging
import re

import pandas as pd
import requests
import yfinance as yf

from . import cache
from .config import (
    DATA_START_DATE,
    FRED_CSV_URL,
    HTTP_TIMEOUT_SECONDS,
    HTTP_USER_AGENT,
    PRICE_CACHE_MAX_AGE_HOURS,
    RISK_FREE_CACHE_MAX_AGE_HOURS,
    RISK_FREE_SERIES_ID,
    TRADING_DAYS_PER_YEAR,
    UNIVERSE,
)

log = logging.getLogger(__name__)

PRICE_CACHE_FILE = "prices_close.csv"
RISK_FREE_CACHE_FILE = "risk_free.csv"

# FRED publishes DTB3 on business days only. Gaps this long are holidays and are
# forward-filled; anything longer means the series genuinely stopped and the
# financing cost would be silently stale.
MAX_RISK_FREE_GAP_DAYS = 10

# A sleeve whose history starts more than this long after DATA_START_DATE is a
# configuration error, not a data hiccup -- see DBC and the 2006 start date.
MAX_LATE_START_DAYS = 10

# Interior gaps up to this many trading days are forward-filled (exchange
# holidays that yfinance reports inconsistently across tickers). Anything longer
# is a real hole and raises.
MAX_INTERIOR_GAP_DAYS = 5


class DataUnavailableError(RuntimeError):
    """A required series could not be obtained, or came back unusable.

    Deliberately fatal. Silently dropping a sleeve would leave a "risk parity"
    portfolio quietly running on three asset classes instead of four, which
    still produces a plausible equity curve and an entirely wrong conclusion.
    """


def _flatten_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull the adjusted close matrix out of whatever shape yfinance returned.

    yfinance gives MultiIndex ``(field, ticker)`` columns for a multi-ticker
    download and flat columns for a single ticker.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise DataUnavailableError(f"yfinance returned no Close field: {raw.columns}")
        close = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise DataUnavailableError(f"yfinance returned no Close column: {list(raw.columns)}")
        close = raw[["Close"]].copy()
        close.columns = tickers[:1]

    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise DataUnavailableError(f"yfinance returned no data for {missing}")
    return close[tickers]


def _normalise_index(df: pd.DataFrame) -> pd.DataFrame:
    index = pd.to_datetime(df.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    out = df.copy()
    out.index = pd.DatetimeIndex(index).normalize()
    out.index.name = "date"
    return out.sort_index()


def _validate(close: pd.DataFrame, start: str) -> pd.DataFrame:
    """Raise on anything that would silently corrupt a backtest."""
    start_ts = pd.Timestamp(start)

    for ticker in close.columns:
        series = close[ticker].dropna()
        if series.empty:
            raise DataUnavailableError(f"{ticker} returned no usable prices at all")
        late = (series.index[0] - start_ts).days
        if late > MAX_LATE_START_DAYS:
            raise DataUnavailableError(
                f"{ticker} history starts {series.index[0].date()}, {late} days after "
                f"the configured start {start_ts.date()}. Either the ETF did not exist "
                f"yet (check config.DATA_START_DATE) or the pull is incomplete."
            )
        if (series <= 0).any():
            raise DataUnavailableError(f"{ticker} contains non-positive prices")

    # Interior gaps: short ones are holiday noise, long ones are real holes.
    filled = close.ffill(limit=MAX_INTERIOR_GAP_DAYS)
    still_missing = filled.isna() & close.isna()
    # Leading NaN before a ticker's own first observation is not a gap.
    for ticker in close.columns:
        first = close[ticker].first_valid_index()
        still_missing.loc[still_missing.index < first, ticker] = False
    if still_missing.to_numpy().any():
        worst = still_missing.sum()
        offenders = {t: int(n) for t, n in worst.items() if n}
        raise DataUnavailableError(
            f"price gaps longer than {MAX_INTERIOR_GAP_DAYS} trading days: {offenders}"
        )

    return filled.dropna(how="any")


def _cached_prices(tickers: list[str], start: str) -> pd.DataFrame | None:
    """Return the cached matrix if it is fresh and actually covers the request."""
    if not cache.is_fresh(PRICE_CACHE_FILE, PRICE_CACHE_MAX_AGE_HOURS):
        return None
    frame = cache.read_frame(PRICE_CACHE_FILE)
    if frame is None or frame.empty:
        return None
    if any(t not in frame.columns for t in tickers):
        log.info("Cached prices are missing requested tickers; re-downloading.")
        return None
    if frame.index[0] > pd.Timestamp(start) + pd.Timedelta(days=MAX_LATE_START_DAYS):
        log.info("Cached prices start too late for the request; re-downloading.")
        return None
    return frame[tickers]


def _restrict(frame, start: str, end: str | None):
    """Trim to the requested window.

    Applied to cache hits and fresh downloads alike. Without it the cache
    silently returns whatever span happens to be on disk: ``fetch_prices(
    end="2015-01-01")`` handed back data through today whenever the cache was
    warm, which would have made step 30's ``--start`` and date flags do nothing
    at all on the second run of any session.
    """
    out = frame.loc[frame.index >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end)]
    if len(out) == 0:
        raise DataUnavailableError(
            f"no observations between {start} and {end or 'today'}"
        )
    return out


def fetch_prices(
    tickers: tuple[str, ...] | list[str] = UNIVERSE,
    start: str = DATA_START_DATE,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Daily total-return close prices, one column per ticker.

    Parameters
    ----------
    tickers:
        Defaults to ``config.UNIVERSE``. Passing extras (e.g. adding GLD) is
        fine; the cache re-downloads when it does not cover the request.
    refresh:
        Bypass the on-disk cache and re-download.

    Raises
    ------
    DataUnavailableError
        If any ticker is missing, empty, starts materially late, or has an
        interior gap longer than ``MAX_INTERIOR_GAP_DAYS``.
    """
    tickers = list(tickers)

    if not refresh:
        cached = _cached_prices(tickers, start)
        if cached is not None:
            log.info("Using cached prices (%d rows).", len(cached))
            return _restrict(cached, start, end)

    log.info("Downloading %s from yfinance...", ", ".join(tickers))
    # Deliberately downloaded and cached through TODAY regardless of ``end``, so
    # one cached pull serves every window a session asks for. ``end`` narrows the
    # returned frame, not the stored one.
    raw = yf.download(
        tickers,
        start=start,
        end=None,
        auto_adjust=True,  # total return: dividends reinvested, splits applied
        progress=False,
        group_by="column",
    )
    if raw is None or raw.empty:
        raise DataUnavailableError(
            f"yfinance returned nothing for {tickers} from {start}. "
            "Check the tickers and network access."
        )

    close = _validate(_normalise_index(_flatten_close(raw, tickers)), start)
    cache.write_frame(PRICE_CACHE_FILE, close)
    log.info("Downloaded %d rows, %s to %s.", len(close), close.index[0].date(),
             close.index[-1].date())
    return _restrict(close, start, end)


def _ticker_cache_name(ticker: str) -> str:
    """A per-ticker cache filename, safe on every filesystem (``^GSPC``, ``BRK-B``)."""
    return f"ticker_{re.sub(r'[^A-Za-z0-9._-]', '_', ticker)}.csv"


def fetch_ticker_prices(ticker: str, refresh: bool = False) -> pd.Series:
    """Total-return daily closes for ONE ticker, over whatever history it has.

    Exists because :func:`fetch_prices` cannot serve a head-to-head comparison
    against an arbitrary ticker, for two independent reasons. Its late-start
    guard (``_validate``) rejects any ticker that listed more than
    ``MAX_LATE_START_DAYS`` after ``DATA_START_DATE`` -- correct for the four
    sleeves, where a late start is a configuration error, and wrong for a stock
    that IPO'd in 2019. And it writes the shared price cache as a single frame
    (``cache.write_frame`` replaces the whole file), so passing a candidate
    alongside the universe rewrites the cached SPY/TLT/DBC/TIP history too.

    So this fetches the ticker alone, from ``DATA_START_DATE``, into **its own**
    cache file; the shared ``prices_close.csv`` is never read or written here.
    Validation is against the ticker's *own* first date, so what is still
    caught is what is genuinely wrong with a series: an empty pull, a
    non-positive price, an interior gap longer than ``MAX_INTERIOR_GAP_DAYS``.
    "It has a short history" is deliberately not one of them -- how much
    overlap is enough is the caller's question (see ``src/compare.py``).

    Raises
    ------
    DataUnavailableError
        If yfinance returns nothing for the ticker, or the series is unusable.
    ValueError
        If ``ticker`` is blank.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("ticker cannot be empty")
    cache_name = _ticker_cache_name(ticker)

    if not refresh and cache.is_fresh(cache_name, PRICE_CACHE_MAX_AGE_HOURS):
        cached = cache.read_frame(cache_name)
        if cached is not None and not cached.empty and ticker in cached.columns:
            log.info("Using cached prices for %s (%d rows).", ticker, len(cached))
            return cached[ticker].dropna()

    log.info("Downloading %s from yfinance...", ticker)
    raw = yf.download(
        ticker,
        start=DATA_START_DATE,
        end=None,
        auto_adjust=True,  # total return: dividends reinvested, splits applied
        progress=False,
        group_by="column",
    )
    if raw is None or raw.empty:
        raise DataUnavailableError(
            f"yfinance returned nothing for {ticker!r} from {DATA_START_DATE}. "
            "Check the ticker symbol and network access."
        )

    close = _normalise_index(_flatten_close(raw, [ticker]))
    first = close[ticker].first_valid_index()
    if first is None:
        raise DataUnavailableError(f"{ticker} returned no usable prices at all")
    close = _validate(close, str(first.date()))
    cache.write_frame(cache_name, close)
    log.info("Downloaded %d rows for %s, %s to %s.", len(close), ticker,
             close.index[0].date(), close.index[-1].date())
    return close[ticker]


# ---------------------------------------------------------------------------
# Risk-free rate -- the financing cost of leverage
# ---------------------------------------------------------------------------
def _parse_fred_csv(text: str, series_id: str) -> pd.Series:
    """Parse a FRED CSV export into a dated float series.

    FRED changed its export header from ``DATE`` to ``observation_date`` and
    both are still in circulation, so the date column is taken positionally
    rather than by name. Missing observations are exported as ``"."``.
    """
    frame = pd.read_csv(io.StringIO(text))
    if frame.shape[1] < 2:
        raise DataUnavailableError(f"unexpected FRED CSV shape {frame.shape} for {series_id}")

    date_col = frame.columns[0]
    value_col = series_id if series_id in frame.columns else frame.columns[1]

    frame[date_col] = pd.to_datetime(frame[date_col])
    values = pd.to_numeric(frame[value_col], errors="coerce")  # "." -> NaN
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(frame[date_col]))
    series.index.name = "date"
    return series.sort_index().dropna()


def fetch_risk_free(
    series_id: str = RISK_FREE_SERIES_ID,
    start: str = DATA_START_DATE,
    refresh: bool = False,
) -> pd.Series:
    """Daily risk-free rate as a **per-trading-day decimal**, not annualized %.

    FRED publishes DTB3 as an annualized percentage (``3.72`` means 3.72%/yr).
    Two conversions happen here and both matter:

    1. percent -> decimal (``/100``);
    2. annual -> per-period by **de-compounding**,
       ``(1 + r_annual) ** (1 / 252) - 1``, matching how ``metrics.sharpe_ratio``
       de-compounds its own risk-free input. Dividing by 252 instead would be
       wrong in the same direction everywhere, which is exactly the kind of
       error a backtest never surfaces.

    This series is the financing cost in ``overlay.py``: leverage is borrowed,
    not free, and omitting this leg is the most common way a risk-parity
    backtest gets oversold.
    """
    if not refresh and cache.is_fresh(RISK_FREE_CACHE_FILE, RISK_FREE_CACHE_MAX_AGE_HOURS):
        cached = cache.read_frame(RISK_FREE_CACHE_FILE)
        if cached is not None and not cached.empty:
            log.info("Using cached risk-free series (%d rows).", len(cached))
            return _restrict(cached.iloc[:, 0].rename("risk_free"), start, None)

    url = FRED_CSV_URL.format(series_id=series_id)
    log.info("Downloading %s from FRED...", series_id)
    try:
        response = requests.get(
            url,
            timeout=HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": HTTP_USER_AGENT},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DataUnavailableError(f"could not fetch {series_id} from FRED: {exc}") from exc

    # Parsed and cached in FULL, then narrowed on return. Truncating before the
    # write meant a request with a late start poisoned the cache for every later
    # request with an earlier one.
    annual_percent = _parse_fred_csv(response.text, series_id)
    if annual_percent.loc[annual_percent.index >= pd.Timestamp(start)].empty:
        raise DataUnavailableError(f"{series_id} has no observations at or after {start}")

    gaps = annual_percent.index.to_series().diff().dt.days.dropna()
    if (gaps > MAX_RISK_FREE_GAP_DAYS).any():
        worst = gaps.max()
        log.warning(
            "%s has a %d-day gap; the financing cost will be forward-filled across it.",
            series_id, int(worst),
        )

    annual = annual_percent / 100.0
    if (annual < -0.01).any() or (annual > 0.25).any():
        raise DataUnavailableError(
            f"{series_id} contains implausible rates "
            f"({annual.min():.3f} to {annual.max():.3f}) -- check the units"
        )

    daily = (1.0 + annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    daily = daily.rename("risk_free")
    cache.write_frame(RISK_FREE_CACHE_FILE, daily.to_frame())
    log.info(
        "Risk-free: %d obs, %s to %s, %.2f%%-%.2f%% annualized.",
        len(daily), daily.index[0].date(), daily.index[-1].date(),
        annual.min() * 100, annual.max() * 100,
    )
    return _restrict(daily, start, None)


def align_risk_free(risk_free: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Put the risk-free series on the price calendar.

    Forward-fill only, and only across holidays: a rate is known from its
    publication date onward, so carrying the last print forward is correct while
    back-filling would leak a future rate into a past decision. Any leading days
    before the first FRED print are filled with the first available rate, which
    is the only non-look-ahead choice available.
    """
    aligned = risk_free.reindex(index.union(risk_free.index)).ffill().reindex(index)
    if aligned.isna().any():
        first_valid = aligned.first_valid_index()
        if first_valid is None:
            raise DataUnavailableError("risk-free series does not overlap the price calendar")
        aligned = aligned.bfill()
        log.warning(
            "Risk-free series starts %s, after the price calendar; "
            "back-filled the leading days with the first available rate.",
            risk_free.index[0].date(),
        )
    return aligned.rename("risk_free")


# ---------------------------------------------------------------------------
# Returns and the rebalance calendar
# ---------------------------------------------------------------------------
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns from a total-return close matrix.

    Simple rather than log returns because portfolio returns aggregate linearly
    across assets: ``r_p = w @ r``. Log returns do not, and mixing the two is a
    quiet source of error in exactly this kind of pipeline.

    The first row is dropped rather than filled with zero -- a zero return on
    day one is a fabricated observation, and it would drag the volatility
    estimate down at precisely the point the covariance warmup is shortest.
    """
    if prices.empty:
        raise DataUnavailableError("cannot compute returns from an empty price frame")
    returns = prices.sort_index().pct_change().iloc[1:]
    if returns.isna().to_numpy().any():
        offenders = {c: int(n) for c, n in returns.isna().sum().items() if n}
        raise DataUnavailableError(f"NaN returns after pct_change: {offenders}")
    return returns


def _is_true_month_end(date: pd.Timestamp) -> bool:
    """True if no business days remain in ``date``'s calendar month."""
    month_end = date + pd.offsets.MonthEnd(0)
    if month_end == date:
        return True
    remaining = pd.bdate_range(date + pd.Timedelta(days=1), month_end)
    return len(remaining) == 0


def month_end_rebalance_dates(
    index: pd.DatetimeIndex,
    warmup_trading_days: int = 0,
    drop_partial_final_month: bool = True,
) -> pd.DatetimeIndex:
    """Last **actual trading day** of each month, after a warmup.

    Taken from the observed index rather than generated with a calendar offset:
    ``pd.date_range(freq="BME")`` yields business-month ends that can fall on an
    exchange holiday, and a rebalance date that is not a real trading day either
    silently shifts or drops when it is used to look up a price.

    ``warmup_trading_days`` drops the leading dates that do not yet have enough
    history behind them for a covariance estimate. The first returned date is
    therefore the first month-end with at least that many observations before
    it, inclusive.
    """
    if len(index) == 0:
        raise DataUnavailableError("cannot build a rebalance calendar from an empty index")
    index = pd.DatetimeIndex(index).sort_values()

    if warmup_trading_days > len(index):
        raise DataUnavailableError(
            f"warmup of {warmup_trading_days} trading days exceeds the {len(index)} "
            "available observations"
        )

    as_series = pd.Series(index, index=index)
    month_ends = pd.DatetimeIndex(
        as_series.groupby([index.year, index.month]).last().to_numpy()
    ).sort_values()

    # The final group is the last *observed* date, which is only a month-end if
    # the month actually finished. Running a backtest mid-month otherwise puts a
    # rebalance on, say, the 10th and opens a stub period that every monthly
    # metric then treats as a full one.
    #
    # Tested against the calendar rather than "is there data next month", because
    # a dataset that legitimately ends on 31 December would fail that version.
    if drop_partial_final_month and len(month_ends) and not _is_true_month_end(month_ends[-1]):
        log.info(
            "Dropping %s as a rebalance date: its month is still in progress.",
            month_ends[-1].date(),
        )
        month_ends = month_ends[:-1]

    if warmup_trading_days > 0:
        cutoff = index[warmup_trading_days - 1]
        month_ends = month_ends[month_ends >= cutoff]

    if len(month_ends) == 0:
        raise DataUnavailableError(
            f"no rebalance dates survive a {warmup_trading_days}-day warmup"
        )
    return month_ends


def asof_slice(frame: pd.DataFrame | pd.Series, date: pd.Timestamp):
    """Everything at or before ``date`` -- the look-ahead guard, in one place.

    Every estimator in this project (covariance, trailing volatility, financing
    cost) takes its input through here. Centralising it means the invariant is
    enforced in a single readable line instead of being re-derived at each call
    site, and ``tests/test_lookahead.py`` has one function to attack.
    """
    return frame.loc[frame.index <= pd.Timestamp(date)]
