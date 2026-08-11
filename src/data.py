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

import numpy as np
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
            return cached

    log.info("Downloading %s from yfinance...", ", ".join(tickers))
    raw = yf.download(
        tickers,
        start=start,
        end=end,
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
    return close


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
            return cached.iloc[:, 0].rename("risk_free")

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

    annual_percent = _parse_fred_csv(response.text, series_id)
    annual_percent = annual_percent.loc[annual_percent.index >= pd.Timestamp(start)]
    if annual_percent.empty:
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
    return daily


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
