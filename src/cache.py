"""Tiny on-disk cache so repeated runs do not re-hit yfinance / FRED.

Everything lands in ``data/cache/`` (gitignored). CSV + JSON are used on
purpose: no pyarrow dependency, and the cache stays human-inspectable when a
backtest result looks surprising.

Ported verbatim from ``abhyudayb-1/sp500-factor-backtester``. It was already
exactly what this project needs -- atomic ``.tmp``-then-``replace`` writes so an
interrupted run cannot leave a half-written cache file behind. Do not rewrite it.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from .config import CACHE_DIR

log = logging.getLogger(__name__)


def _path(name: str) -> Path:
    return CACHE_DIR / name


def is_fresh(name: str, max_age_hours: float) -> bool:
    """True if the cache entry exists and is younger than ``max_age_hours``."""
    p = _path(name)
    if not p.exists():
        return False
    age_hours = (time.time() - p.stat().st_mtime) / 3600.0
    return age_hours < max_age_hours


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------
def read_json(name: str, default=None):
    p = _path(name)
    if not p.exists():
        return default
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Cache file %s is unreadable (%s); ignoring it.", name, exc)
        return default


def write_json(name: str, payload) -> None:
    p = _path(name)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    tmp.replace(p)


# --------------------------------------------------------------------------
# DataFrames
# --------------------------------------------------------------------------
def read_frame(name: str, parse_dates_index: bool = True) -> pd.DataFrame | None:
    p = _path(name)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, index_col=0)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        log.warning("Cache file %s is unreadable (%s); ignoring it.", name, exc)
        return None
    if parse_dates_index:
        df.index = pd.to_datetime(df.index)
    return df


def write_frame(name: str, df: pd.DataFrame) -> None:
    p = _path(name)
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp)
    tmp.replace(p)


def read_table(name: str) -> pd.DataFrame | None:
    """Read a cached table that has a plain RangeIndex (no index column).

    Distinct from :func:`read_frame`, which treats the first CSV column as the
    index -- correct for the date-indexed price frame, wrong for a tidy panel
    whose first column is real data.
    """
    p = _path(name)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        log.warning("Cache file %s is unreadable (%s); ignoring it.", name, exc)
        return None


def write_table(name: str, df: pd.DataFrame) -> None:
    """Write a tidy table without its RangeIndex."""
    p = _path(name)
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(p)
