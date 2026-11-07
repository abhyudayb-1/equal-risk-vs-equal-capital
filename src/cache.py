"""Tiny on-disk cache so repeated runs do not re-hit yfinance / FRED.

Everything lands in ``data/cache/`` (gitignored) as plain CSV: no pyarrow
dependency, and the cache stays human-inspectable when a result looks
surprising. Writes go to a ``.tmp`` file that is then renamed into place, so an
interrupted run cannot leave a half-written cache file behind.
"""

from __future__ import annotations

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
