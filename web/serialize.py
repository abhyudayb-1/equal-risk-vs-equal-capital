"""JSON-safe serialisation for the exporter."""
from __future__ import annotations

import dataclasses
from datetime import datetime
import math

import numpy as np
import pandas as pd


def clean(value):
    if dataclasses.is_dataclass(value):
        return {field.name: clean(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, pd.DataFrame):
        axis = value.index.name or ("date" if isinstance(value.index, pd.DatetimeIndex) else "name")
        if isinstance(value.index, pd.RangeIndex):
            return clean(value.to_dict("records"))
        return clean(value.rename_axis(axis).reset_index().to_dict("records"))
    if isinstance(value, pd.Series):
        return {str(clean(k)): clean(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()[:10]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
