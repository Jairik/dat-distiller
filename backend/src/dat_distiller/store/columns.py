"""Column type inference.

One shared heuristic so Dataset Version previews, Profiles (#16) and the
Train step agree on what a column *is*. ``kind`` values:
``bool``, ``integer``, ``number``, ``datetime``, ``categorical``, ``text``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

#: A non-numeric object column at least this sparse in unique values (or with
#: at most this many distinct values) is categorical rather than free text.
_CATEGORICAL_MAX_CARDINALITY = 20
_CATEGORICAL_MAX_RATIO = 0.5
#: Free text tends to be long strings; short tokens look like categories.
_CATEGORICAL_MAX_MEAN_LENGTH = 64.0


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    dtype: str
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_kind(series: pd.Series) -> str:
    """Infer the semantic kind of one column."""
    non_null = series.dropna()
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_bool_dtype(non_null) and len(non_null) > 0:
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "number"
    if len(non_null) == 0:
        return "text"
    # object column holding numbers (possibly with nulls)?
    try:
        converted = pd.to_numeric(non_null, errors="coerce")
    except (TypeError, ValueError):
        converted = None
    if converted is not None and converted.notna().all():
        if (converted == converted.astype("int64")).all():
            return "integer"
        return "number"
    strings = non_null.astype(str)
    nunique = strings.nunique()
    mean_len = float(strings.str.len().mean())
    if nunique <= _CATEGORICAL_MAX_CARDINALITY or (
        nunique / len(non_null) <= _CATEGORICAL_MAX_RATIO
        and mean_len <= _CATEGORICAL_MAX_MEAN_LENGTH
    ):
        return "categorical"
    return "text"


def infer_column_info(df: pd.DataFrame) -> list[ColumnInfo]:
    return [
        ColumnInfo(name=str(col), dtype=str(df[col].dtype), kind=infer_kind(df[col]))
        for col in df.columns
    ]


def epoch_seconds(series: pd.Series) -> np.ndarray:
    """A datetime column as epoch seconds, with a missing timestamp left missing.

    Casting a datetime64 column to int64 does not turn ``NaT`` into a gap — it
    turns it into the int64 minimum, about 5e10 times larger than any real date
    and on the wrong side of it. That number is finite, so it passes every
    ``isfinite`` guard downstream and lands in a median, a mean, a scale or a
    distance as though it were a real observation. ``np.datetime64('NaT')``
    widened to float gives the same sentinel, so the gaps have to be masked
    explicitly against ``isna()``.

    Lives beside :func:`infer_kind` because the two answer one question between
    them: what kind is this column, and what are its values as numbers. Every
    caller that needs the second needs the first, and the conversion has been
    written out five times.
    """
    converted = pd.to_datetime(series, errors="coerce", utc=True)
    # `to_numpy(dtype="datetime64[ns]")` first, because the resolution is not
    # guaranteed: pandas carries datetime64[s] and datetime64[us] as readily as
    # [ns], and casting whatever is there to int64 and dividing by 1e9 then
    # yields milliseconds, quietly, for a third of the values in a frame.
    values = converted.to_numpy(dtype="datetime64[ns]").astype("int64") / 1_000_000_000.0
    values[converted.isna().to_numpy()] = np.nan
    return values
