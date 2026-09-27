"""Reading uploaded files into DataFrames, and streaming Dataset Versions out.

Input formats: CSV, Parquet, JSONL. Output: CSV (default) and Parquet.
"""

from __future__ import annotations

import io

import pandas as pd


class UnsupportedFormatError(ValueError):
    pass


class MalformedFileError(ValueError):
    pass


SUPPORTED_SUFFIXES = (".csv", ".parquet", ".jsonl")


def read_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    """Parse raw upload bytes into a DataFrame by filename suffix."""
    lowered = filename.lower()
    try:
        if lowered.endswith(".csv"):
            return pd.read_csv(io.BytesIO(content))
        if lowered.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(content), engine="pyarrow")
        if lowered.endswith(".jsonl"):
            return pd.read_json(io.BytesIO(content), lines=True, orient="records")
    except (pd.errors.EmptyDataError, ValueError, UnicodeDecodeError) as exc:
        raise MalformedFileError(f"{filename} could not be parsed: {exc}") from exc
    except Exception as exc:  # pyarrow / json decode errors
        raise MalformedFileError(f"{filename} could not be parsed: {exc}") from exc
    raise UnsupportedFormatError(
        f"unsupported file type for {filename!r}; use one of {SUPPORTED_SUFFIXES}"
    )


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow", index=False)
    return buffer.getvalue()
