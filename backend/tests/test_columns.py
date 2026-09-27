"""Column kind inference on representative data."""

from __future__ import annotations

import pandas as pd
from dat_distiller.store.columns import infer_column_info, infer_kind


def kinds(df: pd.DataFrame) -> dict[str, str]:
    return {c.name: c.kind for c in infer_column_info(df)}


def test_basic_kinds() -> None:
    df = pd.DataFrame(
        {
            "count": pd.Series([1, 2, 3]),
            "price": pd.Series([1.5, 2.5, 3.5]),
            "flag": pd.Series([True, False, True]),
            "when": pd.Series(pd.to_datetime(["2024-01-01", "2024-06-01", "2025-01-01"])),
        }
    )
    assert kinds(df) == {
        "count": "integer",
        "price": "number",
        "flag": "bool",
        "when": "datetime",
    }


def test_categorical_versus_free_text() -> None:
    colours = ["red", "green", "blue"]
    df = pd.DataFrame(
        {
            "colour": [colours[i % 3] for i in range(200)],
            "review": [
                "a long free text sentence about the product " + str(i)
                for i in range(200)
            ],
        }
    )
    assert kinds(df) == {"colour": "categorical", "review": "text"}


def test_mixed_and_missing_values() -> None:
    df = pd.DataFrame(
        {
            "maybe_int": pd.Series([1, 2, None, 4], dtype=object),
            "maybe_num": pd.Series([1, 2.5, None, 4], dtype=object),
            "all_null": pd.Series([None, None, None]),
            "sparse_text": pd.Series([f"unique value number {i} with length" for i in range(50)]),
        }
    )
    inferred = kinds(df)
    assert inferred["maybe_int"] == "integer"
    assert inferred["maybe_num"] == "number"
    assert inferred["all_null"] == "text"
    assert inferred["sparse_text"] == "text"  # every value unique, long -> free text


def test_dtype_recorded() -> None:
    df = pd.DataFrame({"x": [1, 2]})
    (info,) = infer_column_info(df)
    assert info.dtype == str(df["x"].dtype)
    assert infer_kind(df["x"]) == "integer"
