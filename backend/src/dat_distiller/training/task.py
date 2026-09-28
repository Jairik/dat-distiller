"""The Target, its Task Type, and the Label Column leakage guard.

Any column can be the Target; the Task Type is *inferred* and the user can
change it. The interesting part is Labeling's output shape: a Label Column
``is_spam`` comes with siblings ``is_spam__confidence``,
``is_spam__p_<option>`` (Choice) or ``is_spam__probabilities`` (Score). Any of
them is a valid Target, and whichever one you pick, the rest of the family
leaks the answer — so they are excluded together, by prefix.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..store.columns import infer_kind

TASK_TYPES = ("classification", "regression")

#: A numeric column with at most this many distinct values reads as a set of
#: classes rather than as a quantity. Above it, a number is treated as a quantity.
MAX_INFERRED_CLASSES = 20

#: Separator Labeling puts between a Label Column and its siblings.
FAMILY_SEPARATOR = "__"
#: Sibling columns written for every Jev Question.
SIBLING_SUFFIXES = ("confidence", "probabilities")
#: Choice questions write one probability column per option, ``<name>__p_<option>``.
OPTION_SUFFIX_PREFIX = "p_"

#: Source kinds that become the "numeric" role downstream.
NUMERIC_KINDS = ("integer", "number")


def label_family(column: str) -> str:
    """The Label Column family a column belongs to.

    Every column Labeling writes for a Jev Question starts with the question's
    name, so the family is that prefix: ``tone``, ``tone__confidence`` and
    ``tone__p_pos`` all live in the ``tone`` family.
    """
    return str(column).split(FAMILY_SEPARATOR, 1)[0]


def sibling_suffix(column: str) -> str | None:
    """The part after ``__`` when the column is a known sibling, else ``None``."""
    if FAMILY_SEPARATOR not in str(column):
        return None
    suffix = str(column).split(FAMILY_SEPARATOR, 1)[1]
    if suffix in SIBLING_SUFFIXES or suffix.startswith(OPTION_SUFFIX_PREFIX):
        return suffix
    return None


def is_label_sibling(column: str) -> bool:
    """Whether the column is one of Labeling's sibling columns."""
    return sibling_suffix(column) is not None


def family_columns(columns: Iterable[str], family: str) -> list[str]:
    """The whole family of ``family``: the column itself plus every ``<family>__*``.

    Prefix-based on purpose — Labeling is free to add another sibling kind
    later and the guard must keep up without a change here.
    """
    prefix = f"{family}{FAMILY_SEPARATOR}"
    return [c for c in columns if c == family or str(c).startswith(prefix)]


def sibling_columns(target: str, columns: Iterable[str]) -> list[str]:
    """The Target's own family, minus the Target: columns that leak the answer."""
    return [c for c in family_columns(columns, label_family(target)) if c != target]


def _is_integral(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype="float64")
    if values.size == 0:
        return False
    return bool(np.all(np.isclose(values, np.round(values))))


def infer_task_type(series: pd.Series) -> str:
    """Infer ``classification`` or ``regression`` from the Target's values.

    - booleans, categories, free text and datetimes: classification
    - numbers: classification while the values read as a small set of classes
      that repeat across rows (a Noul answer, Score levels), regression once
      they behave like a quantity — an Noul confidence or a Choice probability,
      say, which stays continuous even on a small frame
    """
    kind = infer_kind(series)
    if kind in ("bool", "categorical", "text", "datetime"):
        return "classification"
    if kind not in NUMERIC_KINDS:
        return "regression"
    non_null = series.dropna()
    distinct = int(non_null.nunique())
    if distinct > MAX_INFERRED_CLASSES or distinct > max(2, len(non_null) // 2):
        return "regression"  # too many values, or they barely repeat: a quantity
    if kind == "integer":
        return "classification"
    return "classification" if _is_integral(non_null) else "regression"


def resolve_task_type(series: pd.Series, override: str | None = None) -> str:
    """The user's Task Type when they set one, else the inferred one."""
    if override is None:
        return infer_task_type(series)
    if override not in TASK_TYPES:
        raise ValueError(f"task type must be one of {TASK_TYPES}")
    return override
