"""Profile: a statistical description of a table's columns.

Built from an uploaded sample (``build_profile``) or from user-declared
**Column Specs** when there is no sample (``profile_from_specs``). Profiles
are JSON-serializable so they can go into Provider prompts and Fidelity
Reports. Kind vocabulary matches ``dat_distiller.store.columns``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .store.columns import infer_kind
from .store.provenance import PROVENANCE_COLUMN

SPEC_TYPES = ("number", "integer", "categorical", "bool", "datetime", "text")


@dataclass
class ColumnProfile:
    name: str
    kind: str
    missing_rate: float
    # numeric / integer
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    quantiles: list[float] | None = None  # 0, 10, 25, 50, 75, 90, 100 percent
    # categorical / bool
    frequencies: dict[str, int] | None = None
    categories: list[str] | None = None  # known categories (specs or sample)
    # datetime
    min_datetime: str | None = None
    max_datetime: str | None = None
    # text
    mean_length: float | None = None
    cardinality: int | None = None
    examples: list[str] | None = None
    # declared-only markers (specs path): ranges/constraints without a sample
    declared_only: bool = False
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Profile:
    columns: list[ColumnProfile]
    correlations: dict[str, dict[str, float]] = field(default_factory=dict)
    row_count: int = 0
    #: True when fitted on a Provider-produced seed set rather than real data
    provider_derived: bool = False
    #: "sample" when built from real rows, "specs" when declared-only
    source: str = "sample"

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "columns": [c.to_dict() for c in self.columns],
            "correlations": self.correlations,
            "row_count": self.row_count,
            "provider_derived": self.provider_derived,
            "source": self.source,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "Profile":
        return cls(
            columns=[ColumnProfile(**c) for c in data["columns"]],
            correlations=data.get("correlations", {}),
            row_count=data.get("row_count", 0),
            provider_derived=data.get("provider_derived", False),
            source=data.get("source", "sample"),
        )


def _quantiles(values: pd.Series) -> list[float]:
    qs = values.quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return [round(float(q), 6) for q in qs]


def _profile_column(series: pd.Series) -> ColumnProfile:
    kind = infer_kind(series)
    non_null = series.dropna()
    base = ColumnProfile(
        name=str(series.name),
        kind=kind,
        missing_rate=round(float(series.isna().mean()), 6),
    )
    if kind in ("number", "integer"):
        values = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(values):
            base.min = round(float(values.min()), 6)
            base.max = round(float(values.max()), 6)
            base.mean = round(float(values.mean()), 6)
            base.std = round(float(values.std()), 6) if len(values) > 1 else 0.0
            base.quantiles = _quantiles(values)
    elif kind in ("categorical", "bool"):
        value_counts = non_null.astype(str).value_counts()
        base.frequencies = {str(k): int(v) for k, v in value_counts.items()}
        base.categories = sorted(base.frequencies)
    elif kind == "datetime":
        values = pd.to_datetime(non_null, errors="coerce").dropna()
        if len(values):
            base.min_datetime = str(values.min())
            base.max_datetime = str(values.max())
    else:  # text
        strings = non_null.astype(str)
        if len(strings):
            base.mean_length = round(float(strings.str.len().mean()), 2)
            base.cardinality = int(strings.nunique())
            base.examples = [s[:120] for s in strings.drop_duplicates().head(3)]
    return base


def build_profile(df: pd.DataFrame, *, provider_derived: bool = False) -> Profile:
    """Profile real rows (the ``sample`` path)."""
    frame = df.drop(columns=[PROVENANCE_COLUMN], errors="ignore")
    columns = [_profile_column(frame[col]) for col in frame.columns]
    numeric = frame.select_dtypes(include=[np.number])
    correlations: dict[str, dict[str, float]] = {}
    if numeric.shape[1] >= 2:
        matrix = numeric.corr(numeric_only=True)
        correlations = {
            str(a): {str(b): round(float(v), 6) for b, v in row.items() if pd.notna(v)}
            for a, row in matrix.iterrows()
        }
    return Profile(
        columns=columns,
        correlations=correlations,
        row_count=len(frame),
        provider_derived=provider_derived,
        source="sample",
    )


# -- Column Specs ---------------------------------------------------------------


@dataclass
class ColumnSpec:
    name: str
    type: str  # one of SPEC_TYPES
    min: float | None = None
    max: float | None = None
    categories: list[str] | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColumnSpec":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


def validate_specs(specs: list[ColumnSpec]) -> list[str]:
    """Human-readable validation errors; empty list means the specs are fine."""
    errors: list[str] = []
    seen: set[str] = set()
    for i, spec in enumerate(specs):
        where = f"column {spec.name or i}"
        if not spec.name:
            errors.append(f"{where}: name is required")
        elif spec.name in seen:
            errors.append(f"{where}: duplicate name")
        seen.add(spec.name)
        if spec.type not in SPEC_TYPES:
            errors.append(f"{where}: type must be one of {SPEC_TYPES}")
        if spec.type in ("number", "integer") and spec.min is not None and spec.max is not None:
            if spec.min > spec.max:
                errors.append(f"{where}: min must be <= max")
        if spec.type == "categorical" and not spec.categories:
            errors.append(f"{where}: categorical columns need at least one category")
    return errors


def profile_from_specs(specs: list[ColumnSpec]) -> Profile:
    """Partial Profile from declared Column Specs (no sample available)."""
    errors = validate_specs(specs)
    if errors:
        raise ValueError("; ".join(errors))
    columns = []
    for spec in specs:
        base = ColumnProfile(
            name=spec.name,
            kind=spec.type,
            missing_rate=0.0,
            declared_only=True,
            description=spec.description,
        )
        if spec.type in ("number", "integer"):
            base.min = float(spec.min) if spec.min is not None else None
            base.max = float(spec.max) if spec.max is not None else None
        elif spec.type == "categorical":
            base.categories = list(spec.categories or [])
        columns.append(base)
    return Profile(columns=columns, correlations={}, row_count=0, source="specs")
