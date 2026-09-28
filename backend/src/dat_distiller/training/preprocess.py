"""The preprocessing pipeline: a serializable object a Model Bundle reuses.

Hand-written on numpy + pandas — no scikit-learn. Four named steps, all fitted
on the training split and stored as a plain dict:

- **imputer** — median for numbers, a ``__missing__`` marker category for
  categorical columns, ``0.0`` for booleans
- **scaler** — per-column z-score
- **one_hot** — one column per observed category
- **tfidf** — L2-normalized word n-grams for text columns, with the fitted
  vocabulary and idf weights

``Preprocessor.to_dict()`` is JSON-able and ``Preprocessor.from_dict()`` brings
it back with identical output, so a Training Run persists exactly what it fitted
and the Model Bundle (#34) ships and re-applies that same object.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..store.columns import infer_kind

#: Bumped when the dict shape changes so a bundle can refuse an old spec.
SPEC_VERSION = 1
#: Category standing in for a missing value in one-hot encoding.
MISSING_MARKER = "__missing__"
DEFAULT_NGRAM_RANGE = (1, 1)
DEFAULT_MAX_FEATURES = 2_000
DEFAULT_MIN_DF = 1

#: Semantic roles a feature column plays, and the step that handles it.
ROLE_NUMERIC = "numeric"
ROLE_CATEGORICAL = "categorical"
ROLE_TEXT = "text"
ROLE_BOOL = "bool"
ROLE_DATETIME = "datetime"
ROLES = (ROLE_NUMERIC, ROLE_CATEGORICAL, ROLE_TEXT, ROLE_BOOL, ROLE_DATETIME)

_KIND_TO_ROLE = {
    "bool": ROLE_BOOL,
    "integer": ROLE_NUMERIC,
    "number": ROLE_NUMERIC,
    "datetime": ROLE_DATETIME,
    "categorical": ROLE_CATEGORICAL,
    "text": ROLE_TEXT,
}

_WORD = re.compile(r"\w+", re.UNICODE)


# -- spec dataclasses (all JSON-able) ---------------------------------------


@dataclass
class ImputerSpec:
    kind: str = "median"
    marker: str = MISSING_MARKER
    values: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "marker": self.marker, "values": dict(self.values)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ImputerSpec:
        return cls(
            kind=str(payload.get("kind", "median")),
            marker=str(payload.get("marker", MISSING_MARKER)),
            values={str(k): float(v) for k, v in (payload.get("values") or {}).items()},
        )


@dataclass
class ScalerSpec:
    kind: str = "standard"
    with_mean: bool = True
    with_std: bool = True
    mean: dict[str, float] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "with_mean": bool(self.with_mean),
            "with_std": bool(self.with_std),
            "mean": dict(self.mean),
            "scale": dict(self.scale),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScalerSpec:
        return cls(
            kind=str(payload.get("kind", "standard")),
            with_mean=bool(payload.get("with_mean", True)),
            with_std=bool(payload.get("with_std", True)),
            mean={str(k): float(v) for k, v in (payload.get("mean") or {}).items()},
            scale={str(k): float(v) for k, v in (payload.get("scale") or {}).items()},
        )


@dataclass
class OneHotSpec:
    kind: str = "one_hot"
    marker: str = MISSING_MARKER
    categories: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "marker": self.marker,
            "categories": {k: list(v) for k, v in self.categories.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OneHotSpec:
        return cls(
            kind=str(payload.get("kind", "one_hot")),
            marker=str(payload.get("marker", MISSING_MARKER)),
            categories={
                str(k): [str(c) for c in v]
                for k, v in (payload.get("categories") or {}).items()
            },
        )


@dataclass
class TfidfSpec:
    kind: str = "tfidf"
    lowercase: bool = True
    ngram_range: list[int] = field(default_factory=lambda: list(DEFAULT_NGRAM_RANGE))
    min_df: int = DEFAULT_MIN_DF
    max_features: int = DEFAULT_MAX_FEATURES
    norm: str = "l2"
    vocabulary: dict[str, list[str]] = field(default_factory=dict)
    idf: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "lowercase": bool(self.lowercase),
            "ngram_range": [int(n) for n in self.ngram_range],
            "min_df": int(self.min_df),
            "max_features": int(self.max_features),
            "norm": self.norm,
            "vocabulary": {k: list(v) for k, v in self.vocabulary.items()},
            "idf": {k: [float(w) for w in v] for k, v in self.idf.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TfidfSpec:
        return cls(
            kind=str(payload.get("kind", "tfidf")),
            lowercase=bool(payload.get("lowercase", True)),
            ngram_range=[int(n) for n in (payload.get("ngram_range") or DEFAULT_NGRAM_RANGE)],
            min_df=int(payload.get("min_df", DEFAULT_MIN_DF)),
            max_features=int(payload.get("max_features", DEFAULT_MAX_FEATURES)),
            norm=str(payload.get("norm", "l2")),
            vocabulary={
                str(k): [str(t) for t in v] for k, v in (payload.get("vocabulary") or {}).items()
            },
            idf={
                str(k): [float(w) for w in v] for k, v in (payload.get("idf") or {}).items()
            },
        )


@dataclass
class PreprocessorSpec:
    """Everything needed to reproduce the transformed matrix, as plain data."""

    features: list[str] = field(default_factory=list)
    roles: dict[str, str] = field(default_factory=dict)
    imputer: ImputerSpec = field(default_factory=ImputerSpec)
    scaler: ScalerSpec = field(default_factory=ScalerSpec)
    one_hot: OneHotSpec = field(default_factory=OneHotSpec)
    tfidf: TfidfSpec = field(default_factory=TfidfSpec)
    output_names: list[str] = field(default_factory=list)
    output_sources: dict[str, str] = field(default_factory=dict)
    version: int = SPEC_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "features": list(self.features),
            "roles": dict(self.roles),
            "imputer": self.imputer.to_dict(),
            "scaler": self.scaler.to_dict(),
            "one_hot": self.one_hot.to_dict(),
            "tfidf": self.tfidf.to_dict(),
            "output_names": list(self.output_names),
            "output_sources": dict(self.output_sources),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PreprocessorSpec:
        version = int(payload.get("version", SPEC_VERSION))
        if version != SPEC_VERSION:
            raise ValueError(
                f"preprocessing spec version {version} is not supported (expected {SPEC_VERSION})"
            )
        return cls(
            features=[str(c) for c in (payload.get("features") or [])],
            roles={str(k): str(v) for k, v in (payload.get("roles") or {}).items()},
            imputer=ImputerSpec.from_dict(payload.get("imputer") or {}),
            scaler=ScalerSpec.from_dict(payload.get("scaler") or {}),
            one_hot=OneHotSpec.from_dict(payload.get("one_hot") or {}),
            tfidf=TfidfSpec.from_dict(payload.get("tfidf") or {}),
            output_names=[str(c) for c in (payload.get("output_names") or [])],
            output_sources={
                str(k): str(v) for k, v in (payload.get("output_sources") or {}).items()
            },
            version=version,
        )


# -- column helpers ---------------------------------------------------------


def column_role(series: pd.Series) -> str:
    """The role a column plays downstream, from the shared kind heuristic."""
    return _KIND_TO_ROLE.get(infer_kind(series), ROLE_CATEGORICAL)


def _numeric_values(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


def _datetime_values(series: pd.Series) -> np.ndarray:
    """Epoch seconds, with a missing timestamp left missing.

    Casting a datetime64 column to int64 does not turn ``NaT`` into a gap — it
    turns it into the int64 minimum, about 5e10 times larger than any real date
    and on the wrong side of it. That number is finite, so the isfinite guards
    below wave it through into the median, the mean and the scale fitted for the
    column: one missing timestamp inflates the scale enough to squash every real
    date into a sliver, and the corrupted pipeline is what gets persisted on the
    run and shipped in the Model Bundle. So the gaps are put back explicitly.
    """
    converted = pd.to_datetime(series, errors="coerce", utc=True)
    values = converted.astype("int64").to_numpy(dtype="float64") / 1_000_000_000.0
    values[converted.isna().to_numpy()] = np.nan
    return values


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    return isinstance(value, float) and math.isnan(value)


def _bool_values(series: pd.Series) -> np.ndarray:
    out = np.zeros(len(series), dtype="float64")
    for index, value in enumerate(series.to_numpy(dtype=object)):
        if _is_missing(value):
            continue
        try:
            out[index] = 1.0 if bool(value) else 0.0
        except (TypeError, ValueError):
            continue
    return out


def _category_values(series: pd.Series, marker: str) -> np.ndarray:
    out = np.full(len(series), marker, dtype=object)
    for index, value in enumerate(series.to_numpy(dtype=object)):
        if not _is_missing(value):
            out[index] = str(value)
    return out.astype(str)


def _median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.median(finite))


def _tokenize(text: str, lowercase: bool = True) -> list[str]:
    return _WORD.findall(text.lower() if lowercase else text)


def ngrams(tokens: Sequence[str], ngram_range: Sequence[int]) -> list[str]:
    low, high = int(ngram_range[0]), int(ngram_range[1])
    out: list[str] = []
    for size in range(max(1, low), max(1, high) + 1):
        if size == 1:
            out.extend(tokens)
        else:
            out.extend(
                " ".join(tokens[start : start + size]) for start in range(len(tokens) - size + 1)
            )
    return out


# -- the pipeline -----------------------------------------------------------


class Preprocessor:
    """A fitted preprocessing pipeline: fit on the training rows, reuse anywhere."""

    def __init__(self, spec: PreprocessorSpec) -> None:
        self.spec = spec

    # -- fitting -------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        *,
        features: Sequence[str],
        ngram_range: Sequence[int] = DEFAULT_NGRAM_RANGE,
        min_df: int = DEFAULT_MIN_DF,
        max_features: int = DEFAULT_MAX_FEATURES,
        scale: bool = True,
    ) -> Preprocessor:
        columns = [str(c) for c in features]
        if not columns:
            raise ValueError("no feature columns to preprocess")
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"feature columns missing from the data: {missing}")

        spec = PreprocessorSpec(features=columns)
        spec.scaler.kind = "standard" if scale else "none"
        spec.tfidf.ngram_range = [int(n) for n in ngram_range]
        spec.tfidf.min_df = int(min_df)
        spec.tfidf.max_features = int(max_features)
        text_columns: list[str] = []

        for column in columns:
            series = df[column]
            role = column_role(series)
            spec.roles[column] = role
            if role in (ROLE_NUMERIC, ROLE_DATETIME):
                values = (
                    _numeric_values(series) if role == ROLE_NUMERIC else _datetime_values(series)
                )
                fill = _median(values)
                spec.imputer.values[column] = fill
                if scale:
                    filled = np.where(np.isfinite(values), values, fill)
                    spread = float(filled.std(ddof=0))
                    spec.scaler.mean[column] = float(filled.mean())
                    spec.scaler.scale[column] = spread if spread > 0 else 1.0
            elif role == ROLE_BOOL:
                spec.imputer.values[column] = 0.0
            elif role == ROLE_CATEGORICAL:
                spec.one_hot.categories[column] = _fit_categories(series, spec.one_hot.marker)
            else:
                text_columns.append(column)

        for column in text_columns:
            _fit_text(spec, df[column], spec.tfidf, column)

        cls._name_outputs(spec)
        return cls(spec)

    @staticmethod
    def _name_outputs(spec: PreprocessorSpec) -> None:
        names: list[str] = []
        sources: dict[str, str] = {}
        for column in spec.features:
            role = spec.roles.get(column, ROLE_CATEGORICAL)
            if role in (ROLE_NUMERIC, ROLE_BOOL, ROLE_DATETIME):
                names.append(column)
                sources[column] = column
            elif role == ROLE_CATEGORICAL:
                for category in spec.one_hot.categories.get(column, []):
                    name = f"{column}={category}"
                    names.append(name)
                    sources[name] = column
            else:
                for term in spec.tfidf.vocabulary.get(column, []):
                    name = f"tfidf:{column}={term}"
                    names.append(name)
                    sources[name] = column
        spec.output_names = names
        spec.output_sources = sources

    # -- transforming ---------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        """The transformed column names, in matrix order."""
        return list(self.spec.output_names)

    @property
    def n_features(self) -> int:
        return len(self.spec.output_names)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """The numeric matrix a Model is fitted on."""
        spec = self.spec
        missing = [c for c in spec.features if c not in df.columns]
        if missing:
            raise ValueError(f"feature columns missing from the data: {missing}")
        blocks: list[np.ndarray] = []
        for column in spec.features:
            role = spec.roles.get(column, ROLE_CATEGORICAL)
            series = df[column]
            if role == ROLE_NUMERIC:
                blocks.append(_scale_block(_numeric_values(series), spec, column))
            elif role == ROLE_DATETIME:
                blocks.append(_scale_block(_datetime_values(series), spec, column))
            elif role == ROLE_BOOL:
                blocks.append(_bool_values(series).reshape(-1, 1))
            elif role == ROLE_CATEGORICAL:
                blocks.append(_one_hot_block(series, spec.one_hot.categories.get(column, []), spec.one_hot.marker))
            else:
                blocks.append(_tfidf_block(series, spec.tfidf, column))
        if not blocks:
            return np.zeros((len(df), 0), dtype="float64")
        matrix = np.hstack(blocks)
        if matrix.shape[1] != self.n_features:  # pragma: no cover - defensive
            raise ValueError(
                f"preprocessing produced {matrix.shape[1]} columns, expected {self.n_features}"
            )
        return np.ascontiguousarray(matrix, dtype="float64")

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """The same matrix as a labelled DataFrame (feature importance, debugging)."""
        return pd.DataFrame(
            self.transform(df), columns=self.feature_names, index=df.index
        )

    # -- serializing -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return self.spec.to_dict()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Preprocessor:
        return cls(PreprocessorSpec.from_dict(payload))

    @classmethod
    def from_json(cls, text: str) -> Preprocessor:
        return cls.from_dict(json.loads(text))


def fit_preprocessor(
    df: pd.DataFrame,
    *,
    features: Sequence[str],
    ngram_range: Sequence[int] = DEFAULT_NGRAM_RANGE,
    min_df: int = DEFAULT_MIN_DF,
    max_features: int = DEFAULT_MAX_FEATURES,
    scale: bool = True,
) -> Preprocessor:
    """Fit a :class:`Preprocessor` on ``df``'s feature columns."""
    return Preprocessor.fit(
        df,
        features=features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        scale=scale,
    )


def transform_with_spec(df: pd.DataFrame, spec: dict[str, Any] | PreprocessorSpec) -> np.ndarray:
    """Apply a persisted spec — how a Model Bundle reuses the Training Run's work."""
    preprocessor = (
        spec if isinstance(spec, Preprocessor) else Preprocessor.from_dict(dict(spec))
    )
    return preprocessor.transform(df)


# -- internals --------------------------------------------------------------


def _fit_categories(series: pd.Series, marker: str) -> list[str]:
    """The observed categories, sorted, with the missing marker last so the
    output column order reads naturally."""
    values = set(_category_values(series, marker).tolist())
    categories = sorted(c for c in values if c != marker)
    if marker in values:
        categories.append(marker)
    return categories


def _scale_block(values: np.ndarray, spec: PreprocessorSpec, column: str) -> np.ndarray:
    fill = spec.imputer.values.get(column, 0.0)
    filled = np.where(np.isfinite(values), values, fill)
    if spec.scaler.kind == "none":
        return filled.reshape(-1, 1)
    mean = spec.scaler.mean.get(column, 0.0)
    scale = spec.scaler.scale.get(column, 1.0) or 1.0
    if spec.scaler.with_mean:
        filled = filled - mean
    if spec.scaler.with_std:
        filled = filled / scale
    return filled.reshape(-1, 1)


def _one_hot_block(series: pd.Series, categories: Iterable[str], marker: str) -> np.ndarray:
    columns = list(categories)
    if not columns:
        return np.zeros((len(series), 0), dtype="float64")
    values = _category_values(series, marker)
    return np.column_stack([(values == category).astype("float64") for category in columns])


def _fit_text(spec: PreprocessorSpec, series: pd.Series, tfidf: TfidfSpec, column: str) -> None:
    documents = [
        [] if _is_missing(value) else _tokenize(str(value), tfidf.lowercase)
        for value in series.to_numpy(dtype=object)
    ]
    document_frequency: dict[str, int] = defaultdict(int)
    for tokens in documents:
        for gram in set(ngrams(tokens, tfidf.ngram_range)):
            document_frequency[gram] += 1
    kept = [g for g, count in document_frequency.items() if count >= tfidf.min_df]
    kept.sort(key=lambda g: (-document_frequency[g], g))
    kept = sorted(kept[: tfidf.max_features])
    total = len(documents)
    tfidf.vocabulary[column] = kept
    tfidf.idf[column] = [
        math.log((1.0 + total) / (1.0 + document_frequency[gram])) + 1.0 for gram in kept
    ]


def _tfidf_block(series: pd.Series, tfidf: TfidfSpec, column: str) -> np.ndarray:
    vocabulary = tfidf.vocabulary.get(column, [])
    if not vocabulary:
        return np.zeros((len(series), 0), dtype="float64")
    index = {term: position for position, term in enumerate(vocabulary)}
    weights = np.asarray(tfidf.idf.get(column, [1.0] * len(vocabulary)), dtype="float64")
    matrix = np.zeros((len(series), len(vocabulary)), dtype="float64")
    for row, value in enumerate(series.to_numpy(dtype=object)):
        if _is_missing(value):
            continue
        counts = Counter(ngrams(_tokenize(str(value), tfidf.lowercase), tfidf.ngram_range))
        if not counts:
            continue
        columns = [index[term] for term in counts if term in index]
        if not columns:
            continue
        raw = np.asarray([counts[vocabulary[position]] for position in columns], dtype="float64")
        data = (raw / max(sum(counts.values()), 1)) * weights[columns]
        norm = float(np.linalg.norm(data))
        if tfidf.norm == "l2" and norm > 0:
            data = data / norm
        matrix[row, columns] = data
    return matrix
