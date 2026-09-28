"""Statistical Generation: a Gaussian copula written by hand.

ADR-0002: no SDV, no `copulas` (both BSL-licensed). Everything here is
numpy/scipy.

Method:
1. **Marginals.** Numeric/integer/datetime columns get an empirical marginal
   (sorted values + quantile interpolation); categorical/bool columns get
   frequency tables. Missing rates are recorded to be re-injected.
2. **Normal space.** Each column's u = midrank/n maps through Φ⁻¹ to z; the
   correlation matrix of the z-columns is estimated (nearest PSD fix-up).
3. **Sample.** Draw z ~ N(0, Σ), u = Φ(z), and invert each marginal.
   **Balance Targets** use rejection sampling: draw an over-supply of rows
   and greedily pick rows to satisfy the requested proportions (within
   tolerance) for the requested categorical column. Deterministic per seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..store.columns import infer_kind
from ..store.provenance import PROVENANCE_COLUMN

_BALANCE_TOLERANCE = 0.03


@dataclass
class _Marginal:
    kind: str  # number | integer | datetime | categorical | bool
    name: str
    missing_rate: float
    # numeric/datetime
    sorted_values: np.ndarray | None = None
    dtype: Any = None
    # categorical/bool
    categories: list[str] | None = None
    cumulative: np.ndarray | None = None  # cumulative frequencies in (0, 1]
    #: False when the column held no values at all, so there is no distribution
    #: to sample from. A column that is entirely missing is not a column with an
    #: unknown distribution — it is a column with none, and the honest sample is
    #: missing. Without this the numeric path indexed an empty array and raised
    #: `IndexError: index -1 is out of bounds`, and the categorical path divided
    #: by zero on the way there.
    observed: bool = True

    def to_normal(self, values: pd.Series) -> np.ndarray:
        """Map observed values to normal space via midrank u (NaN stays NaN)."""
        out = np.full(len(values), np.nan)
        mask = values.notna().to_numpy()
        sub = values[mask]
        if len(sub) == 0 or not self.observed:
            return out
        if self.kind in ("categorical", "bool"):
            cat_index = {c: i for i, c in enumerate(self.categories)}
            lower = np.concatenate([[0.0], self.cumulative[:-1]])
            mid = (lower + self.cumulative) / 2.0
            codes = [cat_index[c] for c in sub.astype(str)]
            u = mid[np.asarray(codes)]
        else:
            numeric = self._numeric(sub).astype(float)
            ranks = stats.rankdata(numeric, method="average")
            u = ranks / (len(numeric) + 1.0)
        out[mask] = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        return out

    def from_normal(self, z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.observed:
            # No values were ever seen here, so there is nothing to interpolate
            # between. Missing is the truth, not a placeholder for a guess.
            return np.full(len(z), None, dtype=object)
        u = stats.norm.cdf(z)
        if self.kind in ("categorical", "bool"):
            idx = np.searchsorted(self.cumulative, u, side="right")
            idx = np.clip(idx, 0, len(self.categories) - 1)
            cats = np.array(self.categories, dtype=object)[idx]
            if self.kind == "bool":
                return np.array([c.lower() == "true" for c in cats], dtype=bool)
            return cats
        # quantile interpolation of the empirical marginal
        n = len(self.sorted_values)
        positions = u * (n - 1)
        low = np.floor(positions).astype(int)
        high = np.minimum(low + 1, n - 1)
        frac = positions - low
        out = self.sorted_values[low] * (1 - frac) + self.sorted_values[high] * frac
        if self.kind == "integer":
            out = np.rint(out)
            out = np.clip(out, self.sorted_values.min(), self.sorted_values.max())
            return out.astype("int64")
        if self.kind == "datetime":
            return pd.to_datetime(out.astype("int64"), unit="ns")
        return out

    def _numeric(self, values: pd.Series) -> np.ndarray:
        if self.kind == "datetime":
            return pd.to_datetime(values).astype("int64").to_numpy().astype(float)
        return pd.to_numeric(values).to_numpy(dtype=float)


@dataclass
class GaussianCopula:
    marginals: list[_Marginal]
    correlation: np.ndarray
    n_rows: int
    skipped: list[str] = field(default_factory=list)
    seed: int | None = None

    @property
    def columns(self) -> list[str]:
        return [m.name for m in self.marginals]

    # -- fitting ------------------------------------------------------------

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "GaussianCopula":
        frame = df.drop(columns=[PROVENANCE_COLUMN], errors="ignore")
        marginals: list[_Marginal] = []
        skipped: list[str] = []
        normal_columns: dict[str, np.ndarray] = {}
        for name in frame.columns:
            series = frame[name]
            kind = infer_kind(series)
            if kind == "text":
                skipped.append(str(name))  # free text is the Provider's job (#20)
                continue
            missing = float(series.isna().mean())
            filled = series.dropna()
            # A column with no values at all has no distribution to fit. It stays
            # in the output — the user put it there — and samples as missing.
            observed = len(filled) > 0
            if kind == "bool":
                cats = sorted(set(filled.astype(str))) or ["False", "True"]
                counts = filled.astype(str).value_counts()
                cum = _cumulative(cats, counts, observed)
                marginal = _Marginal(
                    "bool", str(name), missing, categories=cats, cumulative=cum, observed=observed
                )
            elif kind == "categorical":
                cats = sorted(set(filled.astype(str)))
                counts = filled.astype(str).value_counts()
                cum = _cumulative(cats, counts, observed)
                marginal = _Marginal(
                    "categorical",
                    str(name),
                    missing,
                    categories=cats,
                    cumulative=cum,
                    observed=observed,
                )
            else:
                values = marginal_values(kind, filled)
                marginal = _Marginal(
                    kind,
                    str(name),
                    missing,
                    sorted_values=np.sort(values),
                    dtype=series.dtype,
                    observed=observed,
                )
            marginals.append(marginal)
            normal_columns[str(name)] = marginal.to_normal(series)  # NaN-aware

        matrix = np.eye(len(marginals))
        if len(marginals) >= 2:
            z = np.column_stack([normal_columns[m.name] for m in marginals])
            # pairwise-complete correlations (missing rows excluded per pair)
            matrix = pd.DataFrame(z).corr().to_numpy()
            matrix = _nearest_psd(np.nan_to_num(matrix, nan=0.0))
            np.fill_diagonal(matrix, 1.0)
        return cls(marginals=marginals, correlation=matrix, n_rows=len(frame), skipped=skipped)

    # -- sampling ------------------------------------------------------------

    def sample(
        self,
        n: int,
        seed: int = 0,
        balance: dict[str, dict[str, float]] | None = None,
    ) -> pd.DataFrame:
        """Generate n rows. ``balance``: {column: {category: proportion}}.

        Balance uses rejection sampling against an over-supply, so proportions
        are met within ~3% while marginals stay empirical.
        """
        self.seed = seed
        rng = np.random.default_rng(seed)
        chol = _chol(self.correlation)

        def draw(rows: int) -> tuple[np.ndarray, pd.DataFrame]:
            z = rng.standard_normal((rows, len(self.marginals))) @ chol.T
            data: dict[str, np.ndarray] = {}
            for j, marginal in enumerate(self.marginals):
                values = marginal.from_normal(z[:, j], rng)
                if marginal.missing_rate > 0:  # re-inject observed missingness
                    hole = rng.random(rows) < marginal.missing_rate
                    holder = pd.Series(values, dtype="object")
                    holder[hole] = None
                    values = holder.to_numpy()
                data[marginal.name] = values
            return z, pd.DataFrame(data)

        if not balance:
            _, frame = draw(n)
            return frame

        # Rejection sampling for Balance Targets: over-draw, then fill quotas.
        over = max(4 * n, n + 1000)
        z_all, frame = draw(over)
        picked: list[int] = []
        quotas: dict[tuple[str, str], int] = {}
        balance_col, proportions = next(iter(balance.items()))
        for category, share in proportions.items():
            quotas[(balance_col, str(category))] = round(share * n)
        actual = frame[balance_col].astype(str)
        taken_per_group: dict[str, int] = {}
        for i, value in enumerate(actual.tolist()):
            if len(picked) >= n:
                break
            used = taken_per_group.get(value, 0)
            if used < quotas.get((balance_col, value), 0):
                picked.append(i)
                taken_per_group[value] = used + 1
        if len(picked) < n:  # greedy fill to reach n rows
            extras = sorted(
                (i for i in range(len(frame)) if i not in set(picked)),
                key=lambda i: taken_per_group.get(actual.iloc[i], 1e9),
            )
            picked.extend(extras[: n - len(picked)])
        frame = frame.iloc[sorted(picked)].reset_index(drop=True)
        return frame


def marginal_values(kind: str, filled: pd.Series) -> np.ndarray:
    if kind == "datetime":
        return pd.to_datetime(filled).astype("int64").to_numpy().astype(float)
    return pd.to_numeric(filled).to_numpy(dtype=float)


def _cumulative(cats: list[str], counts: Any, observed: bool) -> np.ndarray:
    """Cumulative category frequencies in (0, 1], or an empty array if unobserved.

    Dividing by the number of non-missing values when nothing was observed is a
    division by zero, and the resulting NaNs went on to index an empty category
    list a sample later. `counts.sum()` is the same denominator and cannot be
    zero here, because `observed` is what says there was at least one value.
    """
    if not observed:
        return np.array([], dtype="float64")
    return np.cumsum([counts.get(c, 0) for c in cats]) / float(counts.sum())


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if eigenvalues.min() >= 1e-8:
        return matrix
    eigenvalues = np.clip(eigenvalues, 1e-8, None)
    fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.diag(fixed))
    return fixed / np.outer(scale, scale)


def _chol(correlation: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(np.clip(correlation, -1, 1))
        return eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0, None)))
