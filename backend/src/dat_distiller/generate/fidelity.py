"""Fidelity Report: how close did Generation get to the source data?

Compares generated rows against the sample (and its Profile):

- **KS tests** per numeric column (generated vs sample distributions);
- **chi-square** per categorical/bool column;
- **correlation drift** (max |Δ| over numeric pairs, sample vs generated);
- **exact duplicates** within the generated rows;
- **near-copies** of sample rows — a privacy guard: mean per-column distance
  over shared columns, with a configurable threshold (numeric distances are
  range-normalized; equal categories count 0, different ones count 1);
- **dropped Provider rows**, **actual vs requested Balance Targets**, and
  whether the Profile was **Provider-derived**.

The dict is JSON-serializable and stored in the generated Dataset Version's
``meta``; :func:`raise_fidelity_checks` turns the risky findings into warning
**Checks** the user must acknowledge.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..store.columns import epoch_seconds, infer_kind
from ..store.provenance import PROVENANCE_COLUMN

NEAR_COPY_THRESHOLD = 0.05
HIGH_DROP_RATIO = 0.1
KS_DRIFT_P = 0.01  # p-value below this = the distributions clearly moved
CORR_DRIFT = 0.2


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if infer_kind(df[c]) in ("number", "integer", "datetime")]
    frame = df[keep].copy()
    for column in frame.columns:
        if infer_kind(df[column]) == "datetime":
            frame[column] = epoch_seconds(df[column])
        else:
            frame[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    return frame


def _ks_tests(sample: pd.DataFrame, generated: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    s_num, g_num = _numeric_frame(sample), _numeric_frame(generated)
    for column in [c for c in s_num.columns if c in g_num.columns]:
        a, b = s_num[column].dropna(), g_num[column].dropna()
        if len(a) >= 2 and len(b) >= 2:
            result = stats.ks_2samp(a, b)
            out[column] = {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
    return out


def _chi2_tests(sample: pd.DataFrame, generated: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for column in sample.columns:
        kind = infer_kind(sample[column])
        if kind not in ("categorical", "bool"):
            continue
        if column not in generated.columns or infer_kind(generated[column]) != kind:
            continue
        cats = sorted(set(sample[column].astype(str).dropna()) | set(generated[column].astype(str).dropna()))
        s_counts = sample[column].astype(str).value_counts().reindex(cats, fill_value=0)
        g_counts = generated[column].astype(str).value_counts().reindex(cats, fill_value=0)
        expected_share = (s_counts / s_counts.sum()).to_numpy()
        expected = np.maximum(expected_share * g_counts.sum(), 0.5)
        expected = expected * (g_counts.sum() / expected.sum())  # keep sums equal
        result = stats.chisquare(g_counts.to_numpy(dtype=float), f_exp=expected)
        out[column] = {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
    return out


def _correlation_drift(sample: pd.DataFrame, generated: pd.DataFrame) -> float:
    columns = [c for c in _numeric_frame(sample).columns if c in _numeric_frame(generated).columns]
    if len(columns) < 2:
        return 0.0
    s_corr = _numeric_frame(sample)[columns].corr().to_numpy()
    g_corr = _numeric_frame(generated)[columns].corr().to_numpy()
    mask = ~(np.isnan(s_corr) | np.isnan(g_corr))
    if not mask.any():
        return 0.0
    return float(np.nanmax(np.where(mask, np.abs(s_corr - g_corr), np.nan)))


def _correlation_matrices(
    sample: pd.DataFrame | None, generated: pd.DataFrame
) -> dict[str, Any]:
    """Real and synthetic correlation matrices, plus their per-pair drift.

    The report's ``correlation_drift`` is the single worst pair, which is the
    right number for a Check and the wrong shape for a heatmap — you cannot see
    *which* relationships moved. This returns the matrices so the UI can, and
    keeps the scalar as ``max_drift`` so both views agree.
    """
    if sample is None:
        return {"columns": [], "sample": [], "generated": [], "drift": [], "max_drift": 0.0}
    shared = [c for c in _numeric_frame(sample).columns if c in _numeric_frame(generated).columns]
    if len(shared) < 2:
        return {"columns": shared, "sample": [], "generated": [], "drift": [], "max_drift": 0.0}
    s_corr = _numeric_frame(sample)[shared].corr()
    g_corr = _numeric_frame(generated)[shared].corr()
    drift = (s_corr - g_corr).abs()
    return {
        "columns": shared,
        "sample": [[_round(v) for v in row] for row in s_corr.to_numpy()],
        "generated": [[_round(v) for v in row] for row in g_corr.to_numpy()],
        "drift": [[_round(v) for v in row] for row in drift.to_numpy()],
        "max_drift": _round(drift.to_numpy().max()),
    }


def _round(value: Any) -> float | None:
    """JSON-safe float: NaN (an undefined correlation) becomes null, not NaN."""
    number = float(value)
    return None if math.isnan(number) else round(number, 4)


def _near_copies(
    sample: pd.DataFrame,
    generated: pd.DataFrame,
    threshold: float | None,
    max_scan: int = 200,
    max_reference: int = 200,
) -> dict[str, Any]:
    """Count generated rows with a sample row closer than ``threshold``.

    Chebyshev distance (max over shared non-text columns: range-normalized
    numerics; 0/1 for categories), so a near-copy is close on EVERY column —
    a genuine "this looks like that real row" signal that plain sampling of
    dense continuous data does not trip.

    ``threshold=None`` derives one from the reference-set density
    (0.05 x nearest-neighbour distance for ``d`` numeric columns), floored
    at exact-copy (0) for purely discrete data. Both scans are capped so the
    report stays cheap on large inputs.
    """
    kinds = {column: infer_kind(sample[column]) for column in sample.columns}
    usable = [
        column
        for column in generated.columns
        if column in kinds
        and kinds[column] != "text"
        and column != PROVENANCE_COLUMN
    ]
    ref = sample[usable].head(max_reference)
    n_numeric = sum(1 for column in usable if kinds[column] in ("number", "integer", "datetime"))
    if threshold is None:  # density-adaptive default
        threshold = 0.05 * (len(ref) ** (-1.0 / n_numeric)) if n_numeric and len(ref) else 0.0
    if not usable:
        return {"count": 0, "threshold": threshold, "examples": [], "scanned": 0}

    gen = generated[usable].head(max_scan)

    reference = np.full((len(ref), len(usable)), np.nan, dtype=float)
    generated_matrix = np.full((len(gen), len(usable)), np.nan, dtype=float)
    for position, column in enumerate(usable):
        kind = kinds[column]

        def numbers(series: pd.Series) -> np.ndarray:
            if kind == "datetime":
                return epoch_seconds(series)
            return pd.to_numeric(series, errors="coerce").astype(float).to_numpy()

        ref_values = numbers(ref[column])
        gen_values = numbers(gen[column])
        if kind in ("number", "integer", "datetime"):
            finite = ref_values[~np.isnan(ref_values)]
            span = (float(finite.max() - finite.min()) if len(finite) else 0.0) or 1.0
            reference[:, position] = ref_values / span
            generated_matrix[:, position] = gen_values / span
        else:
            categories = {value: index for index, value in enumerate(sorted(set(ref[column].astype(str))))}
            reference[:, position] = [categories.get(str(v), -1) for v in ref[column]]
            generated_matrix[:, position] = [categories.get(str(v), -2) for v in gen[column]]

    # per-column distance tensor (gen, ref, cols)
    both_nan = np.isnan(generated_matrix)[:, None, :] & np.isnan(reference)[None, :, :]
    one_nan = np.isnan(generated_matrix)[:, None, :] ^ np.isnan(reference)[None, :, :]
    with np.errstate(invalid="ignore"):
        delta = np.abs(generated_matrix[:, None, :] - reference[None, :, :])
    # categorical columns: any difference counts as full distance
    column_is_cat = np.array([kinds[column] in ("categorical", "bool") for column in usable])
    delta[:, :, column_is_cat] = np.where(delta[:, :, column_is_cat] == 0, 0.0, 1.0)
    delta = np.where(both_nan, 0.0, delta)
    delta = np.where(one_nan, 1.0, delta)
    delta = np.nan_to_num(delta, nan=1.0)

    # Chebyshev (max-column) distance: a near-copy must be close on EVERY
    # column (a matching category is required; every numeric within
    # threshold x its range). Mean distance would flag ordinary dense data.
    distance = delta.max(axis=2)  # (gen, ref)
    closest = distance.min(axis=1) if len(gen) and len(ref) else np.full(len(gen), np.inf)
    hits = closest <= threshold
    examples = [int(i) for i in np.nonzero(hits)[0][:5]]
    return {
        "count": int(hits.sum()),
        "threshold": threshold,
        "examples": examples,
        "scanned": int(len(gen)),
    }


def _balance_actual(
    generated: pd.DataFrame, requested: dict[str, dict[str, float]] | None
) -> dict[str, dict[str, dict[str, float]]]:
    if not requested:
        return {}
    out: dict[str, dict[str, dict[str, float]]] = {}
    total = max(len(generated), 1)
    for column, shares in requested.items():
        if column not in generated.columns:
            continue
        actual = generated[column].astype(str).value_counts(normalize=True)
        out[column] = {
            category: {"requested": share, "actual": float(actual.get(str(category), 0.0))}
            for category, share in shares.items()
        }
    return out


def build_fidelity_report(
    sample: pd.DataFrame,
    generated: pd.DataFrame,
    *,
    profile_derived: bool = False,
    dropped_rows: int = 0,
    requested_count: int | None = None,
    balance: dict[str, dict[str, float]] | None = None,
    near_copy_threshold: float | None = None,
) -> dict[str, Any]:
    """JSON-serializable Fidelity Report comparing generated vs sample rows."""
    sample = sample.drop(columns=[PROVENANCE_COLUMN], errors="ignore")
    generated = generated.drop(columns=[PROVENANCE_COLUMN], errors="ignore")
    duplicates = int(generated.duplicated().sum())
    ks = _ks_tests(sample, generated)
    chi2 = _chi2_tests(sample, generated)
    drift = _correlation_drift(sample, generated)
    near = _near_copies(sample, generated, near_copy_threshold)
    target_rows = requested_count if requested_count is not None else len(generated)
    report: dict[str, Any] = {
        "ks_tests": ks,
        "chi2_tests": chi2,
        "correlation_drift": round(drift, 4),
        "exact_duplicates": {"count": duplicates, "fraction": round(duplicates / max(len(generated), 1), 4)},
        "near_copies": near,
        "dropped_rows": int(dropped_rows),
        "dropped_ratio": round(dropped_rows / max(target_rows, 1), 4),
        "balance": _balance_actual(generated, balance),
        "provider_derived_profile": bool(profile_derived),
        "generated_rows": int(len(generated)),
    }
    report["warnings"] = _warning_codes(report)
    return report


def _warning_codes(report: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    if report["near_copies"]["count"] > 0:
        codes.append("near_copies")
    if report["dropped_ratio"] > HIGH_DROP_RATIO:
        codes.append("dropped_rows_high")
    if report["provider_derived_profile"]:
        codes.append("provider_derived_profile")
    low_ks = any(values["p_value"] < KS_DRIFT_P for values in report["ks_tests"].values())
    low_chi = any(values["p_value"] < KS_DRIFT_P for values in report["chi2_tests"].values())
    if low_ks or low_chi or report["correlation_drift"] > CORR_DRIFT:
        codes.append("fidelity_drift")
    return codes


_CHECK_MESSAGES = {
    "near_copies": lambda r: (
        f"{r['near_copies']['count']} generated row(s) are near-copies of real sample rows "
        f"(distance <= {r['near_copies']['threshold']}); treat this Version as sensitive"
    ),
    "dropped_rows_high": lambda r: (
        f"{r['dropped_rows']} Provider rows ({int(r['dropped_ratio'] * 100)}%) were dropped as invalid"
    ),
    "provider_derived_profile": lambda r: (
        "Column stats were drafted by a Provider (no real sample): fidelity is unverified"
    ),
    "fidelity_drift": lambda r: (
        f"Generated distributions drift from the sample (correlation delta {r['correlation_drift']})"
    ),
}


def raise_fidelity_checks(checks_store, version_id: str, report: dict[str, Any]) -> list[str]:
    """Register warning Checks for each risky finding (deduped per Version)."""
    raised: list[str] = []
    for code in report.get("warnings", []):
        if checks_store.has_check(code, "dataset_version", version_id):
            continue
        checks_store.register(
            kind=code,
            severity="warning",
            message=_CHECK_MESSAGES[code](report),
            subject_type="dataset_version",
            subject_id=version_id,
            details={
                key: report[key]
                for key in ("correlation_drift", "dropped_rows", "near_copies")
                if key in report
            },
        )
        raised.append(code)
    return raised
