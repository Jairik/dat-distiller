"""Fidelity Report endpoints: the stored report, plus the series it charts.

``GET /dataset-versions/{id}/fidelity`` returns the report Generation already
stored in the version's ``meta``. The UI could chart distributions from the
preview endpoint, but a 200-row page is a sample of a sample — this endpoint
sends real binned distributions and the actual correlation matrices instead, so
a chart is never quietly wrong because it only saw part of the data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from ..generate.fidelity import _correlation_matrices
from ..store.provenance import PROVENANCE_COLUMN
from .projects import _require_version

router = APIRouter(tags=["fidelity"])

#: Bins per numeric column. Enough to see a shape, few enough to stay small.
BINS = 20
#: Cap on categories sent per column; the tail is folded into "other".
MAX_CATEGORIES = 12


@router.get("/dataset-versions/{version_id}/fidelity")
def fidelity_report(
    version_id: str,
    request: Request,
    bins: int = Query(BINS, ge=4, le=60),
) -> dict[str, Any]:
    """The Fidelity Report for a generated Dataset Version, chart-ready.

    404 when the version was not produced by Generation — there is no report to
    show, and an empty chart would read as "everything matched".
    """
    version = _require_version(request, version_id)
    report = (version.meta or {}).get("fidelity")
    if not isinstance(report, dict):
        raise HTTPException(
            404, "this Dataset Version has no Fidelity Report (it was not generated)"
        )
    store = request.app.state.store
    generated = store.load_dataframe(version.id)
    parent_id = report.get("sample_version_id") or version.parent_id
    sample = store.load_dataframe(parent_id) if parent_id else None

    return {
        "version_id": version.id,
        "sample_version_id": parent_id,
        "report": report,
        "distributions": distributions(sample, generated, bins=bins),
        "correlation": _correlation_matrices(sample, generated),
    }


def _clean(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None:
        return None
    return frame.drop(columns=[PROVENANCE_COLUMN], errors="ignore")


def distributions(
    sample: pd.DataFrame | None,
    generated: pd.DataFrame,
    *,
    bins: int = BINS,
) -> list[dict[str, Any]]:
    """Per column, the real-versus-synthetic distribution to chart.

    Numeric and datetime columns get a shared-bin histogram; categorical and
    bool columns get category shares. A column only in one side is still
    listed, with an empty series on the other — that asymmetry is itself the
    finding.
    """
    sample = _clean(sample) if sample is not None else None
    generated = _clean(generated)
    out: list[dict[str, Any]] = []
    columns = list(generated.columns)
    if sample is not None:
        columns += [c for c in sample.columns if c not in generated.columns]
    for column in columns:
        in_gen = column in generated.columns
        in_sample = sample is not None and column in sample.columns
        g_values = generated[column] if in_gen else None
        s_values = sample[column] if in_sample and sample is not None else None
        kind = _kind(g_values if g_values is not None else s_values)
        if kind in ("number", "integer", "datetime"):
            out.append(
                {
                    "column": column,
                    "kind": kind,
                    "chart": "histogram",
                    **_histogram_pair(s_values, g_values, bins),
                    "only_in": _only_in(in_sample, in_gen),
                }
            )
        else:
            out.append(
                {
                    "column": column,
                    "kind": kind,
                    "chart": "bar",
                    **_category_pair(s_values, g_values),
                    "only_in": _only_in(in_sample, in_gen),
                }
            )
    return out


def _only_in(in_sample: bool, in_generated: bool) -> str | None:
    if in_sample and not in_generated:
        return "sample"
    if in_generated and not in_sample:
        return "generated"
    return None


def _kind(series: pd.Series | None) -> str:

    if series is None:
        return "text"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    non_null = series.dropna()
    if len(non_null) == 0:
        return "text"
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_bool_dtype(non_null):
        return "bool"
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        return "number"
    try:
        if pd.to_numeric(non_null, errors="coerce").notna().all():
            return "number"
    except (TypeError, ValueError):
        pass
    unique = non_null.nunique()
    return "categorical" if unique <= max(len(non_null) * 0.5, 1) else "text"


def _histogram_pair(
    sample: pd.Series | None, generated: pd.Series | None, bins: int
) -> dict[str, Any]:
    """One set of edges covering both series, so the bars line up."""
    finite = []
    for series in (sample, generated):
        if series is not None and len(series.dropna()):
            finite.append(pd.to_numeric(series, errors="coerce").dropna().astype(float))
    if not finite:
        return {"edges": [], "sample": [], "generated": []}
    low = min(float(f.min()) for f in finite)
    high = max(float(f.max()) for f in finite)
    if low == high:
        high = low + 1.0

    def counts(series: pd.Series | None) -> list[int]:
        if series is None:
            return []
        values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
        if len(values) == 0:
            return []
        hist, _ = np.histogram(values, bins=bins, range=(low, high))
        return [int(c) for c in hist]

    # edges from any one series: the range is fixed above, so all sides share them
    _, edges = np.histogram(finite[0], bins=bins, range=(low, high))
    return {
        "edges": [float(e) for e in edges],
        "sample": counts(sample),
        "generated": counts(generated),
    }


def _category_pair(
    sample: pd.Series | None, generated: pd.Series | None
) -> dict[str, Any]:
    """Category shares, with the long tail folded into `other`."""
    shares: dict[str, dict[str, float]] = {}
    for label, series in (("sample", sample), ("generated", generated)):
        if series is None or len(series.dropna()) == 0:
            continue
        counts = series.astype(str).value_counts(normalize=True)
        top = list(counts.index[:MAX_CATEGORIES])
        entry = {str(k): float(counts[k]) for k in top}
        tail = float(counts.iloc[MAX_CATEGORIES:].sum())
        if tail > 0:
            entry["other"] = round(tail, 4)
        shares[label] = entry
    categories = sorted(set(shares.get("sample", {})) | set(shares.get("generated", {})))
    return {
        "categories": categories,
        "sample": [shares.get("sample", {}).get(c, 0.0) for c in categories],
        "generated": [shares.get("generated", {}).get(c, 0.0) for c in categories],
    }


__all__ = ["BINS", "MAX_CATEGORIES", "distributions", "router"]
