"""Per-row Provenance: where a row and each of its labels came from.

Provenance travels inside the Parquet file itself, in the reserved
``__provenance__`` column: one JSON object per row, e.g.::

    {"row_origin": "synthetic",
     "label_origins": {"spam": {"origin": "jev", "confidence": 0.91}},
     "provider": "openrouter", "model": "x/y", "mode": "hybrid", "seed": 7}

``row_origin`` is where the row itself came from (``uploaded`` or
``synthetic``); ``label_origins`` records per-Label-Column origins
(``jev`` with confidence, or ``human_reviewed``). The ``provider``,
``model`` and ``mode`` fields are set on synthetic rows; ``seed`` on rows
produced by Generation. All optional.

The per-version *breakdown* assigns each row to exactly one bucket for the
history panel's stacked bar, most-informed-origin wins:
``human_reviewed`` > ``jev`` > ``row_origin``.
"""

from __future__ import annotations

import json
from typing import Any

PROVENANCE_COLUMN = "__provenance__"

BREAKDOWN_BUCKETS = ("uploaded", "synthetic", "jev", "human_reviewed")


class Provenance:
    """Helpers for building and reading the ``__provenance__`` column."""

    @staticmethod
    def row(
        row_origin: str,
        *,
        label_origins: dict[str, dict[str, Any]] | None = None,
        provider: str | None = None,
        model: str | None = None,
        mode: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        prov: dict[str, Any] = {"row_origin": row_origin}
        if label_origins:
            prov["label_origins"] = label_origins
        if provider is not None:
            prov["provider"] = provider
        if model is not None:
            prov["model"] = model
        if mode is not None:
            prov["mode"] = mode
        if seed is not None:
            prov["seed"] = seed
        return prov

    @staticmethod
    def parse(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if value is None or (isinstance(value, float) and value != value):
            return {"row_origin": "uploaded"}
        return json.loads(value)


def provenance_breakdown(values: list[Any] | None) -> dict[str, int]:
    """Count rows per origin bucket. Empty list still yields zeroed buckets."""
    counts = dict.fromkeys(BREAKDOWN_BUCKETS, 0)
    for raw in values or []:
        prov = Provenance.parse(raw)
        label_origins = prov.get("label_origins", {})
        if any(o.get("origin") == "human_reviewed" for o in label_origins.values()):
            bucket = "human_reviewed"
        elif any(o.get("origin") == "jev" for o in label_origins.values()):
            bucket = "jev"
        else:
            bucket = prov.get("row_origin", "uploaded")
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts
