"""One Generation run: preview, estimate, and the background job.

A run takes a description, a sample Dataset Version **or** Column Specs, a
mode, a row count, optional Balance Targets, an optional Provider override,
and a seed (None = derived from the request so re-runs are reproducible).

- ``preview`` generates a handful of rows synchronously (for the UI's
  "looks right?" moment).
- ``estimate`` returns expected rows and Provider calls before spending.
- The full run is a ``generate`` job producing a new Dataset Version whose
  rows carry **synthetic Provenance** (mode, Provider, model, seed) and whose
  ``meta`` holds the **Fidelity Report**; risky findings become warning
  Checks. Exceeding the Generation soft limit is also a warning Check —
  never an error.
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from ..checks import CheckStore
from ..jobs import JobContext
from ..pii import register_pii_checks, scan, summarize_findings
from ..profile import ColumnSpec, Profile, build_profile, profile_from_specs
from ..providers import resolve_provider
from ..store import DatasetStore
from .fidelity import build_fidelity_report, raise_fidelity_checks
from .llm import DEFAULT_BATCH_SIZE
from .modes import SEED_ROWS, TEXT_BATCH_SIZE, ModeResult, generate_with_mode

MAX_COUNT = 200_000
MODES = ("statistical", "llm", "hybrid")
SOFT_LIMIT_KIND = "generation_soft_limit"


class GenerationRequest(BaseModel):
    """Body shared by preview / estimate / run."""

    project_id: str
    description: str = Field(min_length=3, max_length=4000)
    mode: str = "hybrid"
    count: int = Field(default=100, ge=1, le=MAX_COUNT)
    specs: list[dict[str, Any]] | None = None
    sample_version_id: str | None = None
    balance: dict[str, dict[str, float]] | None = None
    provider: str | None = None
    model: str | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def _check_source(self) -> "GenerationRequest":
        if not self.specs and not self.sample_version_id:
            raise ValueError("generation needs Column Specs or a sample Dataset Version")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        return self


@dataclass
class ResolvedRun:
    request: GenerationRequest
    profile: Profile
    sample_df: pd.DataFrame | None
    seed: int
    warnings: list[str] = field(default_factory=list)


def effective_seed(request: GenerationRequest) -> int:
    """Explicit seed wins; otherwise a stable hash of the request, so an
    unseeded re-run of the same configuration reproduces the same rows."""
    if request.seed is not None:
        return request.seed
    canonical = json.dumps(request.model_dump(exclude={"seed"}), sort_keys=True, default=str)
    return zlib.crc32(canonical.encode()) & 0x7FFFFFFF


def resolve_run(store: DatasetStore, request: GenerationRequest) -> ResolvedRun:
    """Profile + frame to fit on + effective seed."""
    sample_df: pd.DataFrame | None = None
    if request.sample_version_id:
        sample_df = store.load_dataframe(request.sample_version_id)
        profile = build_profile(sample_df)
    else:
        profile = profile_from_specs([ColumnSpec.from_dict(s) for s in request.specs or []])
    return ResolvedRun(request=request, profile=profile, sample_df=sample_df, seed=effective_seed(request))


def estimate(request: GenerationRequest, profile: Profile | None = None) -> dict[str, Any]:
    """Expected rows and Provider calls before the real run.

    ``profile`` (when known) lets `hybrid` count only the free-text fill:
    hybrid over a sample without text columns is pure statistics and needs
    no Provider at all.
    """
    specs_only = not request.sample_version_id
    calls = 0
    if request.mode == "llm":
        calls = math.ceil(request.count / DEFAULT_BATCH_SIZE)
    elif request.mode == "hybrid":
        if specs_only:
            may_need_text = True
        elif profile is not None:
            may_need_text = any(column.kind == "text" for column in profile.columns)
        else:
            may_need_text = True  # conservative without a Profile
        if may_need_text:
            calls += math.ceil(request.count / TEXT_BATCH_SIZE)
    if specs_only and request.mode in ("statistical", "hybrid"):
        calls += math.ceil(SEED_ROWS / DEFAULT_BATCH_SIZE)  # seed set from the Provider
    return {
        "rows": request.count,
        "estimated_provider_calls": calls,
        "uses_provider": calls > 0,
        "seed_set": specs_only and request.mode in ("statistical", "hybrid"),
    }


def _counting_provider(provider: Any, on_call: Any) -> Any:
    """Thin proxy that reports Provider calls for progress and checkpoints."""

    class _Proxy:
        id = getattr(provider, "id", "provider")
        model = getattr(provider, "model", None)

        def complete(self, prompt: str, schema: dict) -> dict:
            result = provider.complete(prompt, schema)
            on_call()
            return result

        def __getattr__(self, name: str) -> Any:  # passthrough for anything else
            return getattr(provider, name)

    return _Proxy()


def run_generation(
    *,
    resolved: ResolvedRun,
    provider: Any,
    store: DatasetStore,
    checks: CheckStore,
    settings: Any,
    ctx: JobContext | None = None,
) -> dict[str, Any]:
    """The full run: generate rows, file them as a Dataset Version, judge
    fidelity. Used by the ``generate`` job (ctx) and by tests directly."""
    request = resolved.request
    calls = {"n": 0}
    counted = _counting_provider(provider, lambda: calls.update(n=calls["n"] + 1))

    def progress(done_rows: int) -> None:
        if ctx is not None:
            ctx.progress(min(done_rows, request.count), request.count, provider_calls=calls["n"])

    progress(0)
    mode_result: ModeResult = generate_with_mode(
        counted,
        request.mode,
        resolved.profile,
        request.description,
        request.count,
        sample_df=resolved.sample_df,
        seed=resolved.seed,
        balance=request.balance,
    )
    if ctx is not None and ctx.cancelled:
        raise RuntimeError("cancelled")
    progress(request.count)

    limits = settings.load().get("soft_limits", {})
    soft_limit = int(limits.get("generation_rows", 5000))
    if request.count > soft_limit:
        checks.register(
            kind=SOFT_LIMIT_KIND,
            severity="warning",
            message=(
                f"Generated {request.count} rows, above the soft limit of {soft_limit}"
            ),
            subject_type="project",
            subject_id=request.project_id,
            details={"count": request.count, "soft_limit": soft_limit},
        )

    rows = mode_result.rows
    provenance = [
        {
            "row_origin": "synthetic",
            "mode": request.mode,
            "provider": getattr(counted, "id", "provider"),
            "model": getattr(counted, "model", None),
            "seed": resolved.seed,
        }
        for _ in range(len(rows))
    ]
    if resolved.sample_df is not None:
        fidelity = build_fidelity_report(
            resolved.sample_df,
            rows,
            profile_derived=mode_result.profile.provider_derived,
            dropped_rows=mode_result.dropped,
            requested_count=request.count,
            balance=request.balance,
        )
    else:  # spec-only run: nothing real to compare against
        fidelity = {
            "ks_tests": {},
            "chi2_tests": {},
            "correlation_drift": 0.0,
            "exact_duplicates": {"count": int(rows.duplicated().sum()), "fraction": 0.0},
            "near_copies": {"count": 0, "threshold": None, "examples": [], "scanned": 0},
            "dropped_rows": mode_result.dropped,
            "dropped_ratio": round(mode_result.dropped / max(request.count, 1), 4),
            "balance": {},
            "provider_derived_profile": True,
            "generated_rows": int(len(rows)),
            "warnings": ["provider_derived_profile"],
        }

    parent = request.sample_version_id
    version = store.create_version(
        request.project_id,
        rows,
        parent_id=parent,
        origin="generated",
        provenance=provenance,
        seed=resolved.seed,
        meta={
            "fidelity": json.loads(json.dumps(fidelity, default=str)),
            "generation": {
                "mode": request.mode,
                "description": request.description,
                "requested_rows": request.count,
                "provider_calls": calls["n"],
                "balance": request.balance,
            },
        },
    )
    raised = raise_fidelity_checks(checks, version.id, fidelity)
    # A Provider can invent a realistic-looking email or phone number. Scan the
    # generated rows before the Version is handed to the user, so the Check is
    # waiting at the end of the step rather than discovered later.
    findings = scan(rows)
    pii_checks = register_pii_checks(checks, version.id, findings)
    progress(request.count)
    return {
        "version_id": version.id,
        "rows": int(len(rows)),
        "seed": resolved.seed,
        "provider_calls": calls["n"],
        "dropped_rows": mode_result.dropped,
        "fidelity_warnings": fidelity["warnings"],
        "checks_raised": raised + [check.id for check in pii_checks],
        "pii": summarize_findings(findings),
    }


def register_generation_job(manager: Any, app: Any) -> None:
    """Wire the ``generate`` job type to the app's stores."""

    def runner(ctx: JobContext) -> dict[str, Any]:
        request = GenerationRequest.model_validate(ctx.params)
        resolved = resolve_run(app.state.store, request)
        if estimate(request, resolved.profile)["uses_provider"]:
            provider = resolve_provider(
                app.state.settings, provider=request.provider, model=request.model
            )
        else:
            provider = None
        return run_generation(
            resolved=resolved,
            provider=provider,
            store=app.state.store,
            checks=app.state.checks,
            settings=app.state.settings,
            ctx=ctx,
        )

    manager.register("generate", runner)
