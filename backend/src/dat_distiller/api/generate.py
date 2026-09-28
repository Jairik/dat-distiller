"""Generation endpoints: Column Spec suggestions, preview, estimate, run."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from ..generate.run import GenerationRequest, estimate, resolve_run
from ..generate.suggest import SuggestColumnSpecsIn, suggest_column_specs
from ..profile import validate_specs
from ..providers import resolve_provider
from ..providers.base import (
    InvalidStructuredOutputError,
    ProviderNotConfiguredError,
    ProviderTimeoutError,
)

router = APIRouter(tags=["generate"])


@router.post("/generate/suggest-columns")
def suggest_columns(body: SuggestColumnSpecsIn, request: Request) -> dict[str, Any]:
    """Provider-proposed Column Specs from a description — a suggestion to edit."""
    try:
        provider = resolve_provider(
            request.app.state.settings, provider=body.provider, model=body.model
        )
        specs = suggest_column_specs(provider, body.description, body.column_names)
    except ProviderNotConfiguredError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ProviderTimeoutError as exc:
        raise HTTPException(504, str(exc)) from exc
    except InvalidStructuredOutputError as exc:
        raise HTTPException(502, f"Provider output was unusable: {exc.errors}") from exc
    return {"specs": [s.to_dict() for s in specs], "validation_errors": validate_specs(specs)}


class PreviewIn(GenerationRequest):
    preview_rows: int = Field(default=5, ge=1, le=20)


def _resolve_or_422(request: Request, body: GenerationRequest):
    try:
        return resolve_provider(request.app.state.settings, provider=body.provider, model=body.model)
    except ProviderNotConfiguredError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/generate/preview")
def preview(body: PreviewIn, request: Request) -> dict[str, Any]:
    """Generate a few rows synchronously — the 'does this look right?' check."""
    from ..generate.modes import generate_with_mode

    try:
        resolved = resolve_run(request.app.state.store, body)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    provider = (
        _resolve_or_422(request, body)
        if estimate(body, resolved.profile)["uses_provider"]
        else None
    )
    try:
        result = generate_with_mode(
            provider,
            body.mode,
            resolved.profile,
            body.description,
            body.preview_rows,
            sample_df=resolved.sample_df,
            seed=resolved.seed,
            balance=body.balance,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except InvalidStructuredOutputError as exc:
        raise HTTPException(502, f"Provider output was unusable: {exc.errors}") from exc
    calls = getattr(provider, "calls", None)
    return {
        "seed": resolved.seed,
        "rows": json.loads(result.rows.to_json(orient="records", date_format="iso")),
        "provider_calls": len(calls) if calls is not None else None,
    }


@router.post("/generate/estimate")
def estimate_endpoint(body: GenerationRequest, request: Request) -> dict[str, Any]:
    """Cost preview: expected rows and Provider calls."""
    try:
        resolved = resolve_run(request.app.state.store, body)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return estimate(body, resolved.profile)


@router.post("/generate/run", status_code=202)
def run(body: GenerationRequest, request: Request) -> dict[str, Any]:
    """Start the full run as a background job; progress over SSE, result is a
    new Dataset Version with synthetic Provenance and a Fidelity Report."""
    try:
        resolved = resolve_run(request.app.state.store, body)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if estimate(body, resolved.profile)["uses_provider"]:
        _resolve_or_422(request, body)  # fail fast on a missing key
    try:
        job = request.app.state.jobs.start(
            "generate", project_id=body.project_id, params=body.model_dump()
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return job.to_dict()
