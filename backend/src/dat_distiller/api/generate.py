"""Generation endpoints — Column Spec suggestions now; the run API lands in #22."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

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
