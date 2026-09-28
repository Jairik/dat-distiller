"""Jev endpoints: draft a Jev Question with a Provider (suggestion only)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..jev import question_kind
from ..jev_suggest import DraftJevQuestionIn, draft_jev_question, question_payload
from ..providers import resolve_provider
from ..providers.base import (
    InvalidStructuredOutputError,
    ProviderNotConfiguredError,
    ProviderTimeoutError,
)

router = APIRouter(tags=["jev"])


@router.post("/jev/draft-question")
def draft_question(body: DraftJevQuestionIn, request: Request) -> dict[str, Any]:
    """A drafted Jev Question (Noul, Choice or Score) for the user to edit."""
    try:
        provider = resolve_provider(
            request.app.state.settings, provider=body.provider, model=body.model
        )
        question, warning = draft_jev_question(
            provider, body.description, body.context, body.sample_rows
        )
    except ProviderNotConfiguredError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ProviderTimeoutError as exc:
        raise HTTPException(504, str(exc)) from exc
    except InvalidStructuredOutputError as exc:
        raise HTTPException(502, f"Provider output was unusable: {exc.errors}") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"question": question_payload(question), "kind": question_kind(question), "warning": warning}
