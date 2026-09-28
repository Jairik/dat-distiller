"""Providers: text in, validated structured output out. Never agentic.

A Provider turns a prompt plus a JSON Schema into a structured payload that
has been validated against that schema. Implementations here are the
deterministic :class:`FakeProvider` (tests, demos) and
:class:`OpenRouterProvider` (HTTP). CLI Providers (#14) live alongside.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

import jsonschema


class ProviderError(RuntimeError):
    """Base error for everything a Provider can do wrong."""


class ProviderNotConfiguredError(ProviderError):
    """No API key / CLI not installed / anything missing to run the Provider."""


class ProviderTimeoutError(ProviderError):
    """The Provider took too long."""


class InvalidStructuredOutputError(ProviderError):
    """The Provider's reply did not match the requested JSON Schema."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


class Provider(Protocol):
    id: str

    def complete(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return a payload validated against ``schema``.

        Raises ProviderNotConfiguredError / ProviderTimeoutError /
        InvalidStructuredOutputError.
        """
        ...


def validate_structured(payload: Any, schema: dict[str, Any]) -> list[str]:
    """Return validation errors for ``payload`` (empty when it conforms)."""
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'.'.join(str(p) for p in error.absolute_path) or '<root>'}: {error.message}"
        for error in validator.iter_errors(payload)
    ]


def parse_json_reply(text: str) -> Any:
    """Parse a Provider reply that should be a JSON document.

    Tolerant of surrounding prose or ```json fences (real models do this), but
    never guesses across two separate JSON documents.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass
    brace = re.search(r"(\{.*\}|\[.*\])", stripped, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(1))
        except json.JSONDecodeError:
            pass
    raise InvalidStructuredOutputError("Provider reply was not JSON")


class FakeProvider:
    """Deterministic Provider for tests and the Playwright e2e suite.

    Given a `response_for(prompt, schema)` callable it uses that; otherwise it
    synthesizes a minimal payload from the schema (first enum value, 0/""/[]
    per type). Prompts are recorded for assertions.
    """

    id = "fake"

    def __init__(self, response_for=None, model: str = "fake-model") -> None:
        self.model = model
        self._response_for = response_for
        self.calls: list[dict[str, Any]] = []

    def complete(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"prompt": prompt, "schema": schema})
        payload = (
            self._response_for(prompt, schema)
            if self._response_for
            else synthesize(schema)
        )
        errors = validate_structured(payload, schema)
        if errors:
            raise InvalidStructuredOutputError("fake provider produced bad output", errors)
        return payload


def synthesize(schema: dict[str, Any]) -> Any:
    """Best-effort minimal value conforming to a JSON Schema."""
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    if "allOf" in schema:
        merged = {}
        for part in schema["allOf"]:
            merged.update(part)
        return synthesize(merged)
    if "examples" in schema:
        return schema["examples"][0]
    if "anyOf" in schema:
        return synthesize(schema["anyOf"][0])
    kind = schema.get("type")
    if isinstance(kind, list):  # e.g. ["number", "null"]
        kind = next((k for k in kind if k != "null"), "null") if kind else "null"
    if kind == "object":
        required = schema.get("required")
        props = schema.get("properties", {})
        keys = [k for k in props if required is None or k in required] or list(props)
        return {key: synthesize(props[key]) for key in keys}
    if kind == "array":
        return [synthesize(schema.get("items", {"type": "string"}))]
    if kind == "integer" or kind == "number":
        value = 0
        if "minimum" in schema and value < schema["minimum"]:
            value = schema["minimum"]
        if "maximum" in schema and value > schema["maximum"]:
            value = schema["maximum"]
        return int(value) if kind == "integer" else float(value)
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    if kind == "string":
        return "s" * int(schema.get("minLength", 0) or 0) or "sample"
    return ""
