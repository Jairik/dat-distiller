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
    synthesizes a **plausible** payload from the schema: an enum value, a number
    inside the declared range, a string, all chosen deterministically from the
    prompt so the same prompt always gives the same answer. Prompts are recorded
    for assertions.

    The values *vary between calls on purpose*. A fake that answered every prompt
    with the first enum value and a zero made every downstream stage degenerate:
    a seed set of fifty identical rows fits a copula with no variance, so hybrid
    generation from Column Specs produced a Dataset Version where **every column
    was constant** — a Review Queue with nothing to review, a classification
    leaderboard with one class, and a Fairness Report with nothing to measure.
    Anything the suite is supposed to exercise downstream of Generation needed
    data with variance in it. Deterministic, offline and varied: the three
    properties a fake has to have.
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
            else synthesize(schema, salt=prompt)
        )
        errors = validate_structured(payload, schema)
        if errors:
            raise InvalidStructuredOutputError("fake provider produced bad output", errors)
        return payload


def _draw(salt: str | None) -> float:
    """A stable 0..1 draw from a string, or 0.0 when there is nothing to salt."""
    if not salt:
        return 0.0
    import hashlib

    digest = hashlib.sha256(salt.encode()).digest()
    return int.from_bytes(digest[:8], "big") / (2**64)


def synthesize(schema: dict[str, Any], salt: str | None = None) -> Any:
    """A value conforming to a JSON Schema, chosen deterministically from ``salt``.

    With no ``salt`` this is the *minimal* conforming value — the first enum
    member, zero inside any declared range — which is what a unit test wants when
    it is asserting "the fake produced something valid". With a ``salt`` the value
    is picked from the same space but varies, so a run of calls yields data with
    variance rather than N copies of one row.
    """
    draw = _draw(salt)
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        options = schema["enum"]
        if not options:
            return None
        return options[0] if salt is None else options[int(draw * len(options)) % len(options)]
    if "allOf" in schema:
        merged = {}
        for part in schema["allOf"]:
            merged.update(part)
        return synthesize(merged, salt)
    if "examples" in schema and "x-range" not in schema:
        # `examples` is a *hint*, not a constraint, and it is where a column's
        # declared categories live. Unsalted it is the first one; salted it is a
        # choice from the set, which is what makes a fake's output vary in the
        # ways the column spec says it may. A numeric `x-range` is handled below,
        # where there is something to draw across rather than one midpoint.
        options = [v for v in schema["examples"] if v is not None]
        if not options:
            return None
        if salt is None or len(options) == 1:
            return options[0]
        return options[int(draw * len(options)) % len(options)]
    if "anyOf" in schema:
        return synthesize(schema["anyOf"][0], salt)
    kind = schema.get("type")
    if isinstance(kind, list):  # e.g. ["number", "null"]
        kind = next((k for k in kind if k != "null"), "null") if kind else "null"
    if kind == "object":
        required = schema.get("required")
        props = schema.get("properties", {})
        keys = [k for k in props if required is None or k in required] or list(props)
        return {key: synthesize(props[key], salt) for key in keys}
    if kind == "array":
        item = schema.get("items", {"type": "string"})
        # Honour a stated length: a Provider asked for fifty rows and given one
        # gets asked again with a byte-identical prompt, so it answers the same
        # way and the "batch" is fifty copies of a single row.
        try:
            wanted = int(schema.get("minItems", 1))
        except (TypeError, ValueError):
            wanted = 1
        wanted = max(1, min(wanted, 200))
        return [
            synthesize(item, None if salt is None else f"{salt}#{index}")
            for index in range(wanted)
        ]
    if kind == "integer" or kind == "number":
        # a declared `x-range` is the *hint* a column spec carries; `minimum` /
        # `maximum` are the enforced bounds. Prefer the hint when salted, so the
        # fake draws across the range a real Provider would be shown.
        hint = schema.get("x-range")
        if salt is not None and isinstance(hint, list) and len(hint) == 2:
            low, high = float(hint[0]), float(hint[1])
        else:
            low = schema.get("minimum", 0)
            high = schema.get("maximum", low + 9)
        if salt is None:
            value = 0
        else:
            value = low + draw * (high - low)
        value = max(value, low)
        value = min(value, high)
        return int(round(value)) if kind == "integer" else float(round(value, 6))
    if kind == "boolean":
        return draw > 0.5 if salt is not None else False
    if kind == "null":
        return None
    if kind == "string":
        return "s" * int(schema.get("minLength", 0) or 0) or "sample"
    return ""
