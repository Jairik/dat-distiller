"""Ask a Provider to suggest Column Specs from a dataset description.

Suggestion only — the user edits the result. Uses the Provider's structured
output, so the reply is schema-validated Column Specs by construction.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..profile import ColumnSpec
from ..providers.base import Provider, validate_structured

_SPEC_ITEM = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "type": {"type": "string", "enum": ["number", "integer", "categorical", "bool", "datetime", "text"]},
        "min": {"type": ["number", "null"]},
        "max": {"type": ["number", "null"]},
        "categories": {"type": ["array", "null"], "items": {"type": "string"}},
        "description": {"type": ["string", "null"]},
    },
    "required": ["name", "type"],
}

SUGGEST_COLUMN_SPECS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"columns": {"type": "array", "items": _SPEC_ITEM, "minItems": 1}},
    "required": ["columns"],
}


def build_suggestion_prompt(description: str, column_names: list[str] | None) -> str:
    hint = ""
    if column_names:
        hint = f"\nThe user already has these column names in mind: {', '.join(column_names)}."
    return (
        "You design dataset schemas. From the description below, propose a "
        "realistic set of columns for a machine-learning dataset. For each "
        "column give: name (snake_case), type (number, integer, categorical, "
        "bool, datetime or text), optional min/max for numeric columns, "
        "optional categories for categorical columns, and a one-line "
        f"description.{hint}\n\nDataset description:\n{description}"
    )


def suggest_column_specs(provider: Provider, description: str, column_names: list[str] | None = None) -> list[ColumnSpec]:
    prompt = build_suggestion_prompt(description, column_names)
    payload = provider.complete(prompt, SUGGEST_COLUMN_SPECS_SCHEMA)
    return [ColumnSpec.from_dict(item) for item in payload["columns"]]


class SuggestColumnSpecsIn(BaseModel):
    description: str = Field(min_length=3, max_length=4000)
    column_names: list[str] | None = None
    provider: str | None = None
    model: str | None = None
