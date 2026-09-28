"""`llm` Generation Mode: the Provider writes rows in validated batches.

- Rows come back batched; every row is coerced and validated against the
  Profile / Column Specs (types, numeric ranges, allowed categories,
  booleans, dates).
- A batch containing anything invalid is retried up to ``retries`` times
  with the validation errors fed back into the prompt; rows still invalid
  afterwards are dropped and counted (the count feeds the Fidelity Report).
- Balance Targets nudge the prompt ("aim for ~50% churned=true").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..profile import Profile
from ..providers.base import (
    InvalidStructuredOutputError,
    Provider,
    ProviderError,
    ProviderTimeoutError,
)

DEFAULT_BATCH_SIZE = 25
DEFAULT_RETRIES = 2


@dataclass
class LLMSynthesisResult:
    rows: pd.DataFrame
    dropped: int = 0
    attempts: int = 0  # Provider calls made
    failures: list[str] = field(default_factory=list)


def rows_schema(profile: Profile) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for column in profile.columns:
        props[column.name] = _column_schema(column.kind)
    return {
        "type": "object",
        "properties": {"rows": {"type": "array", "items": {"type": "object", "properties": props, "required": list(props)}}},
        "required": ["rows"],
    }


def _column_schema(kind: str) -> dict[str, Any]:
    # Keep the JSON Schema permissive: coercion + validation happen against
    # the Profile (ranges/categories), where errors can be reported back.
    return {
        "number": {"type": ["number", "string", "null"]},
        "integer": {"type": ["integer", "string", "number", "null"]},
        "categorical": {"type": ["string", "number", "null"]},
        "bool": {"type": ["boolean", "string", "integer", "null"]},
        "datetime": {"type": ["string", "null"]},
        "text": {"type": ["string", "null"]},
    }[kind]


def build_prompt(description: str, profile: Profile, count: int, sample_rows: list[dict] | None) -> str:
    parts = [
        "You synthesize realistic dataset rows that match the description and column spec below.",
        f"Target: about {count} rows in this batch.",
        f"Dataset description: {description}",
        "Columns (JSON): " + json.dumps(
            [
                {
                    k: v
                    for k, v in c.to_dict().items()
                    if v is not None and k in ("name", "kind", "min", "max", "categories", "examples", "description")
                }
                for c in profile.columns
            ],
            ensure_ascii=False,
        ),
    ]
    if sample_rows:
        parts.append("Example real rows for style and plausible values: " + json.dumps(sample_rows, ensure_ascii=False))
    if profile.provider_derived:
        parts.append("(These column stats were drafted by a language model, not measured; invent plausible values.)")
    return "\n\n".join(parts)


def balance_nudge(balance: dict[str, dict[str, float]] | None) -> str | None:
    if not balance:
        return None
    lines = ["Balance Targets: aim for these proportions in the batch"]
    for column, shares in balance.items():
        for value, share in shares.items():
            lines.append(f"- {column}={value}: about {round(share * 100)}%")
    return "\n".join(lines)


def coerce_and_validate(obj: Any, profile: Profile) -> tuple[dict[str, Any] | None, list[str]]:
    """One row: coerced in place against the Profile; errors listed."""
    if not isinstance(obj, dict):
        return None, ["row is not an object"]
    row: dict[str, Any] = {}
    errors: list[str] = []
    for column in profile.columns:
        raw = obj.get(column.name)
        try:
            row[column.name] = _coerce_value(column.name, column.kind, raw, column)
        except ValueError as exc:
            errors.append(str(exc))
    return (None if errors else row), errors


def _coerce_value(name: str, kind: str, raw: Any, column: Any) -> Any:
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return None
    if kind in ("number", "integer"):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{name}: {raw!r} is not numeric")
        if kind == "integer" and value != int(value):
            raise ValueError(f"{name}: {raw!r} is not an integer")
        if column.min is not None and value < float(column.min):
            raise ValueError(f"{name}: {value} below minimum {column.min}")
        if column.max is not None and value > float(column.max):
            raise ValueError(f"{name}: {value} above maximum {column.max}")
        return int(value) if kind == "integer" else value
    if kind == "categorical":
        text = str(raw)
        if column.categories and text not in column.categories:
            raise ValueError(f"{name}: {text!r} not one of {column.categories}")
        return text
    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in ("true", "yes", "1"):
            return True
        if text in ("false", "no", "0"):
            return False
        raise ValueError(f"{name}: {raw!r} is not boolean")
    if kind == "datetime":
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"{name}: {raw!r} is not a date/time")
        return parsed.isoformat()
    return str(raw)  # text


def generate_rows(
    provider: Provider,
    profile: Profile,
    description: str,
    count: int,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    retries: int = DEFAULT_RETRIES,
    sample_rows: list[dict[str, Any]] | None = None,
    balance: dict[str, dict[str, float]] | None = None,
) -> LLMSynthesisResult:
    """Generate ``count`` rows via the Provider, batch by batch.

    Terminates on: enough rows, a hard Provider failure, or a global request
    budget (so a Provider that only emits garbage cannot spin forever).
    """
    schema = rows_schema(profile)
    base_prompt = build_prompt(description, profile, batch_size, sample_rows)
    nudge = balance_nudge(balance)
    if nudge:
        base_prompt += "\n\n" + nudge

    rows: list[dict[str, Any]] = []
    dropped = 0
    attempts = 0
    failures: list[str] = []
    batch_no = 0
    no_progress = 0
    max_requests = ((count + batch_size - 1) // batch_size) * (retries + 1) + 8

    while len(rows) < count and batch_no < max_requests:
        batch_no += 1
        target = min(batch_size, count - len(rows))
        prompt = f"{base_prompt}\n\nBatch {batch_no}: produce exactly {target} rows."
        accepted, invalid, batch_attempts, hard = _run_batch(
            provider, prompt, schema, profile, retries
        )
        attempts += batch_attempts
        before = len(rows)
        rows.extend(accepted)
        if invalid:
            dropped += len(invalid)
            for bad in invalid[:5]:
                failures.append("; ".join(bad["errors"]))
        if hard:  # Provider itself is down/timing out — stop, don't hammer
            failures.append("stopped early: Provider call failed repeatedly")
            break
        no_progress = no_progress + 1 if len(rows) == before else 0
        if no_progress >= (retries + 1) * 2:
            failures.append("stopped early: Provider kept producing invalid rows")
            break

    return LLMSynthesisResult(
        rows=pd.DataFrame(rows[:count], columns=[c.name for c in profile.columns]),
        dropped=dropped,
        attempts=attempts,
        failures=failures,
    )


def _run_batch(
    provider: Provider,
    prompt: str,
    schema: dict[str, Any],
    profile: Profile,
    retries: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, bool]:
    """One batch with error-feedback retries.

    Returns (accepted rows, invalid rows, provider attempts, hard_failure).
    A hard failure (timeout/outage) gives up the batch immediately.
    """
    accepted: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    feedback = ""
    attempts = 0
    for _attempt in range(retries + 1):
        try:
            payload = provider.complete(prompt + feedback, schema)
            attempts += 1
        except InvalidStructuredOutputError as exc:
            attempts += 1
            feedback = "\n\nYour previous reply was unusable:\n" + "\n".join(exc.errors[:10])
            continue
        except ProviderError:
            return [], [], attempts + 1, True  # Provider itself failed

        accepted, invalid, error_text = [], [], []
        for obj in payload.get("rows", []):
            row, errors = coerce_and_validate(obj, profile)
            if errors:
                invalid.append({"row": obj, "errors": errors})
                error_text.extend(errors)
            else:
                accepted.append(row)
        if not invalid:
            return accepted, invalid, attempts, False
        feedback = "\n\nFix these problems and resend the full batch:\n" + "\n".join(error_text[:10])
    return accepted, invalid, attempts, False
