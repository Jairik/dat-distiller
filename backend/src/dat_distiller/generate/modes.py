"""Generation Modes: statistical, llm, and the default hybrid.

- ``statistical``: the Gaussian copula samples the structured columns.
- ``llm``: the Provider writes whole rows (see ``llm.py``).
- ``hybrid`` (default): the copula produces the structured columns, then the
  Provider fills free-text columns **conditioned on each row's structured
  values**.

With no sample to learn a Profile from, statistical/hybrid first have the
Provider generate a **seed set** from the Column Specs, fit the copula on
it, and mark the Profile ``provider_derived`` so the Fidelity Report warns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..profile import ColumnSpec, Profile, build_profile, profile_from_specs
from ..providers.base import InvalidStructuredOutputError, Provider, ProviderError
from .copula import GaussianCopula
from .llm import build_prompt, coerce_and_validate, generate_rows

SEED_ROWS = 50
TEXT_BATCH_SIZE = 10
TEXT_RETRIES = 2
STATISTICAL_TEXT_LIMIT = 500  # rows of text after which statistical mode gives up politely


@dataclass
class ModeResult:
    rows: pd.DataFrame
    profile: Profile
    dropped: int = 0
    provider_text_rows: int = 0


def ensure_fittable_profile(
    provider: Provider,
    profile: Profile,
    description: str,
    sample_df: pd.DataFrame | None,
) -> tuple[Profile, pd.DataFrame]:
    """What statistical/hybrid fits its copula on.

    With a real sample: that sample. With declared-only Column Specs: a
    Provider-generated **seed set**, with the Profile marked
    ``provider_derived`` so the Fidelity Report can warn.
    """
    if sample_df is not None and len(sample_df):
        return profile, sample_df
    if profile.source != "specs":
        raise ValueError("statistical generation needs a sample or Column Specs")
    seed = generate_rows(provider, profile, description, SEED_ROWS, batch_size=SEED_ROWS)
    return build_profile(seed.rows, provider_derived=True), seed.rows


def _text_schema(text_columns: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {c: {"type": ["string", "null"]} for c in text_columns},
                    "required": text_columns,
                },
            }
        },
        "required": ["rows"],
    }


def fill_text_columns(
    provider: Provider,
    frame: pd.DataFrame,
    text_columns: list[str],
    description: str,
    *,
    batch_size: int = TEXT_BATCH_SIZE,
    retries: int = TEXT_RETRIES,
) -> tuple[pd.DataFrame, int]:
    """Ask the Provider for free-text values conditioned on each structured row."""
    frame = frame.copy()
    for column in text_columns:
        frame[column] = None
    missing = len(frame)
    if missing == 0:
        return frame, 0
    schema = _text_schema(text_columns)
    for start in range(0, missing, batch_size):
        chunk = frame.iloc[start : start + batch_size]
        rows_hint = chunk.dropna(axis=1, how="all").to_dict(orient="records")
        prompt = (
            f"{description}\n\nFor EACH of the {len(chunk)} dataset rows below, write a "
            f"plausible value for the free-text column(s) {text_columns}, consistent "
            "with that row's other values. Return the same number of rows, in order: "
            '{"rows": [{...}]}.\n\nRows:\n'
            + json.dumps(rows_hint, ensure_ascii=False, default=str)
        )
        filled: list[dict[str, Any]] | None = None
        feedback = ""
        for _attempt in range(retries + 1):
            try:
                payload = provider.complete(prompt + feedback, schema)
            except (InvalidStructuredOutputError, ProviderError):
                continue
            got = payload.get("rows", [])
            if len(got) != len(chunk):
                feedback = (
                    f"\n\nSend exactly {len(chunk)} rows (got {len(got)}), in the same order."
                )
                continue
            filled = got
            break
        if filled is None:
            continue  # these rows keep None text; counted via dropped
            # note: `continue` moves to the next chunk; unfilled stay None
        for offset, values in enumerate(filled):
            for column in text_columns:
                value = values.get(column)
                frame.iloc[start + offset, frame.columns.get_loc(column)] = (
                    str(value) if value is not None else None
                )
    dropped_frame = frame[text_columns]
    dropped = int(dropped_frame.isna().all(axis=1).sum())
    return frame, dropped


def generate_with_mode(
    provider: Provider,
    mode: str,
    profile: Profile,
    description: str,
    count: int,
    *,
    sample_df: pd.DataFrame | None = None,
    seed: int = 0,
    balance: dict[str, dict[str, float]] | None = None,
) -> ModeResult:
    """Run Generation in the given mode; returns rows plus the fitted Profile."""
    if mode not in ("statistical", "llm", "hybrid"):
        raise ValueError(f"unknown Generation Mode {mode!r}")
    sample_rows = (
        sample_df.head(3).to_dict(orient="records") if sample_df is not None else None
    )
    if mode == "llm":
        result = generate_rows(
            provider, profile, description, count, sample_rows=sample_rows, balance=balance
        )
        return ModeResult(rows=result.rows, profile=profile, dropped=result.dropped)

    fittable, fit_frame = ensure_fittable_profile(provider, profile, description, sample_df)
    copula = GaussianCopula.fit(fit_frame)
    structured = copula.sample(count, seed=seed, balance=balance)

    text_columns = [c.name for c in profile.columns if c.kind == "text"]
    if mode == "statistical":
        for column in text_columns:
            structured[column] = None
        dropped = count if text_columns else 0
        return ModeResult(rows=structured, profile=fittable, dropped=dropped)

    # hybrid: copula structured columns + Provider-written free text
    if not text_columns:
        return ModeResult(rows=structured, profile=fittable, dropped=0)
    filled, dropped = fill_text_columns(provider, structured, text_columns, description)
    ordered = [c.name for c in profile.columns if c.name in filled.columns]
    extra = [c for c in filled.columns if c not in ordered]
    return ModeResult(
        rows=filled[ordered + extra],
        profile=fittable,
        dropped=dropped,
        provider_text_rows=count - dropped,
    )
