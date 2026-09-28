"""LLM Generation mode: batches, coercion, error-feedback retry, drops."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dat_distiller.generate.llm import coerce_and_validate, generate_rows, rows_schema
from dat_distiller.profile import Profile, build_profile
from dat_distiller.providers import FakeProvider
from dat_distiller.providers.base import ProviderTimeoutError


@pytest.fixture
def profile() -> Profile:
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 90, 100),
            "plan": rng.choice(["basic", "pro"], 100),
            "active": rng.random(100) < 0.5,
        }
    )
    return build_profile(df)


def valid_row(i: int) -> dict:
    return {"age": 20 + (i % 60), "plan": ["basic", "pro"][i % 2], "active": i % 2 == 0}


def queue_provider(responses):
    """FakeProvider whose replies come from a queue (last one repeats)."""
    remaining = {"n": 0, "responses": list(responses)}

    def responder(prompt, schema):
        i = min(remaining["n"], len(remaining["responses"]) - 1)
        remaining["n"] += 1
        reply = remaining["responses"][i]
        if isinstance(reply, Exception):
            raise reply
        return reply

    fake = FakeProvider(response_for=responder)
    fake.reply_index = remaining
    return fake


def test_valid_batch_is_accepted_as_is(profile) -> None:
    provider = queue_provider([{"rows": [valid_row(i) for i in range(10)]}])
    result = generate_rows(provider, profile, "test", 10, batch_size=10)
    assert len(result.rows) == 10
    assert result.dropped == 0
    assert result.attempts == 1
    assert list(result.rows.columns) == ["age", "plan", "active"]


def test_invalid_batch_retried_with_error_feedback(profile) -> None:
    bad = {"rows": [{"age": 5, "plan": "gold", "active": "maybe"}]}  # all three wrong
    good = {"rows": [valid_row(0), valid_row(1)]}
    provider = queue_provider([bad, good])
    result = generate_rows(provider, profile, "d", 2, batch_size=2)
    assert len(result.rows) == 2
    # second prompt must carry the validation errors back to the Provider
    second_prompt = provider.calls[1]["prompt"]
    assert "below minimum" in second_prompt
    assert "not one of" in second_prompt
    assert "not boolean" in second_prompt
    assert result.attempts == 2


def test_still_invalid_rows_are_dropped_and_counted(profile) -> None:
    mixed = {
        "rows": [valid_row(0), valid_row(1), {"age": 999, "plan": "platinum", "active": True}]
    }
    provider = queue_provider([mixed])  # repeats forever after the queue ends
    result = generate_rows(provider, profile, "d", 2, batch_size=3, retries=1)
    assert len(result.rows) == 2  # the two good rows survived
    assert result.dropped >= 1  # the bad row was counted on every attempt it appeared
    assert any("above maximum" in f or "not one of" in f for f in result.failures)


def test_coercion_accepts_plausible_imposters(profile) -> None:
    column = profile.column("age")
    row, errors = coerce_and_validate({"age": "42", "plan": "pro", "active": "yes"}, profile)
    assert errors == []
    assert row["age"] == 42 and isinstance(row["age"], int)
    assert row["active"] is True


def test_timeout_surfaces_and_stops_without_hammering(profile) -> None:
    provider = queue_provider([ProviderTimeoutError("down")])
    result = generate_rows(provider, profile, "d", 5, batch_size=5, retries=2)
    assert len(result.rows) == 0
    assert "Provider call failed repeatedly" in " ".join(result.failures)


def test_balance_nudge_in_prompt(profile) -> None:
    provider = queue_provider([{"rows": [valid_row(0)]}])
    generate_rows(provider, profile, "d", 1, batch_size=1, balance={"plan": {"basic": 0.5, "pro": 0.5}})
    assert "about 50%" in provider.calls[0]["prompt"]
    assert "plan=basic" in provider.calls[0]["prompt"]


def test_prompt_contains_profile_and_samples(profile) -> None:
    provider = queue_provider([{"rows": [valid_row(0)]}])
    generate_rows(provider, profile, "customer dataset", 1, batch_size=1, sample_rows=[valid_row(3)])
    prompt = provider.calls[0]["prompt"]
    assert "customer dataset" in prompt
    assert '"categories": ["basic", "pro"]' in prompt
    assert "Example real rows" in prompt


def test_rows_schema_requires_every_column(profile) -> None:
    schema = rows_schema(profile)
    item = schema["properties"]["rows"]["items"]
    assert set(item["required"]) == {"age", "plan", "active"}
