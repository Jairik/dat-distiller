"""Provider-suggested Column Specs."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from dat_distiller.generate.suggest import suggest_column_specs
from dat_distiller.profile import validate_specs
from dat_distiller.providers import FakeProvider

SPEC_SCHEMA = {
    "columns": [
        {"name": "age", "type": "integer", "min": 18, "max": 120, "description": "years"},
        {"name": "plan", "type": "categorical", "categories": ["basic", "pro"]},
        {"name": "review", "type": "text"},
    ]
}


def scripted_provider(payload):
    return FakeProvider(response_for=lambda prompt, schema: payload)


def test_suggest_returns_valid_specs_with_fake_provider() -> None:
    specs = suggest_column_specs(scripted_provider(SPEC_SCHEMA), "customer data")
    assert [s.name for s in specs] == ["age", "plan", "review"]
    assert specs[0].min == 18 and specs[0].type == "integer"
    assert specs[1].categories == ["basic", "pro"]
    assert validate_specs(specs) == []


def test_prompt_includes_description_and_column_hint() -> None:
    fake = FakeProvider(response_for=lambda p, s: SPEC_SCHEMA)
    suggest_column_specs(fake, "churn dataset with a twist", ["user_id"])
    prompt = fake.calls[0]["prompt"]
    assert "churn dataset with a twist" in prompt
    assert "user_id" in prompt
    assert fake.calls[0]["schema"]["properties"]["columns"]["items"]["properties"]["type"]["enum"]


def test_synthesized_fake_provider_output_is_valid() -> None:
    # default FakeProvider synthesis must satisfy the suggestion schema
    specs = suggest_column_specs(FakeProvider(), "any description here")
    assert len(specs) >= 1  # passes schema (FakeProvider validates before return)


def test_endpoint(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "dat_distiller.api.generate.resolve_provider",
        lambda settings, provider=None, model=None: scripted_provider(SPEC_SCHEMA),
    )
    r = client.post("/api/generate/suggest-columns", json={"description": "sales data"})
    assert r.status_code == 200
    body = r.json()
    assert [s["name"] for s in body["specs"]] == ["age", "plan", "review"]
    assert body["validation_errors"] == []
    assert client.post("/api/generate/suggest-columns", json={"description": "ab"}).status_code == 422


def test_endpoint_maps_provider_errors(client: TestClient, monkeypatch) -> None:
    from dat_distiller.providers.base import ProviderNotConfiguredError

    def unconfigured(settings, provider=None, model=None):
        raise ProviderNotConfiguredError("OpenRouter needs an API key")

    monkeypatch.setattr("dat_distiller.api.generate.resolve_provider", unconfigured)
    r = client.post("/api/generate/suggest-columns", json={"description": "sales data"})
    assert r.status_code == 422 and "API key" in r.text
