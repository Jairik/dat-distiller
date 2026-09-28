"""Providers: interface, FakeProvider, OpenRouter (stubbed HTTP), resolution."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from dat_distiller.providers import FakeProvider
from dat_distiller.providers.base import (
    InvalidStructuredOutputError,
    ProviderNotConfiguredError,
    ProviderTimeoutError,
    parse_json_reply,
    synthesize,
    validate_structured,
)
from dat_distiller.providers.openrouter import OpenRouterProvider, list_models
from dat_distiller.providers.registry import FAKE_PROVIDERS_ENV, resolve_provider
from dat_distiller.settings import SettingsStore

SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "age": {"type": "number", "minimum": 18, "maximum": 90},
                    "plan": {"type": "string", "enum": ["basic", "pro"]},
                },
                "required": ["age", "plan"],
            },
        },
        "count": {"type": "integer"},
    },
    "required": ["rows", "count"],
}


def test_fake_provider_synthesizes_schema_conforming_output() -> None:
    fake = FakeProvider()
    out = fake.complete("make rows", SCHEMA)
    assert validate_structured(out, SCHEMA) == []
    assert out["rows"][0]["age"] >= 18  # honors minimum
    assert fake.calls[0]["prompt"] == "make rows"


def test_fake_provider_scripted_responses() -> None:
    queue = iter([{"rows": [], "count": 0}, {"bad": 1}])

    def responder(prompt, schema):
        return next(queue)

    fake = FakeProvider(response_for=responder)
    assert fake.complete("p", SCHEMA) == {"rows": [], "count": 0}
    with pytest.raises(InvalidStructuredOutputError) as err:
        fake.complete("p", SCHEMA)
    assert err.value.errors  # schema violations listed for feedback


def test_parse_json_reply_variants() -> None:
    assert parse_json_reply('{"a": 1}') == {"a": 1}
    assert parse_json_reply('Sure! ```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_reply('Here: {"a": [1,2]} done') == {"a": [1, 2]}
    with pytest.raises(InvalidStructuredOutputError):
        parse_json_reply("no json here")


def test_synthesize_handles_common_shapes() -> None:
    assert synthesize({"type": "object", "properties": {"x": {"type": "integer"}}}) == {"x": 0}
    assert synthesize({"type": "array", "items": {"type": "boolean"}}) == [False]
    assert synthesize({"type": "string", "enum": ["a", "b"]}) == "a"
    assert synthesize({"type": "number", "minimum": -5, "maximum": -1}) == -1.0


def stub_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def openrouter_reply(payload) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
    )


def test_openrouter_validates_structured_output() -> None:
    provider = OpenRouterProvider("k", "m/x", client=stub_client(lambda req: openrouter_reply({"rows": [{"age": 30, "plan": "pro"}], "count": 1})))
    out = provider.complete("p", SCHEMA)
    assert out["count"] == 1


def test_openrouter_error_paths() -> None:
    bad_json = OpenRouterProvider("k", client=stub_client(lambda req: httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})))
    with pytest.raises(InvalidStructuredOutputError):
        bad_json.complete("p", SCHEMA)

    schema_violation = OpenRouterProvider("k", client=stub_client(lambda req: openrouter_reply({"rows": [{"age": 5, "plan": "gold"}]})))
    with pytest.raises(InvalidStructuredOutputError) as err:
        schema_violation.complete("p", SCHEMA)
    assert err.value.errors

    unauthorized = OpenRouterProvider("bad-key", client=stub_client(lambda req: httpx.Response(401)))
    with pytest.raises(ProviderNotConfiguredError):
        unauthorized.complete("p", SCHEMA)

    def timeout_handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    slow = OpenRouterProvider("k", client=stub_client(timeout_handler))
    with pytest.raises(ProviderTimeoutError):
        slow.complete("p", SCHEMA)


def test_openrouter_never_reaches_the_network_in_tests(monkeypatch) -> None:
    # the tests above route through MockTransport; assert list_models does too
    models = list_models("k", client=stub_client(lambda req: httpx.Response(200, json={"data": [{"id": "a/b", "name": "A B"}, {"nope": 1}]})))
    assert models == [{"id": "a/b", "name": "A B"}]


def test_resolution_default_overrides_and_guards(isolated_data_dir, monkeypatch) -> None:
    store = SettingsStore(isolated_data_dir.parent / "config")
    monkeypatch.delenv(FAKE_PROVIDERS_ENV, raising=False)

    with pytest.raises(ProviderNotConfiguredError, match="API key"):
        resolve_provider(store)  # default openrouter, no key

    store.set_key("openrouter", "sk-or")
    resolved = resolve_provider(store)
    assert resolved.id == "openrouter" and resolved.model == "openai/gpt-4o-mini"

    store.save({"models": {"openrouter": "anthropic/claude-sonnet-4"}})
    assert resolve_provider(store).model == "anthropic/claude-sonnet-4"
    # per-request overrides win
    assert resolve_provider(store, provider="fake").id == "fake"
    assert resolve_provider(store, model="x/y").model == "x/y"

    with pytest.raises(ProviderNotConfiguredError, match="unknown provider"):
        resolve_provider(store, provider="gpt-master")

    monkeypatch.setenv(FAKE_PROVIDERS_ENV, "1")
    assert resolve_provider(store).id == "fake"


def test_provider_api_endpoints(client: TestClient, monkeypatch) -> None:
    options = client.get("/api/providers/options").json()["providers"]
    by_id = {o["id"]: o for o in options}
    assert by_id["openrouter"]["available"] is False
    assert by_id["fake"]["available"] is True
    assert client.get("/api/providers/openrouter/models").status_code == 422

    client.app.state.settings.set_key("openrouter", "sk-or")  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "dat_distiller.api.providers.list_models",
        lambda key: [{"id": "a/b", "name": "A B"}],
    )
    models = client.get("/api/providers/openrouter/models").json()
    assert models == {"models": [{"id": "a/b", "name": "A B"}]}
