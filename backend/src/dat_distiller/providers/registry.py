"""Resolving which Provider to use: global default, per-request override."""

from __future__ import annotations

import os
from typing import Any

from .base import FakeProvider, Provider, ProviderNotConfiguredError
from .cli import CLI_PROVIDERS
from .openrouter import OpenRouterProvider

#: When set, every Provider resolves to FakeProvider (used by the e2e suite
#: and demos so no real agent or network call happens).
FAKE_PROVIDERS_ENV = "DAT_DISTILLER_FAKE_PROVIDERS"


def resolve_provider(
    settings_store,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> Provider:
    """Build the Provider for a request.

    ``provider``/``model`` are per-request overrides; otherwise the global
    default and its configured model win. Raises ProviderNotConfiguredError
    with a human-readable reason when the choice cannot run. When
    ``DAT_DISTILLER_FAKE_PROVIDERS`` is set, always returns a FakeProvider.
    """
    if os.environ.get(FAKE_PROVIDERS_ENV):
        return FakeProvider()
    settings = settings_store.load()
    provider_id = provider or settings.get("default_provider") or "openrouter"
    chosen_model = model or settings.get("models", {}).get(provider_id)

    if provider_id == "fake":
        return FakeProvider(model=chosen_model or "fake-model")
    if provider_id in CLI_PROVIDERS:
        # call sites needing the first-use Check pass checks_store/project_id
        # to complete() directly; resolution stays uniform here.
        return CLI_PROVIDERS[provider_id](model=chosen_model)
    if provider_id == "openrouter":
        key = settings_store.api_key("openrouter")
        if not key:
            raise ProviderNotConfiguredError(
                "OpenRouter needs an API key (set it in Settings or OPENROUTER_API_KEY)"
            )
        return OpenRouterProvider(api_key=key, model=chosen_model)
    raise ProviderNotConfiguredError(
        f"unknown provider {provider_id!r}; available: openrouter, "
        + "/".join(sorted(CLI_PROVIDERS))
        + ", fake"
    )


def provider_options(settings_store) -> list[dict[str, Any]]:
    """What the UI can pick as a Provider, with availability + reason."""
    settings = settings_store.load()
    options = []
    for provider_id in ("openrouter", "claude", "codex", "opencode", "fake"):
        available = True
        reason = None
        if provider_id == "openrouter" and not settings_store.api_key("openrouter"):
            available, reason = False, "no API key configured"
        if provider_id in ("claude", "codex", "opencode"):
            available = settings_store.detect_cli(provider_id) is not None
            reason = None if available else "CLI not found on PATH"
        options.append(
            {
                "id": provider_id,
                "available": available,
                "reason": reason,
                "model": settings.get("models", {}).get(provider_id),
            }
        )
    return options
