"""Dat Distiller Providers: text-in, structured-out generators.

See base.py for the Provider protocol + FakeProvider, openrouter.py for the
HTTP Provider, cli.py (#14) for Claude Code / Codex / OpenCode, and
registry.py to resolve which one a request should use.
"""

from .base import (
    FakeProvider,
    InvalidStructuredOutputError,
    Provider,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderTimeoutError,
    parse_json_reply,
    validate_structured,
)
from .cli import CLI_PROVIDERS, ClaudeCodeProvider, CodexProvider, OpenCodeProvider
from .registry import FAKE_PROVIDERS_ENV, provider_options, resolve_provider

__all__ = [
    "CLI_PROVIDERS",
    "ClaudeCodeProvider",
    "CodexProvider",
    "FAKE_PROVIDERS_ENV",
    "FakeProvider",
    "InvalidStructuredOutputError",
    "OpenCodeProvider",
    "Provider",
    "ProviderError",
    "ProviderNotConfiguredError",
    "ProviderTimeoutError",
    "parse_json_reply",
    "provider_options",
    "resolve_provider",
    "validate_structured",
]
