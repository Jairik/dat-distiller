"""CLI Providers: Claude Code, Codex, OpenCode — headless, locked down.

Uploaded data may contain prompt injection, so every call runs the CLI in
its most restrictive mode, in a fresh empty temp working directory that is
deleted afterwards, with the prompt on stdin and a hard timeout:

- Claude Code: ``claude --print --output-format json --allowedTools ""``
  (headless mode: https://docs.anthropic.com/en/docs/claude-code/headless ;
  ``--allowedTools ""`` enables no tools).
- Codex: ``codex exec --sandbox read-only --skip-git-repo-check``
  (https://developers.openai.com/codex/cli/scripting ; read-only sandbox is
  Codex's most restrictive mode).
- OpenCode: ``opencode run --format json``
  (https://opencode.ai/docs/cli ; ``run`` is non-interactive, but OpenCode
  has no no-tools flag — callers see a ``cli_no_tools`` warning Check the
  first time it is used in a Project).

Tests stub the subprocess layer and assert the exact command lines.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .base import (
    InvalidStructuredOutputError,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderTimeoutError,
    parse_json_reply,
    validate_structured,
)

DEFAULT_TIMEOUT = 120.0
AGENT_DEFAULT = "agent default"


class SubprocessProvider:
    """Base: one headless, sandboxed subprocess call per structured completion."""

    id = ""
    cli_bin = ""
    docs_url = ""
    #: False when the CLI cannot be put into a no-tools mode (#14 Check)
    has_no_tools_mode = True

    def __init__(self, model: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.model = None if not model or model == AGENT_DEFAULT else model
        self.timeout = timeout

    # -- subclass hooks -------------------------------------------------------

    def command(self) -> list[str]:
        raise NotImplementedError

    def extract_text(self, stdout: str) -> str:
        """Pull the assistant's final text out of the CLI's stdout."""
        return stdout

    # -- Provider protocol -------------------------------------------------------

    def complete(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        checks_store=None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        if shutil.which(self.cli_bin) is None:
            raise ProviderNotConfiguredError(
                f"{self.cli_bin} was not found on PATH. Install it and sign in: {self.docs_url}"
            )
        with tempfile.TemporaryDirectory(prefix="dat-distiller-provider-") as workdir:
            try:
                completed = subprocess.run(  # noqa: S603 - fixed argv, sandboxed cwd
                    self.command(),
                    input=prompt,
                    capture_output=True,
                    text=True,
                    cwd=workdir,  # fresh empty dir, deleted by the context manager
                    timeout=self.timeout,
                    shell=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ProviderTimeoutError(
                    f"{self.cli_bin} timed out after {self.timeout}s"
                ) from exc
            except OSError as exc:
                raise ProviderError(f"could not run {self.cli_bin}: {exc}") from exc
        if completed.returncode != 0:
            raise ProviderError(
                f"{self.cli_bin} exited {completed.returncode}: {completed.stderr[-400:]}"
            )
        # First-use warning for CLIs without a no-tools mode (once per Project).
        if (
            not self.has_no_tools_mode
            and checks_store is not None
            and project_id is not None
            and not checks_store.has_check("cli_no_tools", "project", project_id)
        ):
            checks_store.register(
                kind="cli_no_tools",
                severity="warning",
                message=f"{self.cli_bin} has no no-tools mode; prompts run with reduced guarantees",
                subject_type="project",
                subject_id=project_id,
                details={"provider": self.id, "docs": self.docs_url},
                step="provider",
            )
        text = self.extract_text(completed.stdout)
        payload = parse_json_reply(text)
        errors = validate_structured(payload, schema)
        if errors:
            raise InvalidStructuredOutputError(
                f"{self.cli_bin} output failed schema validation", errors
            )
        return payload


class ClaudeCodeProvider(SubprocessProvider):
    id = "claude"
    cli_bin = "claude"
    docs_url = "https://docs.anthropic.com/en/docs/claude-code/headless"
    has_no_tools_mode = True

    def command(self) -> list[str]:
        cmd = [
            "claude",
            "--print",  # headless, single shot
            "--output-format", "json",
            "--allowedTools", "",  # no tools may run: text in, text out
        ]
        if self.model:
            cmd += ["--model", self.model]
        return cmd

    def extract_text(self, stdout: str) -> str:
        try:
            return str(json.loads(stdout)["result"])
        except (ValueError, KeyError) as exc:
            raise ProviderError(f"unexpected claude output: {stdout[:200]}") from exc


class CodexProvider(SubprocessProvider):
    id = "codex"
    cli_bin = "codex"
    docs_url = "https://developers.openai.com/codex/cli/scripting"
    has_no_tools_mode = True  # read-only sandbox + no network/approval prompts

    def command(self) -> list[str]:
        cmd = [
            "codex", "exec",
            "--sandbox", "read-only",  # most restrictive sandbox
            "--skip-git-repo-check",  # our cwd is not a git repo (by design)
            "-",  # read the prompt from stdin
        ]
        if self.model:
            cmd += ["--model", self.model]
        return cmd

    def extract_text(self, stdout: str) -> str:
        # `codex exec --json` emits JSONL events; the last assistant message wins.
        text = stdout
        found = False
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("type") in ("agent_message", "item.completed"):
                candidate = event.get("message") or (event.get("item") or {}).get("text")
                if candidate:
                    text, found = candidate, True
        if not found and not stdout.strip():
            raise ProviderError("codex produced no output")
        return str(text)


class OpenCodeProvider(SubprocessProvider):
    id = "opencode"
    cli_bin = "opencode"
    docs_url = "https://opencode.ai/docs/cli"
    #: `opencode run` is headless but has no flag to disable tools — the
    #: `cli_no_tools` warning Check (once per Project) covers this.
    has_no_tools_mode = False

    def command(self) -> list[str]:
        cmd = ["opencode", "run", "--format", "json", "-"]
        if self.model:
            cmd += ["--model", self.model]
        return cmd

    def extract_text(self, stdout: str) -> str:
        # `opencode run --format json` emits event objects per line; keep the
        # last text part.
        text = stdout
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            parts = (event.get("part") or {}) if isinstance(event, dict) else {}
            if parts.get("type") == "text" and parts.get("text"):
                text = parts["text"]
        return str(text)


CLI_PROVIDERS = {
    ClaudeCodeProvider.id: ClaudeCodeProvider,
    CodexProvider.id: CodexProvider,
    OpenCodeProvider.id: OpenCodeProvider,
}
