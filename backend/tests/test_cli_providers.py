"""CLI Providers run locked-down headless commands; stubbed subprocess layer."""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from dat_distiller.checks import CheckStore
from dat_distiller.providers import (
    ClaudeCodeProvider,
    CodexProvider,
    InvalidStructuredOutputError,
    OpenCodeProvider,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderTimeoutError,
)
from dat_distiller.providers import cli as cli_module
from dat_distiller.store import DatasetStore
from dat_distiller.store.paths import AppPaths

SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}
ANSWER_JSON = '{"answer": 7}'


class StubRun:
    def __init__(self, stdout="", returncode=0, error=None):
        self.stdout, self.returncode, self.error = stdout, returncode, error
        self.calls: list[dict] = []

    def __call__(self, argv, **kwargs):
        self.calls.append(
            {
                "argv": argv,
                "cwd": kwargs.get("cwd"),
                "cwd_was_dir": os.path.isdir(kwargs.get("cwd", "")),
                "input": kwargs.get("input"),
                "timeout": kwargs.get("timeout"),
            }
        )
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(argv, self.returncode, self.stdout, "err-tail")


@pytest.fixture(autouse=True)
def on_path(monkeypatch):
    monkeypatch.setattr(cli_module.shutil, "which", lambda name: f"/usr/bin/{name}")


@pytest.fixture
def patch_run(monkeypatch):
    def apply(stub):
        monkeypatch.setattr(cli_module.subprocess, "run", stub)
        return stub

    return apply


def test_claude_command_is_locked_down_and_reads_stdin(patch_run) -> None:
    stub = patch_run(StubRun(stdout=json.dumps({"type": "result", "result": ANSWER_JSON})))
    provider = ClaudeCodeProvider(model="claude-sonnet-4")
    assert provider.complete("make json", SCHEMA) == {"answer": 7}
    call = stub.calls[0]
    assert call["argv"] == [
        "claude", "--print", "--output-format", "json", "--allowedTools", "",
        "--model", "claude-sonnet-4",
    ]
    assert call["input"] == "make json"
    assert call["cwd_was_dir"] is True  # fresh temp working dir existed during the run
    assert call["timeout"] == 120.0
    # temp dir is gone again afterwards
    assert not os.path.isdir(call["cwd"])


def test_codex_command_uses_read_only_sandbox_and_parses_events(patch_run) -> None:
    event = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": ANSWER_JSON}})
    stub = patch_run(StubRun(stdout=f"{json.dumps({'type':'cmd'})}\n{event}\n"))
    provider = CodexProvider(model=None)
    assert provider.complete("make json", SCHEMA) == {"answer": 7}
    argv = stub.calls[0]["argv"]
    assert argv[:5] == ["codex", "exec", "--sandbox", "read-only", "--skip-git-repo-check"]
    assert "-" in argv  # prompt via stdin


def test_opencode_command_and_text_extraction(patch_run) -> None:
    event = json.dumps({"type": "text", "part": {"type": "text", "text": ANSWER_JSON}})
    stub = patch_run(StubRun(stdout=event))
    provider = OpenCodeProvider(model="anthropic/claude-sonnet-4")
    assert provider.complete("make json", SCHEMA) == {"answer": 7}
    assert stub.calls[0]["argv"] == [
        "opencode", "run", "--format", "json", "-", "--model", "anthropic/claude-sonnet-4"
    ]


def test_model_agent_default_defers_to_cli(patch_run) -> None:
    stub = patch_run(StubRun(stdout=json.dumps({"result": ANSWER_JSON})))
    ClaudeCodeProvider(model="agent default").complete("p", SCHEMA)
    assert "--model" not in stub.calls[0]["argv"]


def test_typed_errors(patch_run) -> None:
    patch_run(StubRun(error=subprocess.TimeoutExpired(cmd="claude", timeout=1)))
    with pytest.raises(ProviderTimeoutError):
        ClaudeCodeProvider().complete("p", SCHEMA)

    patch_run(StubRun(stdout="", returncode=2))
    with pytest.raises(ProviderError, match="exited 2"):
        ClaudeCodeProvider().complete("p", SCHEMA)

    patch_run(StubRun(stdout="chatter without json"))
    # the JSON *envelope* itself is broken -> transport-level failure
    with pytest.raises(ProviderError, match="unexpected claude output"):
        ClaudeCodeProvider().complete("p", SCHEMA)

    patch_run(StubRun(stdout=json.dumps({"result": '{"answer": "seven"}'})))
    with pytest.raises(InvalidStructuredOutputError) as err:
        ClaudeCodeProvider().complete("p", SCHEMA)
    assert err.value.errors


def test_missing_binary_is_actionable(monkeypatch) -> None:
    monkeypatch.setattr(cli_module.shutil, "which", lambda name: None)
    with pytest.raises(ProviderNotConfiguredError, match="docs.anthropic.com"):
        ClaudeCodeProvider().complete("p", SCHEMA)


def test_opencode_raises_once_per_project_warning_check(patch_run, isolated_data_dir) -> None:
    patch_run(StubRun(stdout=json.dumps({"part": {"type": "text", "text": ANSWER_JSON}})))
    store = DatasetStore(AppPaths(root=isolated_data_dir))
    checks = CheckStore(store.db)
    project = store.create_project("CLI")
    provider = OpenCodeProvider()

    provider.complete("p", SCHEMA, checks_store=checks, project_id=project.id)
    provider.complete("p", SCHEMA, checks_store=checks, project_id=project.id)
    warnings = checks.list_for_subject("project", project.id)
    assert len(warnings) == 1 and warnings[0].kind == "cli_no_tools"

    # Claude has a no-tools mode: no warning
    patch_run(StubRun(stdout=json.dumps({"result": ANSWER_JSON})))
    ClaudeCodeProvider().complete("p", SCHEMA, checks_store=checks, project_id=project.id)
    assert not checks.has_check("cli_no_tools", "project", project.id + "-none")
    claude_project = store.create_project("ClaudeUser")
    ClaudeCodeProvider().complete("p", SCHEMA, checks_store=checks, project_id=claude_project.id)
    assert not checks.has_check("cli_no_tools", "project", claude_project.id)
