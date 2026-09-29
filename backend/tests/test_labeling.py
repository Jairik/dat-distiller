"""Labeling: naming conventions, resumable runs, retry reporting, soft limit."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dat_distiller.jev import Choice, FakeJev, Noul, Score
from dat_distiller.jev import FAKE_JEV_ENV
from dat_distiller.label import (
    LabelRequest,
    estimate_labeling,
    label_columns,
    parse_question,
    run_labeling,
)
from dat_distiller.store.db import Database
from dat_distiller.store.paths import AppPaths
from dat_distiller.store.provenance import PROVENANCE_COLUMN
from dat_distiller.store.store import DatasetStore

QUESTIONS = [
    {"type": "noul", "name": "is_spam", "instructions": "Is this spam?"},
    {
        "type": "choice",
        "name": "tone",
        "instructions": "Tone?",
        "criteria": {"pos": "positive", "neg": "negative"},
    },
    {"type": "score", "name": "quality", "instructions": "How good?", "levels": ["bad", "ok", "great"]},
]


@pytest.fixture
def store(isolated_data_dir) -> DatasetStore:
    return DatasetStore(AppPaths(root=isolated_data_dir))


@pytest.fixture
def version(store: DatasetStore) -> str:
    project = store.create_project("label lab")
    df = pd.DataFrame({"body": [f"text number {i}" for i in range(8)]})
    return store.create_version(project.id, df, origin="uploaded").id


class FakeCtx:
    """In-memory JobContext double with checkpoint persistence + cancel."""

    def __init__(self, params: dict) -> None:
        self.params = params
        self.project_id = "p"
        self.checkpoint: dict | None = None
        self.progress_log: list[tuple[int, int]] = []
        self.cancel = False

    def progress(self, done: int, total: int, **extra) -> None:
        self.progress_log.append((done, total))

    @property
    def cancelled(self) -> bool:
        return self.cancel

    def save_checkpoint(self, data: dict) -> None:
        self.checkpoint = json.loads(json.dumps(data))

    def load_checkpoint(self) -> dict | None:
        return self.checkpoint


class FlakyJev:
    """Answers rows in order; can cancel the ctx mid-run; can fail rows hard."""

    def __init__(self, fail_indices: set[int] = frozenset(), cancel_after: int | None = None, ctx: FakeCtx | None = None):
        self.fail_indices = fail_indices
        self.cancel_after = cancel_after
        self.ctx = ctx
        self.calls = 0

    def answer_row(self, state, questions):
        self.calls += 1
        index = int(state.split("text number ")[1].splitlines()[0])
        if self.cancel_after is not None and self.calls >= self.cancel_after and self.ctx is not None:
            self.ctx.cancel = True
        if index in self.fail_indices:
            raise RuntimeError("jev exploded")
        return {
            q.name: {
                "kind": type(q).__name__.lower(),
                "answer": {"noul": "yes", "choice": "pos", "score": 1.0}[type(q).__name__.lower()],
                "confidence": 0.9,
                "probabilities": {"yes": 0.9, "no": 0.1} if isinstance(q, Noul) else (
                    {"pos": 0.9, "neg": 0.1} if isinstance(q, Choice) else {"bad": 0.1, "ok": 0.8, "great": 0.1}
                ),
            }
            for q in questions
        }


def test_naming_convention_covers_all_three_types() -> None:
    questions = [parse_question(q) for q in QUESTIONS]
    assert label_columns(questions) == [
        "is_spam",
        "is_spam__confidence",
        "tone",
        "tone__confidence",
        "tone__p_pos",
        "tone__p_neg",
        "quality",
        "quality__confidence",
        "quality__probabilities",
    ]


# -- criteria that would share a probability column --------------------------


def test_criteria_that_normalise_alike_are_refused() -> None:
    """Two options, one column: the second write would erase the first.

    Each Choice option's probability goes in a sibling column named after it, so
    options that normalise alike land on the same one. "Very Positive" and "very
    positive" both become `p_very_positive`, and the Jev's probability for the
    option it actually answered was being written to the other's column while
    the answer column disagreed with both. Nothing errored; half the data was
    simply gone.
    """
    with pytest.raises(ValueError, match="both become the column tone__p_very_positive"):
        parse_question(
            {
                "type": "choice",
                "name": "tone",
                "instructions": "how warm is it?",
                "criteria": {"Very Positive": 0.5, "very positive": 0.5},
            }
        )
    # Punctuation and underscores collide the same way.
    with pytest.raises(ValueError, match="both become the column"):
        parse_question(
            {
                "type": "choice",
                "name": "tone",
                "instructions": "how warm is it?",
                "criteria": {"a-b": 0.5, "a_b": 0.5},
            }
        )


def test_label_columns_can_never_return_a_duplicate() -> None:
    """The guarantee lives with the function that mints the names.

    `parse_question` catches this over the API, but a `Choice` can be built
    directly — and a duplicate column is silently destructive rather than merely
    wrong, so the check belongs where the names are produced too.
    """
    clashing = [
        Choice(
            name="tone",
            instructions="how warm is it?",
            criteria={"Very Positive": 0.5, "very positive": 0.5},
        )
    ]
    with pytest.raises(ValueError, match="write the same column twice"):
        label_columns(clashing)


def test_distinct_criteria_still_get_distinct_columns() -> None:
    """The check must not fire on options that merely look similar."""
    question = parse_question(
        {
            "type": "choice",
            "name": "tone",
            "instructions": "how warm is it?",
            "criteria": {"Very positive": 0.4, "Neutral": 0.4, "not neutral": 0.2},
        }
    )
    assert label_columns([question]) == [
        "tone",
        "tone__confidence",
        "tone__p_very_positive",
        "tone__p_neutral",
        "tone__p_not_neutral",
    ]


def test_question_validation() -> None:
    with pytest.raises(ValueError, match="snake_case"):
        parse_question({"type": "noul", "name": "Bad Name", "instructions": "x"})
    with pytest.raises(ValueError, match="at least 2 criteria"):
        parse_question({"type": "choice", "name": "c", "instructions": "x", "criteria": {"a": None}})
    with pytest.raises(ValueError, match="at least 2 levels"):
        parse_question({"type": "score", "name": "s", "instructions": "x", "levels": ["only"]})
    with pytest.raises(ValueError, match="unknown question type"):
        parse_question({"type": "vibes", "name": "v", "instructions": "x"})


def test_estimate_is_one_call_per_row() -> None:
    assert estimate_labeling(120) == {
        "rows": 120,
        "estimated_jev_calls": 120,
        "uses_jev": True,
    }


def test_run_produces_label_version_with_jev_provenance(store: DatasetStore, version: str) -> None:
    ctx = FakeCtx({"version_id": version, "questions": QUESTIONS})
    jev = FakeJev()
    result = run_labeling(ctx, store=store, jev=jev, soft_limit=None)
    assert result["labeled_rows"] == 8
    assert result["failed_count"] == 0

    new_version = store.get_version(result["version_id"])
    assert new_version.origin == "labeled"
    assert new_version.parent_id == version
    frame = store.load_dataframe(result["version_id"], include_provenance=True)
    for column in label_columns([parse_question(q) for q in QUESTIONS]):
        assert column in frame.columns
    assert frame["is_spam"].isin([True, False, None]).all()
    assert frame["tone__p_pos"].between(0, 1).all()
    json.loads(frame["quality__probabilities"].iloc[0])  # valid JSON
    prov = [json.loads(p) for p in frame[PROVENANCE_COLUMN]]
    assert all("is_spam" in p.get("label_origins", {}) for p in prov)
    assert prov[0]["label_origins"]["is_spam"]["origin"] == "jev"
    assert new_version.provenance_summary.get("jev") == 8


def test_resume_skips_already_labeled_rows(store: DatasetStore, version: str) -> None:
    params = {"version_id": version, "questions": QUESTIONS}
    ctx1 = FakeCtx(params)
    jev1 = FlakyJev(cancel_after=4, ctx=ctx1)
    with pytest.raises(RuntimeError, match="cancelled"):
        run_labeling(ctx1, store=store, jev=jev1, soft_limit=None)
    assert len(ctx1.checkpoint["labeled"]) >= 3  # progress survived the cancel
    calls_after_cancel = jev1.calls

    ctx2 = FakeCtx(params)
    ctx2.checkpoint = ctx1.checkpoint  # same job store checkpoint
    jev2 = FlakyJev()
    result = run_labeling(ctx2, store=store, jev=jev2, soft_limit=None)
    # second pass only asked Jev about the rows that were NOT yet labeled
    assert jev2.calls == 8 - len(ctx1.checkpoint["labeled"])
    assert result["labeled_rows"] == 8
    assert jev1.calls + jev2.calls == 8  # no row ever labeled twice
    _ = calls_after_cancel


def test_failed_rows_are_retried_then_reported(store: DatasetStore, version: str) -> None:
    ctx = FakeCtx({"version_id": version, "questions": QUESTIONS})
    jev = FlakyJev(fail_indices={2, 5})
    result = run_labeling(ctx, store=store, jev=jev, soft_limit=None)
    assert sorted(result["failed_rows"]) == [2, 5]
    assert result["labeled_rows"] == 6
    frame = store.load_dataframe(result["version_id"])
    assert frame.loc[2, "is_spam"] is None or pd.isna(frame.loc[2, "is_spam"])
    assert not pd.isna(frame.loc[0, "is_spam"])


def test_soft_limit_raises_warning_check(store: DatasetStore, version: str) -> None:
    from dat_distiller.checks import CheckStore

    checks = CheckStore(store.db)
    raised: list[tuple[str, dict]] = []

    def raise_check(kind: str, details: dict) -> None:
        checks.register(
            kind=kind, severity="warning", message=kind, subject_type="project",
            subject_id="proj", details=details,
        )
        raised.append((kind, details))

    ctx = FakeCtx({"version_id": version, "questions": QUESTIONS})
    run_labeling(ctx, store=store, jev=FakeJev(), soft_limit=3, raise_check=raise_check)
    assert raised and raised[0][0] == "labeling_soft_limit"
    assert checks.unacknowledged_warnings("project", "proj")


# -- endpoints ---------------------------------------------------------------


def _seed_version(client: TestClient) -> tuple[str, str]:
    project_id = client.post("/api/projects", json={"name": "lab"}).json()["id"]
    df = pd.DataFrame({"body": [f"text number {i}" for i in range(6)]})
    version_id = client.app.state.store.create_version(
        project_id, df, origin="uploaded"
    ).id
    return project_id, version_id


def wait_for_job(client: TestClient, job_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_label_endpoints_end_to_end(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv(FAKE_JEV_ENV, "1")
    project_id, version_id = _seed_version(client)

    estimate = client.post("/api/label/estimate", json={"version_id": version_id, "questions": QUESTIONS})
    assert estimate.json()["estimated_jev_calls"] == 6

    preview = client.post(
        "/api/label/preview", json={"version_id": version_id, "questions": QUESTIONS, "preview_rows": 2}
    )
    rows = preview.json()["rows"]
    assert len(rows) == 2
    assert set(rows[0]["answers"]) == {"is_spam", "tone", "quality"}
    assert "body: text number 0" in rows[0]["state"]

    start = client.post("/api/label/run", json={"version_id": version_id, "questions": QUESTIONS})
    assert start.status_code == 202
    job = wait_for_job(client, start.json()["id"])
    assert job["status"] == "completed", job["error"]
    assert job["result"]["labeled_rows"] == 6
    assert job["resumable"] is True

    new_version = client.get(f"/api/dataset-versions/{job['result']['version_id']}").json()
    assert new_version["origin"] == "labeled"
    kinds = {c["kind"] for c in new_version["columns"]}
    assert "boolean" in kinds or "bool" in kinds  # is_spam column landed


def test_label_endpoints_validate(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv(FAKE_JEV_ENV, "1")
    _, version_id = _seed_version(client)
    bad = client.post(
        "/api/label/preview",
        json={"version_id": version_id, "questions": [{"type": "choice", "name": "c", "instructions": "x", "criteria": {"one": None}}]},
    )
    assert bad.status_code == 422
    missing = client.post(
        "/api/label/preview",
        json={"version_id": "nope", "questions": QUESTIONS},
    )
    assert missing.status_code == 404
    clash = client.post(
        "/api/label/preview",
        json={"version_id": version_id, "questions": QUESTIONS, "state_columns": ["body", "tone"]},
    )
    assert clash.status_code == 422
