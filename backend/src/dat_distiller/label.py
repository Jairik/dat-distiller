"""Labeling: run Jev Questions over a Dataset Version.

Every row's State goes to Jev in ONE call (all questions at once), rows move
through the concurrency pool (retries + exponential backoff), and progress is
checkpointed so a cancelled or interrupted run **resumes with already-labeled
rows skipped**. The output is a new Dataset Version:

- Noul  → ``<name>`` (bool) + ``<name>__confidence``
- Choice → ``<name>`` + ``<name>__confidence`` + ``<name>__p_<option>``
- Score → ``<name>`` (expected score) + ``<name>__confidence`` +
  ``<name>__probabilities`` (JSON)

plus ``jev`` label Provenance per row. Rows Jev keeps failing after retries
get null Label Columns and are reported in the job result.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from .jev import Choice, Noul, Score, resolve_jev, serialize_state
from .jobs import JobContext
from .pool import run_pool
from .store.provenance import PROVENANCE_COLUMN, Provenance
from .store.store import DatasetStore

CHECKPOINT_EVERY = 10
CONCURRENCY = 4
RETRIES = 2

_QuestionDict = dict[str, Any]


class LabelRequest(BaseModel):
    version_id: str
    questions: list[_QuestionDict] = Field(min_length=1)
    state_columns: list[str] | None = None
    jev_model: str | None = None
    preview_rows: int = Field(default=5, ge=1, le=20)


def parse_question(spec: _QuestionDict) -> Noul | Choice | Score:
    qtype = spec.get("type")
    name = str(spec.get("name", "")).strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(f"question name {name!r} must be snake_case")
    instructions = str(spec.get("instructions", "")).strip()
    if not instructions:
        raise ValueError(f"question {name!r} needs instructions")
    if qtype == "noul":
        return Noul(name=name, instructions=instructions)
    if qtype == "choice":
        criteria = spec.get("criteria") or {}
        if len(criteria) < 2:
            raise ValueError(f"choice question {name!r} needs at least 2 criteria")
        return Choice(name=name, instructions=instructions, criteria=dict(criteria))
    if qtype == "score":
        levels = spec.get("levels") or []
        if len(levels) < 2:
            raise ValueError(f"score question {name!r} needs at least 2 levels")
        return Score(name=name, instructions=instructions, levels=list(levels))
    raise ValueError(f"unknown question type {qtype!r}")


def _option_column(option: str) -> str:
    return "p_" + re.sub(r"[^a-zA-Z0-9_]+", "_", option).strip("_").lower()


def label_columns(questions: list[Noul | Choice | Score]) -> list[str]:
    """Output columns per the naming convention, deterministic order."""
    columns: list[str] = []
    for question in questions:
        columns += [question.name, f"{question.name}__confidence"]
        if isinstance(question, Choice):
            columns += [f"{question.name}__{_option_column(key)}" for key in question.criteria]
        elif isinstance(question, Score):
            columns.append(f"{question.name}__probabilities")
    return columns


def _values_for(question: Any, answer: dict[str, Any]) -> dict[str, Any]:
    """Normalized Jev answer → output-column values."""
    name = question.name
    confidence = answer.get("confidence")
    if isinstance(question, Noul):
        return {name: answer["answer"] == "yes", f"{name}__confidence": confidence}
    if isinstance(question, Choice):
        values = {name: answer["answer"], f"{name}__confidence": confidence}
        probabilities = answer.get("probabilities") or {}
        for key in question.criteria:
            values[f"{name}__{_option_column(key)}"] = probabilities.get(key)
        return values
    values = {
        name: answer["answer"],
        f"{name}__confidence": confidence,
        f"{name}__probabilities": json.dumps(answer.get("probabilities") or {}),
    }
    return values


def state_columns_for(df: pd.DataFrame, requested: list[str] | None, outputs: list[str]) -> list[str]:
    base = requested or [c for c in df.columns if c != PROVENANCE_COLUMN]
    invalid = [c for c in base if c not in df.columns]
    if invalid:
        raise ValueError(f"unknown State columns: {invalid}")
    clash = [c for c in base if c in outputs]
    if clash:
        raise ValueError(f"State columns would overwrite label outputs: {clash}")
    return list(base)


def preview_labeling(
    jev: Any,
    df: pd.DataFrame,
    questions: list[Any],
    state_columns: list[str],
    rows: int = 5,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _, row in df.head(rows).iterrows():
        state = serialize_state(row.to_dict(), state_columns)
        answers = jev.answer_row(state, questions)
        out.append({"state": state, "answers": answers})
    return out


def estimate_labeling(row_count: int) -> dict[str, Any]:
    """One Jev call per row — all Jev Questions travel together."""
    return {
        "rows": row_count,
        "estimated_jev_calls": row_count,
        "uses_jev": row_count > 0,
    }


def run_labeling(
    ctx: JobContext,
    *,
    store: DatasetStore,
    jev: Any,
    soft_limit: int | None,
    raise_check: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    request = LabelRequest.model_validate(ctx.params)
    questions = [parse_question(q) for q in request.questions]
    names = [q.name for q in questions]
    if len(set(names)) != len(names):
        raise ValueError("duplicate question names")

    frame = store.load_dataframe(request.version_id, include_provenance=True)
    outputs = label_columns(questions)
    clash = [c for c in outputs if c in frame.columns]
    if clash:
        raise ValueError(f"label columns already exist on this version: {clash}")
    state_columns = state_columns_for(frame, request.state_columns, outputs)

    if soft_limit and len(frame) > soft_limit and raise_check:
        raise_check(
            "labeling_soft_limit",
            {"rows": len(frame), "soft_limit": soft_limit},
        )

    checkpoint = dict(ctx.load_checkpoint() or {})
    labeled: dict[str, dict[str, Any]] = dict(checkpoint.get("labeled", {}))

    todo = [i for i in range(len(frame)) if str(i) not in labeled]

    def worker(index: int) -> tuple[int, dict[str, Any]]:
        state = serialize_state(frame.iloc[index].to_dict(), state_columns)
        return index, jev.answer_row(state, questions)

    since_checkpoint = 0
    cancelled_midway = False
    while todo:
        chunk = todo[: max(CONCURRENCY * 5, 20)]
        todo = todo[len(chunk):]
        outcome = run_pool(
            chunk,
            worker,
            concurrency=CONCURRENCY,
            retries=RETRIES,
            should_cancel=lambda: ctx.cancelled,
        )
        for result in outcome.results:
            if result.error is None and not result.skipped:
                index, answers = result.value
                labeled[str(index)] = answers
                since_checkpoint += 1
                if since_checkpoint >= CHECKPOINT_EVERY:
                    ctx.save_checkpoint({"labeled": labeled})
                    since_checkpoint = 0
            if ctx.cancelled and result.skipped:
                cancelled_midway = True
        ctx.progress(min(len(labeled), len(frame)), len(frame), jev_calls=len(labeled))
        if ctx.cancelled:
            outcome.cancelled = True
            # keep everything done so far; resume skips it
            ctx.save_checkpoint({"labeled": labeled})
            raise RuntimeError("cancelled")
        if cancelled_midway:
            break

    # rows Jev never answered (permanent failures)
    failed = [i for i in range(len(frame)) if str(i) not in labeled]
    frame_out = frame.copy()
    n_rows = len(frame_out)
    for question in questions:
        if isinstance(question, Noul):
            frame_out[question.name] = pd.array([pd.NA] * n_rows, dtype="boolean")
        else:
            frame_out[question.name] = (
                pd.array([pd.NA] * n_rows, dtype="Float64")
                if isinstance(question, Score)
                else pd.Series([None] * n_rows, dtype="object")
            )
        frame_out[f"{question.name}__confidence"] = pd.array([pd.NA] * n_rows, dtype="Float64")
        if isinstance(question, Choice):
            for key in question.criteria:
                frame_out[f"{question.name}__{_option_column(key)}"] = pd.array(
                    [pd.NA] * n_rows, dtype="Float64"
                )
        elif isinstance(question, Score):
            frame_out[f"{question.name}__probabilities"] = pd.Series([None] * n_rows, dtype="object")
    provenance: list[dict[str, Any]] = []
    for index in range(len(frame_out)):
        answers = labeled.get(str(index))
        row_prov = Provenance.parse(frame_out[PROVENANCE_COLUMN].iloc[index]) if PROVENANCE_COLUMN in frame_out else Provenance.row("uploaded")
        if answers:
            label_origins = dict(row_prov.get("label_origins") or {})
            for question in questions:
                answer = answers.get(question.name)
                if answer:
                    values = _values_for(question, answer)
                    for column, value in values.items():
                        frame_out.iat[index, frame_out.columns.get_loc(column)] = value
                    label_origins[question.name] = {"origin": "jev", "confidence": answer.get("confidence")}
            row_prov["label_origins"] = label_origins
        provenance.append(row_prov)
    frame_out[PROVENANCE_COLUMN] = [json.dumps(p) for p in provenance]

    version = store.create_version(
        store.get_version(request.version_id).project_id,
        frame_out,
        parent_id=request.version_id,
        origin="labeled",
        meta={
            "labeling": {
                "questions": request.questions,
                "state_columns": state_columns,
                "failed_rows": failed,
            }
        },
    )
    ctx.save_checkpoint({"labeled": labeled, "completed_version": version.id})
    return {
        "version_id": version.id,
        "labeled_rows": len(labeled),
        "failed_rows": failed,
        "failed_count": len(failed),
        "label_columns": outputs,
    }


def register_label_job(manager: Any, app: Any) -> None:
    """Wire the resumable ``label`` job type to the app's stores."""

    def runner(ctx: JobContext) -> dict[str, Any]:
        from .checks import CheckStore

        settings = app.state.settings
        jev = resolve_jev(settings, model=ctx.params.get("jev_model"))
        checks: CheckStore = app.state.checks
        soft = int(settings.load().get("soft_limits", {}).get("labeling_calls", 5000))

        def raise_check(kind: str, details: dict[str, Any]) -> None:
            subject_id = ctx.project_id or ""
            if checks.has_check(kind, "project", subject_id):
                return
            checks.register(
                kind=kind,
                severity="warning",
                message=(
                    f"Labeling {details['rows']} rows exceeds the soft limit of "
                    f"{details['soft_limit']} Jev calls"
                ),
                subject_type="project",
                subject_id=subject_id,
                details=details,
            )

        return run_labeling(
            ctx,
            store=app.state.store,
            jev=jev,
            soft_limit=soft,
            raise_check=raise_check,
        )

    manager.register("label", runner, resumable=True)
