"""Draft a Jev Question with a Provider — a draft the user then edits.

Labeling normally starts from a question the user writes. This asks a Provider
for a first draft of it from a plain-English description of the wanted Label
Column: the Provider **picks the Jev Question type** (Noul / Choice / Score),
names it, and writes the instructions plus criteria or levels. Uses the
Provider's structured output, so the reply is schema-validated by
construction; anything semantically broken (a Choice without options, a Score
without a scale) raises :class:`ValueError` rather than reaching Jev.

Drafting only — nothing here labels anything, and a Provider never labels
(CONTEXT.md: Providers generate text only; Jev answers Jev Questions).
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from pydantic import BaseModel, Field

from .jev import Choice, JevQuestion, Noul, Score, question_kind
from .providers.base import Provider

#: A Provider's answer must say which kind of question it drafted, so the enum
#: order doubles as the fallback (``noul`` is what a bare schema synthesizes).
QUESTION_TYPES = ("noul", "choice", "score")

_CRITERION_ITEM = {
    "type": "object",
    "properties": {
        "key": {"type": "string", "minLength": 1},
        "description": {"type": ["string", "null"]},
    },
    "required": ["key"],
}

DRAFT_QUESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question_type": {
            "type": "string",
            "enum": list(QUESTION_TYPES),
            "description": "noul = yes/no judgement, choice = pick one category, score = graded level",
        },
        "name": {
            "type": "string",
            "minLength": 1,
            "description": "snake_case name, it becomes the Label Column name",
        },
        "instructions": {
            "type": "string",
            "minLength": 1,
            "description": "what Jev must decide for one row, precisely",
        },
        # choice only: the named options. Kept permissive (an array of
        # {key, description} pairs, or a plain key->description object) because
        # coercion accepts plausible imposters; the semantic check below is
        # what actually guarantees a usable question.
        "criteria": {"type": ["array", "object", "null"], "items": _CRITERION_ITEM},
        # score only: the ordinal scale, lowest -> highest
        "levels": {"type": ["array", "string", "null"], "items": {"type": "string", "minLength": 1}},
    },
    "required": ["question_type", "name", "instructions"],
}

_TYPE_GUIDE = """Pick the Jev Question type that fits what the column holds:
- noul: a yes/no judgement. Jev answers yes or no with a confidence. No options.
- choice: assigning the row to exactly one category. Give every category a short
  key plus a one-line description of what belongs in it.
- score: a graded judgement on an ordinal scale. Give the levels from lowest to
  highest, each a short phrase a reader could recognize (for example
  "1 - unusable" ... "5 - publication ready")."""


def build_draft_prompt(
    description: str, context: str | None = None, sample_rows: list[dict[str, Any]] | None = None
) -> str:
    """Prompt for one drafted Jev Question: description, context, sample rows."""
    parts = [
        (
            "You write Jev Questions for a Labeling step. Jev reads one row's State "
            "(that row's columns written out as 'column: value' lines) and answers "
            "the question; the answer is written into a new Label Column."
        ),
        _TYPE_GUIDE,
        (
            "Give the question a name: snake_case and short — it becomes the Label "
            "Column name. Write instructions that tell Jev exactly what to decide "
            "for a single row: what counts, what to ignore, and how to judge it when "
            "the row is ambiguous. Ask one thing, never two."
        ),
        "Label Column description (what the user wants this column to hold):\n" + description,
    ]
    if context:
        parts.append("Additional context:\n" + context)
    if sample_rows:
        parts.append(
            "Sample rows of the data Jev will read: " + json.dumps(list(sample_rows), ensure_ascii=False)
        )
    return "\n\n".join(parts)


def _snake_case(raw: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", raw.strip()).strip("_").lower()
    slug = re.sub(r"_{2,}", "_", slug)
    return slug or "jev_question"


def _description(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _criteria(raw: Any) -> dict[str, str | None] | None:
    """The Choice options as ``{key: description | None}``, tolerating shapes."""
    if isinstance(raw, Mapping):  # plain {"refund": "what counts as a refund", ...}
        plain = {str(key): _description(value) for key, value in raw.items() if _description(key)}
        return plain or None
    if isinstance(raw, (list, tuple)):
        items: list[Any] = list(raw)
    else:
        return None
    criteria: dict[str, str | None] = {}
    for item in items:
        if isinstance(item, Mapping):
            key = next(
                (
                    _description(item.get(field))
                    for field in ("key", "name", "label", "option")
                    if _description(item.get(field))
                ),
                None,
            )
            if key is None:
                continue
            text = next(
                (
                    _description(item.get(field))
                    for field in ("description", "desc", "definition", "criteria")
                    if _description(item.get(field))
                ),
                None,
            )
            criteria.setdefault(str(key), text)
        else:
            key = _description(item)
            if key is not None:
                criteria.setdefault(str(key), None)
    return criteria or None


def _levels(raw: Any) -> list[str] | None:
    """The Score scale as an ordered list of level phrases."""
    if isinstance(raw, str):
        items: list[Any] = re.split(r"[,\n;]+", raw)
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        return None
    levels: list[str] = []
    for item in items:
        text = _description(item) if not isinstance(item, Mapping) else None
        if text is None and isinstance(item, Mapping):
            text = next(
                (
                    _description(item.get(field))
                    for field in ("label", "name", "key", "level")
                    if _description(item.get(field))
                ),
                None,
            )
        if text and text not in levels:
            levels.append(text)
    return levels or None


def _warning(question: JevQuestion, raw_name: str, renamed: bool) -> str | None:
    """A note worth showing above the draft — never a reason to reject it."""
    if renamed:
        return (
            f"The Provider's name {raw_name!r} was rewritten as {question.name!r} "
            "because a Label Column name is snake_case."
        )
    if isinstance(question, Choice):
        undescribed = [key for key, text in question.criteria.items() if not text]
        if undescribed:
            return (
                "These options came back without a description: "
                + ", ".join(undescribed)
                + ". Describe each option so Jev applies it consistently."
            )
    if isinstance(question, Score) and len(question.levels) == 2:
        return (
            "The scale has only two levels; a Noul (yes/no) or Choice question "
            "usually expresses that better than a Score."
        )
    return None


def draft_jev_question(
    provider: Provider,
    description: str,
    context: str | None = None,
    sample_rows: list[dict[str, Any]] | None = None,
) -> tuple[JevQuestion, str | None]:
    """Ask the Provider for a draft Jev Question.

    Returns ``(question, warning)`` where ``question`` is a real
    :class:`~dat_distiller.jev` Noul/Choice/Score ready to edit, and
    ``warning`` is a note for the UI (or ``None``). Raises ``ValueError`` with
    a readable message when the draft is semantically unusable.
    """
    prompt = build_draft_prompt(description, context, sample_rows)
    payload = provider.complete(prompt, DRAFT_QUESTION_SCHEMA)
    if not isinstance(payload, Mapping):
        # ValueError, not TypeError: a malformed reply is the Provider's fault,
        # and the API maps ValueError to a 422 the user can act on.
        raise ValueError("The Provider's draft was not a JSON object")  # noqa: TRY004

    kind = str(payload.get("question_type") or payload.get("type") or payload.get("kind") or "").strip().lower()
    raw_name = str(payload.get("name") or "").strip()
    instructions = _description(payload.get("instructions"))
    if kind not in QUESTION_TYPES:
        raise ValueError(
            f"The Provider picked an unknown Jev Question type {kind!r}; expected one of {', '.join(QUESTION_TYPES)}"
        )
    if not raw_name:
        raise ValueError("The Provider's draft had no question name")
    if not instructions:
        raise ValueError("The Provider's draft had no instructions for Jev to follow")

    name = _snake_case(raw_name)
    if kind == "noul":
        question: JevQuestion = Noul(name=name, instructions=instructions)
    elif kind == "choice":
        criteria = _criteria(payload.get("criteria"))
        if not criteria or len(criteria) < 2:
            raise ValueError(
                "A Choice Jev Question needs at least two options to choose between; the Provider's draft "
                f"described {len(criteria or {})}."
            )
        question = Choice(name=name, instructions=instructions, criteria=criteria)
    else:
        levels = _levels(payload.get("levels"))
        if not levels or len(levels) < 2:
            raise ValueError(
                "A Score Jev Question needs an ordinal scale of at least two levels, lowest to highest; the "
                f"Provider's draft described {len(levels or [])}."
            )
        question = Score(name=name, instructions=instructions, levels=levels)

    return question, _warning(question, raw_name, renamed=name != raw_name)


def question_payload(question: JevQuestion) -> dict[str, Any]:
    """The draft as the API returns it (ready for the UI's question editor)."""
    out: dict[str, Any] = {
        "type": question_kind(question),
        "name": question.name,
        "instructions": question.instructions,
    }
    if isinstance(question, Choice):
        out["criteria"] = dict(question.criteria)
    elif isinstance(question, Score):
        out["levels"] = list(question.levels)
    return out


class DraftJevQuestionIn(BaseModel):
    description: str = Field(min_length=3, max_length=4000)
    context: str | None = Field(default=None, max_length=4000)
    sample_rows: list[dict[str, Any]] | None = Field(default=None, max_length=5)
    provider: str | None = None
    model: str | None = None
