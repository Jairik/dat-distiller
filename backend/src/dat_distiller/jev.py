"""Jev (TypeSafe's hosted evaluation model) behind a small interface.

Wraps ``typesafe_sdk.TypeSafeClient.system_one`` so that **all** of a row's
Jev Questions go out in a **single call**. Also ships :class:`FakeJev` so no
test ever touches the real API, and a deterministic **State** serializer that
turns selected columns into ``column: value`` lines.

Normalized output (what ``answer_row`` returns, keyed by question name) —
common shape across all three kinds::

    Noul   -> {"kind": "noul",   "answer": "yes"|"no",
               "confidence": float 0..1, "probabilities": {"yes": p, "no": 1-p}}
    Choice -> {"kind": "choice", "answer": <option>,
               "confidence": float, "probabilities": {option: p, ...}}
    Score  -> {"kind": "score",  "answer": float (expected score),
               "confidence": float, "probabilities": {level: p, ...},
               "levels": [..], "argmax": <level>}

Noul confidence is ``abs(p_yes - 0.5) * 2`` — distance from a coin flip, so
uncertain answers score low (the Review Queue in #28 relies on this).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

#: When set, resolve_jev returns FakeJev (e2e suite / demos).
FAKE_JEV_ENV = "DAT_DISTILLER_FAKE_JEV"


class JevError(RuntimeError):
    pass


class JevNotConfiguredError(JevError):
    """No TypeSafe API key."""


@dataclass(frozen=True)
class Noul:
    name: str
    instructions: str


@dataclass(frozen=True)
class Choice:
    name: str
    instructions: str
    criteria: dict[str, str | None]


@dataclass(frozen=True)
class Score:
    name: str
    instructions: str
    levels: list[str]  # ordered, lowest -> highest


JevQuestion = Noul | Choice | Score


def question_kind(question: JevQuestion) -> str:
    return {Noul: "noul", Choice: "choice", Score: "score"}[type(question)]


def serialize_state(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    """Render one row's **State** as ``column: value`` lines.

    Deterministic: columns appear in the given order; newlines inside values
    are collapsed so one row stays one logical block; missing values are empty.
    """
    lines: list[str] = []
    for column in columns:
        value = row.get(column)
        if value is None:
            text = ""
        else:
            text = str(value)
        text = " ".join(text.split())  # collapse newlines/tabs/spaces
        lines.append(f"{column}: {text}".rstrip())
    return "\n".join(lines)


class Jev(Protocol):
    def answer_row(self, state: str, questions: Sequence[JevQuestion]) -> dict[str, dict[str, Any]]:
        """Answer every question for one State in a single Jev call."""
        ...


class TypeSafeJev:
    """Real Jev client. ``client`` is injectable for tests (duck-typed)."""

    def __init__(self, api_key: str | None = None, model: str | None = None, *, client: Any | None = None) -> None:
        if client is None:
            from typesafe_sdk import TypeSafeClient

            if not api_key:
                raise JevNotConfiguredError("Jev needs a TypeSafe API key (TYPESAFE_API_KEY)")
            client = TypeSafeClient(api_key=api_key, model=model or "system-one")
        self._client = client

    def answer_row(self, state: str, questions: Sequence[JevQuestion]) -> dict[str, dict[str, Any]]:
        from typesafe_sdk import Choice as SDKChoice
        from typesafe_sdk import Noul as SDKNoul
        from typesafe_sdk import Score as SDKScore

        sdk_questions: dict[str, Any] = {}
        for q in questions:
            if isinstance(q, Noul):
                sdk_questions[q.name] = SDKNoul(instructions=q.instructions)
            elif isinstance(q, Choice):
                sdk_questions[q.name] = SDKChoice(instructions=q.instructions, criteria=q.criteria)
            elif isinstance(q, Score):
                sdk_questions[q.name] = SDKScore(instructions=q.instructions, criteria=list(q.levels))
        response = self._client.system_one(state={"state": state}, questions=sdk_questions)
        answers = getattr(response, "answers", None)
        if answers is None:
            raise JevError("Jev response had no answers")
        return {q.name: _normalize(q, answers[q.name]) for q in questions}


def _normalize(question: JevQuestion, answer: Any) -> dict[str, Any]:
    kind = question_kind(question)
    if kind == "noul":
        p_yes = float(answer.noul)
        return {
            "kind": "noul",
            "answer": "yes" if p_yes >= 0.5 else "no",
            "confidence": round(abs(p_yes - 0.5) * 2, 6),
            "probabilities": {"yes": p_yes, "no": round(1 - p_yes, 6)},
        }
    if kind == "choice":
        probs = {str(k): float(v) for k, v in dict(answer.probabilities).items()}
        return {
            "kind": "choice",
            "answer": str(answer.choice),
            "confidence": float(answer.confidence),
            "probabilities": probs,
        }
    # score
    levels = list(question.levels)  # type: ignore[union-attr]
    probs = {str(k): float(v) for k, v in dict(answer.probabilities).items()}
    argmax = max(probs, key=probs.get) if probs else (levels[0] if levels else "")
    return {
        "kind": "score",
        "answer": float(answer.score),
        "confidence": float(answer.confidence),
        "probabilities": probs,
        "levels": levels,
        "argmax": argmax,
    }


class FakeJev:
    """Deterministic Jev for tests and the e2e suite — never touches the API.

    Answers are a stable function of (state, question name, kind), so the same
    input always yields the same labels. ``low_confidence=True`` forces
    near-coin-flip confidences (to exercise the Review Queue). An optional
    ``script`` maps ``"<state>::<question>"`` to a full answer dict to pin
    specific rows in a test.
    """

    def __init__(self, *, low_confidence: bool = False, script: dict[str, dict[str, Any]] | None = None) -> None:
        self.low_confidence = low_confidence
        self.script = script or {}
        self.calls: list[dict[str, Any]] = []

    def _rand(self, state: str, name: str) -> float:
        digest = hashlib.sha256(f"{state}::{name}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / (2**64)

    def answer_row(self, state: str, questions: Sequence[JevQuestion]) -> dict[str, dict[str, Any]]:
        self.calls.append({"state": state, "questions": [q.name for q in questions]})
        out: dict[str, dict[str, Any]] = {}
        for q in questions:
            pinned = self.script.get(f"{state}::{q.name}")
            out[q.name] = pinned if pinned else self._synthesize(state, q)
        return out

    def _synthesize(self, state: str, q: JevQuestion) -> dict[str, Any]:
        kind = question_kind(q)
        r = self._rand(state, q.name)
        lo = self.low_confidence
        if kind == "noul":
            p = 0.5 + (0.02 if lo else (r - 0.5) * 1.8)  # lo -> near 0.5
            p = min(0.999, max(0.001, p))
            return {
                "kind": "noul",
                "answer": "yes" if p >= 0.5 else "no",
                "confidence": round(abs(p - 0.5) * 2, 6),
                "probabilities": {"yes": round(p, 6), "no": round(1 - p, 6)},
            }
        if kind == "choice":
            keys = list(q.criteria)  # type: ignore[union-attr]
            weights = [self._rand(state, f"{q.name}:{k}") + 0.01 for k in keys]
            total = sum(weights)
            probs = {k: round(w / total, 6) for k, w in zip(keys, weights)}
            best = max(probs, key=probs.get)
            conf = 0.34 if lo else probs[best]
            return {"kind": "choice", "answer": best, "confidence": round(conf, 6), "probabilities": probs}
        levels = list(q.levels)  # type: ignore[union-attr]
        idx = int(r * len(levels)) % len(levels)
        if lo:  # spread the mass so no level dominates
            probs = {lvl: round(1 / len(levels), 6) for lvl in levels}
        else:
            probs = {lvl: 0.02 for lvl in levels}
            probs[levels[idx]] = 0.92
            s = sum(probs.values())
            probs = {k: round(v / s, 6) for k, v in probs.items()}
        argmax = levels[idx]
        return {
            "kind": "score",
            "answer": float(idx),
            "confidence": 0.34 if lo else 0.9,
            "probabilities": probs,
            "levels": levels,
            "argmax": argmax,
        }


def resolve_jev(settings_store, model: str | None = None):
    """The Jev client for this deployment: fake under the env switch, else
    real (which requires a TypeSafe key, env or stored)."""
    if os.environ.get(FAKE_JEV_ENV):
        return FakeJev()
    key = settings_store.api_key("typesafe")
    if not key:
        raise JevNotConfiguredError(
            "Labeling needs a TypeSafe API key (Settings or TYPESAFE_API_KEY)"
        )
    chosen = model or settings_store.load().get("models", {}).get("jev")
    return TypeSafeJev(api_key=key, model=chosen)
