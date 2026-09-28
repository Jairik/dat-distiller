"""Jev: State serialization, normalization, FakeJev, single-call multi-question."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dat_distiller.jev import (
    FAKE_JEV_ENV,
    Choice,
    FakeJev,
    JevNotConfiguredError,
    Noul,
    Score,
    TypeSafeJev,
    resolve_jev,
    serialize_state,
)
from dat_distiller.settings import SettingsStore


def test_state_serialization_is_deterministic_and_collapses_whitespace() -> None:
    row = {"subject": "Re:  hi\nthere", "body": "  lots\n\nof \t space ", "num": 7, "none": None}
    a = serialize_state(row, ["subject", "body", "num", "none"])
    b = serialize_state(row, ["subject", "body", "num", "none"])
    assert a == b
    assert a.splitlines() == [
        "subject: Re: hi there",
        "body: lots of space",
        "num: 7",
        "none:",
    ]
    # column selection + order drives the output
    assert serialize_state(row, ["num", "subject"]) == "num: 7\nsubject: Re: hi there"


def test_fake_jev_is_deterministic_and_offline() -> None:
    questions = [
        Noul("spam", "Is this spam?"),
        Choice("sentiment", "Tone?", {"pos": "positive", "neg": "negative"}),
        Score("quality", "How good?", ["bad", "ok", "great"]),
    ]
    a = FakeJev().answer_row("body: hello", questions)
    b = FakeJev().answer_row("body: hello", questions)
    assert a == b
    assert a["spam"]["kind"] == "noul" and a["spam"]["answer"] in ("yes", "no")
    assert a["sentiment"]["answer"] in ("pos", "neg")
    assert abs(sum(a["sentiment"]["probabilities"].values()) - 1.0) < 1e-4
    assert a["quality"]["levels"] == ["bad", "ok", "great"]
    # different states can differ
    assert FakeJev().answer_row("body: totally different", questions) != a


def test_fake_jev_all_three_types_single_call_and_low_confidence() -> None:
    questions = [
        Noul("a", "?"),
        Choice("b", "?", {"x": None, "y": None, "z": None}),
        Score("c", "?", ["low", "high"]),
    ]
    fake = FakeJev()
    fake.answer_row("s", questions)
    assert len(fake.calls) == 1  # all three questions went out together
    low = FakeJev(low_confidence=True).answer_row("s", questions)
    assert low["a"]["confidence"] < 0.1
    assert low["b"]["confidence"] < 0.5


def test_fake_jev_script_overrides() -> None:
    fake = FakeJev(script={"state text::q": {"kind": "noul", "answer": "yes", "confidence": 0.99, "probabilities": {"yes": 0.99, "no": 0.01}}})
    out = fake.answer_row("state text", [Noul("q", "?")])
    assert out["q"]["confidence"] == 0.99


class StubClient:
    """Captures the one system_one call and returns canned SDK-shaped answers."""

    def __init__(self):
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append({"state": state, "questions": questions})
        answers = {}
        for name, q in questions.items():
            if q.type == "noul":
                answers[name] = SimpleNamespace(noul=0.85)
            elif q.type == "choice":
                answers[name] = SimpleNamespace(
                    choice="pos", confidence=0.7, probabilities={"pos": 0.7, "neg": 0.3}
                )
            else:
                answers[name] = SimpleNamespace(
                    score=1.5, confidence=0.6, probabilities={"bad": 0.2, "ok": 0.5, "great": 0.3}
                )
        return SimpleNamespace(answers=answers)


def test_typesafe_jev_multi_question_single_call_and_normalization() -> None:
    stub = StubClient()
    jev = TypeSafeJev(client=stub)
    out = jev.answer_row(
        "body: hi",
        [
            Noul("spam", "?"),
            Choice("tone", "?", {"pos": None, "neg": None}),
            Score("quality", "?", ["bad", "ok", "great"]),
        ],
    )
    assert len(stub.calls) == 1
    assert set(stub.calls[0]["questions"]) == {"spam", "tone", "quality"}
    assert stub.calls[0]["state"] == {"state": "body: hi"}
    assert out["spam"] == {
        "kind": "noul",
        "answer": "yes",
        "confidence": 0.7,  # abs(0.85-0.5)*2
        "probabilities": {"yes": 0.85, "no": 0.15},
    }
    assert out["tone"]["answer"] == "pos" and out["tone"]["confidence"] == 0.7
    assert out["quality"]["answer"] == 1.5 and out["quality"]["argmax"] == "ok"


def test_real_client_requires_key() -> None:
    with pytest.raises(JevNotConfiguredError):
        TypeSafeJev(api_key=None)


def test_resolve_jev(isolated_data_dir, monkeypatch) -> None:
    store = SettingsStore(isolated_data_dir.parent / "config")
    monkeypatch.delenv(FAKE_JEV_ENV, raising=False)
    with pytest.raises(JevNotConfiguredError):
        resolve_jev(store)
    store.set_key("typesafe", "ts-key")
    assert isinstance(resolve_jev(store), TypeSafeJev)
    monkeypatch.setenv(FAKE_JEV_ENV, "1")
    assert isinstance(resolve_jev(store), FakeJev)
