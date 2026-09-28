"""Provider-drafted Jev Questions (drafting only, never Labeling)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dat_distiller.jev import Choice, Noul, Score, question_kind
from dat_distiller.jev_suggest import (
    DRAFT_QUESTION_SCHEMA,
    draft_jev_question,
    question_payload,
)
from dat_distiller.providers import FakeProvider

NOUL = {
    "question_type": "noul",
    "name": "needs_human_review",
    "instructions": "Answer yes when the ticket clearly asks for money back; otherwise answer no.",
}
CHOICE = {
    "question_type": "choice",
    "name": "ticket_topic",
    "instructions": "Pick the single topic that best matches the customer's main request in this ticket.",
    "criteria": [
        {"key": "billing", "description": "about invoices, charges or refunds"},
        {"key": "bug", "description": "something does not work as documented"},
        {"key": "feature_request", "description": "asking for something new"},
    ],
}
SCORE = {
    "question_type": "score",
    "name": "review_quality",
    "instructions": "Grade how useful this review is for a buyer deciding whether to purchase.",
    "levels": ["1 - empty", "3 - usable", "5 - decisive"],
}


def scripted(payload):
    return FakeProvider(response_for=lambda prompt, schema: payload)


def test_each_question_type_comes_back_as_its_dataclass() -> None:
    noul, warning = draft_jev_question(scripted(NOUL), "flag tickets needing a human")
    assert isinstance(noul, Noul) and question_kind(noul) == "noul"
    assert noul.name == "needs_human_review"
    assert "money back" in noul.instructions
    assert warning is None

    choice, warning = draft_jev_question(scripted(CHOICE), "categorize support tickets")
    assert isinstance(choice, Choice) and question_kind(choice) == "choice"
    assert choice.instructions.startswith("Pick the single topic")
    assert choice.criteria == {
        "billing": "about invoices, charges or refunds",
        "bug": "something does not work as documented",
        "feature_request": "asking for something new",
    }
    assert warning is None

    score, warning = draft_jev_question(scripted(SCORE), "grade review usefulness")
    assert isinstance(score, Score) and question_kind(score) == "score"
    assert score.levels == ["1 - empty", "3 - usable", "5 - decisive"]
    assert warning is None


def test_prompt_carries_description_context_and_sample_rows() -> None:
    fake = scripted(CHOICE)
    draft_jev_question(
        fake,
        "a column that says whether this support ticket is about billing",
        context="Rows come from a SaaS help desk; tickets are in English.",
        sample_rows=[{"subject": "double charged", "body": "I paid twice this month"}],
    )
    prompt = fake.calls[0]["prompt"]
    assert "whether this support ticket is about billing" in prompt
    assert "SaaS help desk" in prompt
    assert "double charged" in prompt
    # the Provider must be told how to pick a type and how to name the question
    assert "noul" in prompt and "choice" in prompt and "score" in prompt
    assert "snake_case" in prompt
    schema = fake.calls[0]["schema"]
    assert schema["properties"]["question_type"]["enum"] == ["noul", "choice", "score"]
    assert schema["required"] == ["question_type", "name", "instructions"]


def test_default_fake_provider_synthesis_yields_a_usable_draft() -> None:
    # No scripting: synthesize() takes the first enum value, so the default
    # draft is a Noul named "s" with the instruction "s". Anything other than
    # that Noul (or a plain ValueError) would be a bug.
    question, warning = draft_jev_question(FakeProvider(), "any Label Column description")
    assert isinstance(question, Noul)
    assert question_kind(question) == "noul"
    assert question.name == "s" and question.instructions == "s"
    assert warning is None


def test_names_are_snake_cased_with_a_warning() -> None:
    payload = {**CHOICE, "name": "Ticket Topic!"}
    question, warning = draft_jev_question(scripted(payload), "categorize tickets")
    assert question.name == "ticket_topic"
    assert warning and "ticket_topic" in warning


def test_criteria_written_as_a_map_still_work() -> None:
    payload = {
        "question_type": "choice",
        "name": "sentiment",
        "instructions": "Pick the overall sentiment of the review.",
        "criteria": {"positive": "praise", "negative": "complaint"},
    }
    question, warning = draft_jev_question(scripted(payload), "review sentiment")
    assert isinstance(question, Choice)
    assert question.criteria == {"positive": "praise", "negative": "complaint"}
    assert warning is None


def test_missing_option_descriptions_are_reported_as_a_warning() -> None:
    payload = {**CHOICE, "criteria": [{"key": "billing"}, {"key": "bug"}]}
    question, warning = draft_jev_question(scripted(payload), "categorize tickets")
    assert isinstance(question, Choice) and len(question.criteria) == 2
    assert warning and "billing" in warning


def test_score_with_two_levels_is_allowed_with_a_warning() -> None:
    payload = {**SCORE, "levels": ["low", "high"]}
    question, warning = draft_jev_question(scripted(payload), "risk band")
    assert isinstance(question, Score) and question.levels == ["low", "high"]
    assert warning and "two levels" in warning


@pytest.mark.parametrize(
    "payload, needle",
    [
        ({**CHOICE, "criteria": [{"key": "billing"}]}, "at least two options"),
        ({**CHOICE, "criteria": []}, "at least two options"),
        ({**CHOICE, "criteria": None}, "at least two options"),
        ({**SCORE, "levels": ["only one"]}, "at least two levels"),
        ({**SCORE, "levels": []}, "at least two levels"),
    ],
)
def test_semantically_broken_drafts_raise_value_error(payload, needle) -> None:
    with pytest.raises(ValueError) as excinfo:
        draft_jev_question(scripted(payload), "whatever description")
    assert needle in str(excinfo.value)


def test_out_of_schema_reply_is_rejected_before_our_own_check() -> None:
    # The schema enum already refuses an unknown type, so it never reaches the
    # semantic guard below — the Provider layer is the first line of defence.
    from dat_distiller.providers.base import InvalidStructuredOutputError

    with pytest.raises(InvalidStructuredOutputError):
        draft_jev_question(scripted({"question_type": "boolean"}), "whatever description")


def test_semantic_guard_still_catches_an_unknown_type() -> None:
    # Defence in depth for a Provider that ignores the requested schema: an
    # unknown type must not become a question.
    class Loose:
        def complete(self, prompt, schema):
            return {"question_type": "boolean", "name": "flag", "instructions": "x"}

    with pytest.raises(ValueError, match="unknown Jev Question type"):
        draft_jev_question(Loose(), "whatever description")


def test_endpoint_drafts_a_choice_question(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "dat_distiller.api.jev.resolve_provider",
        lambda settings, provider=None, model=None: scripted(CHOICE),
    )
    r = client.post(
        "/api/jev/draft-question",
        json={
            "description": "which topic this ticket belongs to",
            "context": "SaaS help desk",
            "sample_rows": [{"subject": "invoice"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "choice"
    assert body["question"]["type"] == "choice"
    assert body["question"]["name"] == "ticket_topic"
    assert body["question"]["criteria"]["bug"].startswith("something does not work")
    assert body["question"].get("levels") is None  # a Choice carries no scale
    assert body["warning"] is None
    # request body validation
    assert client.post("/api/jev/draft-question", json={"description": "ab"}).status_code == 422
    assert (
        client.post(
            "/api/jev/draft-question",
            json={"description": "long enough", "sample_rows": [{"a": i} for i in range(6)]},
        ).status_code
        == 422
    )


def test_endpoint_maps_semantic_and_provider_errors_to_422(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "dat_distiller.api.jev.resolve_provider",
        lambda settings, provider=None, model=None: scripted({**CHOICE, "criteria": [{"key": "solo"}]}),
    )
    r = client.post("/api/jev/draft-question", json={"description": "categorize tickets"})
    assert r.status_code == 422 and "at least two options" in r.text

    from dat_distiller.providers.base import ProviderNotConfiguredError

    def unconfigured(settings, provider=None, model=None):
        raise ProviderNotConfiguredError("OpenRouter needs an API key")

    monkeypatch.setattr("dat_distiller.api.jev.resolve_provider", unconfigured)
    r = client.post("/api/jev/draft-question", json={"description": "categorize tickets"})
    assert r.status_code == 422 and "API key" in r.text


def test_endpoint_maps_timeout_and_bad_output(client: TestClient, monkeypatch) -> None:
    from dat_distiller.providers.base import (
        InvalidStructuredOutputError,
        ProviderTimeoutError,
    )

    def slow(settings, provider=None, model=None):
        raise ProviderTimeoutError("the Provider took too long")

    monkeypatch.setattr("dat_distiller.api.jev.resolve_provider", slow)
    assert client.post("/api/jev/draft-question", json={"description": "anything"}).status_code == 504

    def junk(settings, provider=None, model=None):
        raise InvalidStructuredOutputError("nope", ["name: is required"])

    monkeypatch.setattr("dat_distiller.api.jev.resolve_provider", junk)
    r = client.post("/api/jev/draft-question", json={"description": "anything"})
    assert r.status_code == 502 and "is required" in r.text


def test_question_payload_is_what_the_ui_edits() -> None:
    score, _ = draft_jev_question(scripted(SCORE), "grade review usefulness")
    assert question_payload(score) == {
        "type": "score",
        "name": "review_quality",
        "instructions": SCORE["instructions"],
        "levels": SCORE["levels"],
    }
    assert "criteria" not in DRAFT_QUESTION_SCHEMA["properties"]["criteria"].get("required", [])
