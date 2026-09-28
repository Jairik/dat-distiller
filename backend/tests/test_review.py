"""The Review Queue: what gets queued, and what a decision does to the data.

The behaviours that matter and are easy to get wrong:

- **Confidence has to mean the same thing for all three question types**, or a
  single threshold silently exempts Noul labels from review forever.
- **An override must rewrite the sibling probabilities.** Replacing a Choice
  answer while leaving ``__p_pos`` at 0.9 is how a corrected label turns into a
  fresh leak the moment it becomes a feature.
- **Accepting is not a no-op.** Nothing about the data changes, but the
  Provenance does: a person looked and agreed is a different fact from
  "Jev said this and nobody checked".
- **The source version is immutable** — every decision produces a child.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dat_distiller.review import (
    DECISIONS,
    EXCLUDED_COLUMN,
    REVIEWED_ORIGIN,
    UNREVIEWED_CHECK_KIND,
    Queue,
    QueueItem,
    apply_decisions,
    apply_review,
    build_queue,
    confidence_of,
    label_families,
    noul_confidence,
    register_unreviewed_check,
    validate_decisions,
)
from dat_distiller.store.provenance import PROVENANCE_COLUMN


def prov(family: str, origin: str, confidence: float | None = None) -> dict:
    entry: dict = {"origin": origin}
    if confidence is not None:
        entry["confidence"] = confidence
    return json.dumps({"row_origin": "synthetic", "label_origins": {family: entry}})


def multi_prov(confidences: dict[str, float]) -> str:
    """Provenance for a row Jev labeled with every question at once."""
    return json.dumps(
        {
            "row_origin": "synthetic",
            "label_origins": {
                family: {"origin": "jev", "confidence": confidence}
                for family, confidence in confidences.items()
            },
        }
    )


# -- a version with one of each question type ---------------------------------

@pytest.fixture()
def labeled() -> pd.DataFrame:
    """One Noul, one Choice and one Score Label Column, with known confidences."""
    return pd.DataFrame(
        {
            "text": ["a", "b", "c", "d"],
            "is_spam": pd.array([True, False, True, False], dtype="boolean"),
            "is_spam__confidence": pd.array([0.95, 0.55, 0.99, 0.52], dtype="Float64"),
            "tone": ["pos", "pos", "neg", "neg"],
            "tone__confidence": pd.array([0.9, 0.6, 0.95, 0.4], dtype="Float64"),
            "tone__p_pos": pd.array([0.9, 0.6, 0.05, 0.3], dtype="Float64"),
            "tone__p_neg": pd.array([0.1, 0.4, 0.95, 0.7], dtype="Float64"),
            "quality": pd.array([5.0, 3.0, 1.0, 2.0], dtype="Float64"),
            "quality__confidence": pd.array([0.8, 0.5, 0.9, 0.3], dtype="Float64"),
            "quality__probabilities": [
                json.dumps({"1": 0.2, "5": 0.8}),
                json.dumps({"3": 0.5, "5": 0.5}),
                json.dumps({"1": 0.9, "5": 0.1}),
                json.dumps({"1": 0.7, "5": 0.3}),
            ],
            PROVENANCE_COLUMN: [
                multi_prov({"is_spam": 0.95, "tone": 0.9, "quality": 0.8}),
                multi_prov({"is_spam": 0.55, "tone": 0.6, "quality": 0.5}),
                multi_prov({"is_spam": 0.99, "tone": 0.95, "quality": 0.9}),
                multi_prov({"is_spam": 0.52, "tone": 0.4, "quality": 0.3}),
            ],
        }
    )


# -- confidence ---------------------------------------------------------------


def test_noul_confidence_is_rescaled_from_a_coin_flip() -> None:
    # a fully committed answer reads as 1.0, not 0.5, so one threshold works
    # for all three question types
    assert noul_confidence(1.0) == 1.0
    assert noul_confidence(0.0) == 1.0
    assert noul_confidence(0.5) == 0.0
    assert noul_confidence(0.8) == pytest.approx(0.6)
    assert noul_confidence(0.55) == pytest.approx(0.1)
    assert noul_confidence(None) is None
    assert noul_confidence("nonsense") is None


def test_confidence_is_the_top_option_for_choice(labeled: pd.DataFrame) -> None:
    assert confidence_of(labeled.iloc[[0]], "tone", "choice") == pytest.approx(0.9)
    assert confidence_of(labeled.iloc[[2]], "tone", "choice") == pytest.approx(0.95)


def test_confidence_is_the_top_level_for_score(labeled: pd.DataFrame) -> None:
    assert confidence_of(labeled.iloc[[0]], "quality", "score") == pytest.approx(0.8)
    assert confidence_of(labeled.iloc[[2]], "quality", "score") == pytest.approx(0.9)


def test_confidence_of_a_missing_sibling_is_none_not_zero(labeled: pd.DataFrame) -> None:
    frame = labeled.copy()
    frame.loc[0, "tone__p_pos"] = pd.NA
    frame.loc[0, "tone__p_neg"] = pd.NA
    assert confidence_of(frame.iloc[[0]], "tone", "choice") is None


def test_label_families_are_read_off_the_column_set(labeled: pd.DataFrame) -> None:
    assert label_families(labeled) == {"is_spam": "noul", "tone": "choice", "quality": "score"}


def test_an_ordinary_column_is_not_a_label_family(labeled: pd.DataFrame) -> None:
    frame = labeled.copy()
    frame["notes__confidence"] = pd.array([0.1, 0.1, 0.1, 0.1], dtype="Float64")
    assert "notes" not in label_families(frame)


def test_a_family_is_recognised_from_any_one_sibling(labeled: pd.DataFrame) -> None:
    # a Choice is a Choice because it has __p_* columns, whether or not a
    # __confidence column happens to sit next to them
    frame = labeled.drop(columns=["tone__confidence"])
    assert label_families(frame)["tone"] == "choice"
    frame = labeled.drop(columns=["tone__p_pos", "tone__p_neg"])
    assert label_families(frame)["tone"] == "noul"  # a bare __confidence is a Noul


# -- the queue itself ---------------------------------------------------------


# The fixture's confidences, for reference (see the assertions below):
#   row 0: noul 0.90  choice 0.90  score 0.80   -> all clear of 0.8
#   row 1: noul 0.10  choice 0.60  score 0.50   -> all below
#   row 2: noul 0.98  choice 0.95  score 0.90   -> all clear of 0.8
#   row 3: noul 0.04  choice 0.70  score 0.70   -> all below
THRESHOLD = 0.8


def test_only_labels_below_the_threshold_are_queued(labeled: pd.DataFrame) -> None:
    queue = build_queue(labeled, "v1", threshold=THRESHOLD)
    assert {(i.row_index, i.family) for i in queue.items} == {
        (1, "is_spam"),
        (1, "tone"),
        (1, "quality"),
        (3, "is_spam"),
        (3, "tone"),
        (3, "quality"),
    }


def test_the_queue_grows_as_the_threshold_drops(labeled: pd.DataFrame) -> None:
    counts = [
        len(build_queue(labeled, "v1", threshold=t).items)
        for t in (0.99, 0.9, THRESHOLD, 0.6, 0.1, 0.0)
    ]
    # a higher bar can only ever mean more work
    assert counts == sorted(counts, reverse=True)
    assert counts == [12, 8, 6, 3, 1, 0]


def test_the_queue_holds_only_the_genuinely_unsure_labels(labeled: pd.DataFrame) -> None:
    # rows 0 and 2 are confident on all three questions; rows 1 and 3 are not
    assert {i.row_index for i in build_queue(labeled, "v1", THRESHOLD).items} == {1, 3}


def test_an_accepted_label_does_not_come_back_to_the_queue(labeled: pd.DataFrame) -> None:
    # accepting leaves the confidence untouched, so the only thing keeping the
    # label out of the queue is the Provenance saying a human resolved it
    decisions = validate_decisions(
        [{"row_index": 1, "family": "is_spam", "question_type": "noul", "decision": "accept"}]
    )
    out, _ = apply_decisions(labeled, decisions, label_families(labeled))
    still_queued = {(i.row_index, i.family) for i in build_queue(out, "v2", THRESHOLD).items}
    assert (1, "is_spam") not in still_queued
    # ...while the row's other unreviewed labels are still there
    assert (1, "tone") in still_queued


def test_an_overridden_label_does_not_come_back_either(labeled: pd.DataFrame) -> None:
    decisions = validate_decisions(
        [
            {
                "row_index": 3,
                "family": "tone",
                "question_type": "choice",
                "decision": "override",
                "override": "pos",
            }
        ]
    )
    out, _ = apply_decisions(labeled, decisions, label_families(labeled))
    assert not any(i.family == "tone" and i.row_index == 3 for i in build_queue(out, "v2", THRESHOLD).items)


def test_is_reviewed_reads_the_provenance(labeled: pd.DataFrame) -> None:
    from dat_distiller.review import is_reviewed

    assert is_reviewed(labeled, 0, "is_spam") is False
    decisions = validate_decisions(
        [{"row_index": 0, "family": "is_spam", "question_type": "noul", "decision": "accept"}]
    )
    out, _ = apply_decisions(labeled, decisions, label_families(labeled))
    assert is_reviewed(out, 0, "is_spam") is True
    assert is_reviewed(out, 1, "is_spam") is False


def test_a_noul_is_reviewable_at_the_same_threshold_as_the_others(labeled: pd.DataFrame) -> None:
    # the bug this guards: a Noul rescaled wrongly (reported as 0.5 rather than
    # 1.0 when committed) would never reach a 0.8 bar and would be silently
    # exempt from human review for every setting
    queue = build_queue(labeled, "v1", threshold=THRESHOLD)
    nouls = {i.family for i in queue.items if i.question_type == "noul"}
    assert nouls == {"is_spam"}
    # ...and a committed one is not queued
    committed = build_queue(labeled.iloc[[0]], "v1", threshold=THRESHOLD)
    assert committed.items == []


def test_a_row_jev_never_answered_is_unlabeled_not_queued(labeled: pd.DataFrame) -> None:
    frame = labeled.copy()
    frame.loc[1, "tone"] = None
    frame.loc[1, "tone__confidence"] = pd.NA
    frame.loc[1, "tone__p_pos"] = pd.NA
    frame.loc[1, "tone__p_neg"] = pd.NA
    queue = build_queue(frame, "v1", threshold=THRESHOLD)
    assert {"row_index": 1, "family": "tone", "question_type": "choice"} in queue.unlabeled
    assert not any(i.row_index == 1 and i.family == "tone" for i in queue.items)


def test_unreviewed_count_covers_both_the_queue_and_the_gaps(labeled: pd.DataFrame) -> None:
    frame = labeled.copy()
    frame.loc[1, "tone"] = None
    frame.loc[1, "tone__confidence"] = pd.NA
    frame.loc[1, "tone__p_pos"] = pd.NA
    frame.loc[1, "tone__p_neg"] = pd.NA
    queue = build_queue(frame, "v1", threshold=THRESHOLD)
    assert queue.unreviewed_count == len(queue.items) + 1


def test_deciding_an_item_lowers_the_unreviewed_count(labeled: pd.DataFrame) -> None:
    queue = build_queue(labeled, "v1", threshold=THRESHOLD)
    before = queue.unreviewed_count
    queue.items[0].decision = "accept"
    assert queue.unreviewed_count == before - 1


def test_a_capped_queue_still_reports_the_whole_version(labeled: pd.DataFrame) -> None:
    full = build_queue(labeled, "v1", threshold=0.99)
    page = build_queue(labeled, "v1", threshold=0.99, rows=3)
    assert len(full.items) == 12
    assert len(page.items) == 3
    # a paged queue must never under-report what is outstanding
    assert page.unreviewed_count == full.unreviewed_count


def test_the_queue_serializes(labeled: pd.DataFrame) -> None:
    body = build_queue(labeled, "v1", threshold=THRESHOLD).to_dict()
    assert body["queued_count"] == 6
    assert body["threshold"] == THRESHOLD

    assert json.dumps(body)  # the API hands this straight to the browser
    assert body["items"][0]["family"] in {"is_spam", "tone", "quality"}


# -- decisions ----------------------------------------------------------------


def test_an_override_needs_a_new_answer() -> None:
    with pytest.raises(ValueError, match="needs a new answer"):
        validate_decisions([{"row_index": 0, "family": "tone", "decision": "override"}])


def test_an_unknown_decision_is_refused() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        validate_decisions([{"row_index": 0, "family": "tone", "decision": "delete"}])
    assert DECISIONS == ("accept", "override", "exclude")


def test_a_decision_must_name_its_label_column() -> None:
    with pytest.raises(ValueError, match="Label Column"):
        validate_decisions([{"row_index": 0, "family": "", "decision": "accept"}])


# -- accept -------------------------------------------------------------------


def test_accept_changes_the_data_by_nothing_and_the_provenance_by_everything(
    labeled: pd.DataFrame,
) -> None:
    decisions = validate_decisions(
        [{"row_index": 3, "family": "tone", "question_type": "choice", "decision": "accept"}]
    )
    out, outcome = apply_decisions(labeled, decisions, label_families(labeled))
    assert outcome.accepted == 1
    # the answer is untouched
    assert out["tone"].iloc[3] == labeled["tone"].iloc[3]
    assert out["tone__p_pos"].iloc[3] == labeled["tone__p_pos"].iloc[3]
    # but it is now a fact about a person, not about Jev
    entry = json.loads(out[PROVENANCE_COLUMN].iloc[3])["label_origins"]["tone"]
    assert entry["origin"] == REVIEWED_ORIGIN
    assert entry["jev_confidence"] == 0.4  # what Jev had said, kept for the record
    assert "confidence" not in entry  # superseded, not silently kept alongside
    # other rows are untouched
    assert json.loads(out[PROVENANCE_COLUMN].iloc[0])["label_origins"]["tone"]["origin"] == "jev"


def test_an_accept_can_carry_a_note(labeled: pd.DataFrame) -> None:
    decisions = validate_decisions(
        [
            {
                "row_index": 3,
                "family": "tone",
                "question_type": "choice",
                "decision": "accept",
                "note": "  checked against the transcript  ",
            }
        ]
    )
    out, _ = apply_decisions(labeled, decisions, label_families(labeled))
    entry = json.loads(out[PROVENANCE_COLUMN].iloc[3])["label_origins"]["tone"]
    assert entry["note"] == "checked against the transcript"


# -- override -----------------------------------------------------------------


def test_a_choice_override_rewrites_every_sibling_probability(labeled: pd.DataFrame) -> None:
    decisions = validate_decisions(
        [
            {
                "row_index": 3,
                "family": "tone",
                "question_type": "choice",
                "decision": "override",
                "override": "pos",
            }
        ]
    )
    out, outcome = apply_decisions(labeled, decisions, label_families(labeled))
    assert outcome.overridden == 1
    assert out["tone"].iloc[3] == "pos"
    # the old distribution would have been a leak waiting to happen
    assert out["tone__p_pos"].iloc[3] == 1.0
    assert out["tone__p_neg"].iloc[3] == 0.0
    assert out["tone__confidence"].iloc[3] == 1.0
    entry = json.loads(out[PROVENANCE_COLUMN].iloc[3])["label_origins"]["tone"]
    assert entry["origin"] == REVIEWED_ORIGIN


def test_a_score_override_rewrites_the_probability_map(labeled: pd.DataFrame) -> None:
    # a Score Label Column holds the expected score, so the reviewer overrides
    # with a number on the scale rather than a level's phrasing
    decisions = validate_decisions(
        [
            {
                "row_index": 3,
                "family": "quality",
                "question_type": "score",
                "decision": "override",
                "override": 5.0,
            }
        ]
    )
    out, _ = apply_decisions(labeled, decisions, label_families(labeled))
    assert json.loads(out["quality__probabilities"].iloc[3]) == {"5": 1.0}
    assert out["quality"].iloc[3] == 5.0
    # the column must stay a Float64, or it stops being a valid Target
    assert isinstance(out["quality"].dtype, pd.Float64Dtype)


def test_a_score_override_with_a_level_name_is_refused(labeled: pd.DataFrame) -> None:
    # writing "5 - decisive" into a Float64 column would raise deep in pandas;
    # failing here names the actual problem
    decisions = validate_decisions(
        [
            {
                "row_index": 3,
                "family": "quality",
                "question_type": "score",
                "decision": "override",
                "override": "5 - decisive",
            }
        ]
    )
    with pytest.raises(ValueError, match="needs a number on the scale"):
        apply_decisions(labeled, decisions, label_families(labeled))


def test_a_noul_override_sets_a_certain_confidence(labeled: pd.DataFrame) -> None:
    decisions = validate_decisions(
        [
            {
                "row_index": 3,
                "family": "is_spam",
                "question_type": "noul",
                "decision": "override",
                "override": True,
            }
        ]
    )
    out, _ = apply_decisions(labeled, decisions, label_families(labeled))
    assert bool(out["is_spam"].iloc[3]) is True
    assert out["is_spam__confidence"].iloc[3] == 1.0


def test_an_override_keeps_the_answer_nullable(labeled: pd.DataFrame) -> None:
    decisions = validate_decisions(
        [
            {
                "row_index": 3,
                "family": "is_spam",
                "question_type": "noul",
                "decision": "override",
                "override": True,
            }
        ]
    )
    out, _ = apply_decisions(labeled, decisions, label_families(labeled))
    # a boolean Label Column must stay boolean, or it stops being a Target
    assert isinstance(out["is_spam"].dtype, pd.BooleanDtype)


def test_overriding_a_column_that_is_not_a_label_is_refused(labeled: pd.DataFrame) -> None:
    decisions = validate_decisions(
        [{"row_index": 0, "family": "text", "question_type": "noul", "decision": "accept"}]
    )
    with pytest.raises(ValueError, match="not a Label Column"):
        apply_decisions(labeled, decisions, label_families(labeled))


def test_the_input_frame_is_never_mutated(labeled: pd.DataFrame) -> None:
    before = labeled.copy(deep=True)
    decisions = validate_decisions(
        [
            {
                "row_index": 3,
                "family": "tone",
                "question_type": "choice",
                "decision": "override",
                "override": "pos",
            }
        ]
    )
    apply_decisions(labeled, decisions, label_families(labeled))
    pd.testing.assert_frame_equal(labeled, before)


# -- exclude ------------------------------------------------------------------


def test_exclude_flags_the_row_rather_than_deleting_it(labeled: pd.DataFrame) -> None:
    decisions = validate_decisions(
        [{"row_index": 3, "family": "tone", "question_type": "choice", "decision": "exclude"}]
    )
    out, outcome = apply_decisions(labeled, decisions, label_families(labeled))
    assert outcome.excluded == [3]
    assert len(out) == len(labeled)  # the version is a record, not a filter
    assert bool(out[EXCLUDED_COLUMN].iloc[3]) is True
    assert bool(out[EXCLUDED_COLUMN].iloc[0]) is False


def test_excluding_a_row_short_circuits_its_other_labels(labeled: pd.DataFrame) -> None:
    decisions = validate_decisions(
        [
            {"row_index": 3, "family": "tone", "question_type": "choice", "decision": "exclude"},
            {"row_index": 3, "family": "is_spam", "question_type": "noul", "decision": "accept"},
        ]
    )
    out, outcome = apply_decisions(labeled, decisions, label_families(labeled))
    # the row is out of training either way, so its other labels are moot and
    # are deliberately not stamped — the Card should not claim work that has
    # no effect
    assert outcome.excluded == [3]
    assert outcome.accepted == 0
    entry = json.loads(out[PROVENANCE_COLUMN].iloc[3])["label_origins"]["is_spam"]
    assert entry["origin"] == "jev"


def test_exclusions_accumulate_across_passes(labeled: pd.DataFrame) -> None:
    first = validate_decisions(
        [{"row_index": 0, "family": "tone", "question_type": "choice", "decision": "exclude"}]
    )
    out, _ = apply_decisions(labeled, first, label_families(labeled))
    second = validate_decisions(
        [{"row_index": 1, "family": "tone", "question_type": "choice", "decision": "exclude"}]
    )
    out2, _ = apply_decisions(out, second, label_families(labeled))
    assert bool(out2[EXCLUDED_COLUMN].iloc[0]) is True
    assert bool(out2[EXCLUDED_COLUMN].iloc[1]) is True


def test_several_labels_on_one_row_can_be_decided_at_once(labeled: pd.DataFrame) -> None:
    decisions = validate_decisions(
        [
            {"row_index": 3, "family": "tone", "question_type": "choice", "decision": "accept"},
            {"row_index": 3, "family": "quality", "question_type": "score", "decision": "accept"},
        ]
    )
    out, outcome = apply_decisions(labeled, decisions, label_families(labeled))
    assert outcome.accepted == 2
    origins = json.loads(out[PROVENANCE_COLUMN].iloc[3])["label_origins"]
    assert origins["tone"]["origin"] == REVIEWED_ORIGIN
    assert origins["quality"]["origin"] == REVIEWED_ORIGIN


# -- the store operation ------------------------------------------------------


@pytest.fixture()
def store_and_version(tmp_path, monkeypatch):
    from dat_distiller.store import DatasetStore
    from dat_distiller.store.paths import AppPaths

    monkeypatch.setenv("DAT_DISTILLER_DATA_DIR", str(tmp_path / "data"))
    store = DatasetStore(AppPaths.from_env())
    project = store.create_project("Review Lab")
    frame = pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "tone": ["pos", "pos", "neg"],
            "tone__confidence": pd.array([0.9, 0.3, 0.8], dtype="Float64"),
            "tone__p_pos": pd.array([0.9, 0.3, 0.2], dtype="Float64"),
            "tone__p_neg": pd.array([0.1, 0.7, 0.8], dtype="Float64"),
            PROVENANCE_COLUMN: [prov("tone", "jev", 0.9)] * 3,
        }
    )
    version = store.create_version(project.id, frame, origin="labeled")
    return store, version


def test_applying_decisions_creates_a_child_version(store_and_version) -> None:
    store, version = store_and_version
    decisions = validate_decisions(
        [
            {
                "row_index": 1,
                "family": "tone",
                "question_type": "choice",
                "decision": "override",
                "override": "neg",
            }
        ]
    )
    result = apply_review(store, version.id, decisions, threshold=0.75)
    child = result["version"]
    assert child["parent_id"] == version.id
    assert child["origin"] == "reviewed"
    # the source is immutable: it still holds the original answer
    original = store.load_dataframe(version.id)
    assert original["tone"].iloc[1] == "pos"
    assert store.load_dataframe(child["id"])["tone"].iloc[1] == "neg"


def test_the_child_records_the_decisions_for_the_dataset_card(store_and_version) -> None:
    store, version = store_and_version
    decisions = validate_decisions(
        [{"row_index": 1, "family": "tone", "question_type": "choice", "decision": "accept"}]
    )
    result = apply_review(store, version.id, decisions, threshold=0.75)
    review = result["version"]["meta"]["review"]
    assert review["threshold"] == 0.75
    assert review["outcome"] == {
        "accepted": 1,
        "overridden": 0,
        "excluded_rows": [],
        "excluded_count": 0,
    }
    assert review["decisions"][0]["family"] == "tone"


def test_applying_reports_what_is_still_unreviewed(store_and_version) -> None:
    store, version = store_and_version
    result = apply_review(store, version.id, [], threshold=0.75)
    assert result["unreviewed_count"] == 1  # row 1's top option is 0.7
    # reviewing it clears the count for the Train step
    decisions = validate_decisions(
        [{"row_index": 1, "family": "tone", "question_type": "choice", "decision": "accept"}]
    )
    assert apply_review(store, version.id, decisions, threshold=0.75)["unreviewed_count"] == 0


def test_reviewing_a_version_with_no_label_columns_is_refused(store_and_version) -> None:
    store, version = store_and_version
    plain = store.create_version(
        version.project_id, pd.DataFrame({"a": [1, 2]}), parent_id=version.id
    )
    with pytest.raises(ValueError, match="no Label Columns"):
        apply_review(store, plain.id, [], threshold=0.75)


# -- the unreviewed Check -----------------------------------------------------


class FakeChecks:
    def __init__(self, existing: bool = False) -> None:
        self.registered: list[dict] = []
        self.existing = existing

    def has_check(self, kind, subject_type, subject_id):
        return self.existing

    def register(self, **kwargs):
        self.registered.append(kwargs)
        return kwargs


def test_the_check_counts_what_is_outstanding() -> None:
    checks = FakeChecks()
    check = register_unreviewed_check(checks, "v1", unreviewed=4, threshold=0.8, total=100)
    assert check is not None
    assert check["kind"] == UNREVIEWED_CHECK_KIND
    assert check["severity"] == "warning"
    assert check["subject_type"] == "dataset_version"
    assert check["details"] == {"unreviewed": 4, "threshold": 0.8, "labels": 100}
    assert "4 label(s)" in check["message"]
    assert "knowingly" in check["message"]


def test_a_clean_version_raises_nothing() -> None:
    checks = FakeChecks()
    assert register_unreviewed_check(checks, "v1", unreviewed=0, threshold=0.8, total=100) is None
    assert checks.registered == []


def test_the_check_is_not_raised_twice() -> None:
    checks = FakeChecks(existing=True)
    assert register_unreviewed_check(checks, "v1", unreviewed=4, threshold=0.8, total=10) is None
    assert checks.registered == []


def test_an_empty_queue_serializes() -> None:
    assert Queue(version_id="v1", threshold=0.8).to_dict()["queued_count"] == 0


def test_a_queue_item_round_trips_json() -> None:
    item = QueueItem(0, "tone", "choice", "pos", 0.3, decision="override", override="neg")
    assert json.loads(json.dumps(item.to_dict()))["override"] == "neg"
