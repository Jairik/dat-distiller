"""The Review Queue over the API.

What the endpoints have to get right: reading the queue registers nothing, a
threshold override is honoured without being saved, applying decisions produces
a *child* version, and the unreviewed Check is raised on the version that was
reviewed — not on the child, which is a different artifact.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from dat_distiller.review import EXCLUDED_COLUMN, REVIEWED_ORIGIN, UNREVIEWED_CHECK_KIND
from dat_distiller.store.provenance import PROVENANCE_COLUMN

CSV = b"""text,tone,tone__confidence,tone__p_pos,tone__p_neg
a,pos,0.9,0.9,0.1
b,pos,0.3,0.3,0.7
c,neg,0.8,0.2,0.8
d,neg,0.4,0.7,0.3
"""


def prov_row(confidences: dict[str, float]) -> str:
    return json.dumps(
        {
            "row_origin": "synthetic",
            "label_origins": {
                family: {"origin": "jev", "confidence": confidence}
                for family, confidence in confidences.items()
            },
        }
    )


@pytest.fixture()
def project(client: TestClient) -> dict:
    r = client.post("/api/projects", json={"name": "Review Lab"})
    assert r.status_code == 201
    return r.json()


def simple_version(client: TestClient, project_id: str) -> dict:
    """Upload the CSV and rewrite it through the store with jev Provenance."""
    store = client.app.state.store
    import pandas as pd

    frame = pd.DataFrame(
        {
            "text": ["a", "b", "c", "d"],
            "tone": ["pos", "pos", "neg", "neg"],
            "tone__confidence": pd.array([0.9, 0.3, 0.8, 0.4], dtype="Float64"),
            "tone__p_pos": pd.array([0.9, 0.3, 0.2, 0.7], dtype="Float64"),
            "tone__p_neg": pd.array([0.1, 0.7, 0.8, 0.3], dtype="Float64"),
            PROVENANCE_COLUMN: [prov_row({"tone": c}) for c in (0.9, 0.3, 0.8, 0.4)],
        }
    )
    return store.create_version(project_id, frame, origin="labeled").to_dict()


# -- reading the queue --------------------------------------------------------


def test_the_queue_lists_the_unsure_labels_with_their_answers(client: TestClient, project) -> None:
    version = simple_version(client, project["id"])
    r = client.get(f"/api/dataset-versions/{version['id']}/review-queue")
    assert r.status_code == 200
    body = r.json()
    assert body["threshold"] == 0.8
    assert body["queued_count"] == 2
    assert {(i["row_index"], i["family"]) for i in body["items"]} == {(1, "tone"), (3, "tone")}
    assert body["items"][0]["answer"] in {"pos", "neg"}
    assert body["decisions"] == ["accept", "override", "exclude"]
    assert body["unreviewed_count"] == 2


def test_reading_the_queue_registers_nothing(client: TestClient, project) -> None:
    version = simple_version(client, project["id"])
    client.get(f"/api/dataset-versions/{version['id']}/review-queue")
    checks = client.get(
        "/api/checks", params={"subject_type": "dataset_version", "subject_id": version["id"]}
    ).json()
    assert checks["checks"] == []


def test_a_threshold_override_changes_the_queue_without_being_saved(
    client: TestClient, project
) -> None:
    version = simple_version(client, project["id"])
    low = client.get(
        f"/api/dataset-versions/{version['id']}/review-queue", params={"threshold": 0.2}
    ).json()
    high = client.get(
        f"/api/dataset-versions/{version['id']}/review-queue", params={"threshold": 0.95}
    ).json()
    assert low["queued_count"] < high["queued_count"]
    # the configured threshold is untouched
    again = client.get(f"/api/dataset-versions/{version['id']}/review-queue").json()
    assert again["threshold"] == 0.8


def test_the_configured_threshold_is_used_when_none_is_given(
    client: TestClient, project
) -> None:
    settings = client.app.state.settings
    settings.save({"review_threshold": 0.95})
    version = simple_version(client, project["id"])
    body = client.get(f"/api/dataset-versions/{version['id']}/review-queue").json()
    assert body["threshold"] == 0.95
    # top-option confidences are 0.9, 0.7, 0.8, 0.7 — all four fall below 0.95
    assert body["queued_count"] == 4


def test_the_queue_is_paged_but_still_reports_the_whole_version(
    client: TestClient, project
) -> None:
    version = simple_version(client, project["id"])
    body = client.get(
        f"/api/dataset-versions/{version['id']}/review-queue",
        params={"threshold": 0.99, "limit": 1},
    ).json()
    assert len(body["items"]) == 1
    assert body["queued_count"] == 4
    assert body["returned_count"] == 1
    assert body["unreviewed_count"] == 4


def test_an_out_of_range_threshold_is_refused(client: TestClient, project) -> None:
    version = simple_version(client, project["id"])
    assert (
        client.get(
            f"/api/dataset-versions/{version['id']}/review-queue", params={"threshold": 1.5}
        ).status_code
        == 422
    )


def test_reading_the_queue_of_an_unknown_version_is_404(client: TestClient) -> None:
    assert client.get("/api/dataset-versions/nope/review-queue").status_code == 404


# -- the status the Train step reads ------------------------------------------


def test_status_is_just_the_count(client: TestClient, project) -> None:
    version = simple_version(client, project["id"])
    body = client.get(f"/api/dataset-versions/{version['id']}/review-status").json()
    assert body == {
        "version_id": version["id"],
        "threshold": 0.8,
        "queued_count": 2,
        "unlabeled_count": 0,
        "unreviewed_count": 2,
    }


def test_a_fully_confident_version_reports_nothing_outstanding(
    client: TestClient, project
) -> None:
    version = simple_version(client, project["id"])
    body = client.get(
        f"/api/dataset-versions/{version['id']}/review-status", params={"threshold": 0.1}
    ).json()
    assert body["unreviewed_count"] == 0


# -- applying decisions -------------------------------------------------------


def test_accept_creates_a_child_with_human_reviewed_provenance(
    client: TestClient, project
) -> None:
    version = simple_version(client, project["id"])
    r = client.post(
        f"/api/dataset-versions/{version['id']}/review",
        json={
            "decisions": [
                {"row_index": 1, "family": "tone", "question_type": "choice", "decision": "accept"}
            ]
        },
    )
    assert r.status_code == 201, r.text
    child = r.json()["version"]
    assert child["parent_id"] == version["id"]
    assert child["origin"] == "reviewed"

    store = client.app.state.store
    child_frame = store.load_dataframe(child["id"], include_provenance=True)
    entry = json.loads(child_frame[PROVENANCE_COLUMN].iloc[1])["label_origins"]["tone"]
    assert entry["origin"] == REVIEWED_ORIGIN
    assert entry["jev_confidence"] == 0.3
    # the source version is immutable
    original = store.load_dataframe(version["id"], include_provenance=True)
    assert json.loads(original[PROVENANCE_COLUMN].iloc[1])["label_origins"]["tone"]["origin"] == "jev"


def test_override_rewrites_the_answer_and_its_probabilities(
    client: TestClient, project
) -> None:
    version = simple_version(client, project["id"])
    client.post(
        f"/api/dataset-versions/{version['id']}/review",
        json={
            "decisions": [
                {
                    "row_index": 1,
                    "family": "tone",
                    "question_type": "choice",
                    "decision": "override",
                    "override": "neg",
                }
            ]
        },
    )
    child = client.app.state.store.list_versions(project["id"])[-1]
    frame = client.app.state.store.load_dataframe(child.id)
    assert frame["tone"].iloc[1] == "neg"
    assert frame["tone__p_neg"].iloc[1] == 1.0
    assert frame["tone__p_pos"].iloc[1] == 0.0


def test_exclude_flags_the_row_on_the_child(client: TestClient, project) -> None:
    version = simple_version(client, project["id"])
    r = client.post(
        f"/api/dataset-versions/{version['id']}/review",
        json={
            "decisions": [
                {"row_index": 3, "family": "tone", "question_type": "choice", "decision": "exclude"}
            ]
        },
    )
    assert r.json()["outcome"] == {
        "accepted": 0,
        "overridden": 0,
        "excluded_rows": [3],
        "excluded_count": 1,
    }
    child = r.json()["version"]
    frame = client.app.state.store.load_dataframe(child["id"])
    assert bool(frame[EXCLUDED_COLUMN].iloc[3]) is True
    assert len(frame) == 4  # the row is flagged, not deleted


def test_all_three_decisions_are_reflected_in_provenance(client: TestClient, project) -> None:
    version = simple_version(client, project["id"])
    r = client.post(
        f"/api/dataset-versions/{version['id']}/review",
        json={
            "decisions": [
                {"row_index": 1, "family": "tone", "question_type": "choice", "decision": "accept"},
                {
                    "row_index": 3,
                    "family": "tone",
                    "question_type": "choice",
                    "decision": "override",
                    "override": "pos",
                },
            ],
            "raise_check": False,
        },
    )
    outcome = r.json()["outcome"]
    assert outcome["accepted"] == 1 and outcome["overridden"] == 1
    child = r.json()["version"]
    frame = client.app.state.store.load_dataframe(child["id"], include_provenance=True)
    assert json.loads(frame[PROVENANCE_COLUMN].iloc[1])["label_origins"]["tone"]["origin"] == REVIEWED_ORIGIN
    assert json.loads(frame[PROVENANCE_COLUMN].iloc[3])["label_origins"]["tone"]["origin"] == REVIEWED_ORIGIN
    assert frame["tone"].iloc[3] == "pos"


def test_reviewing_clears_the_count_for_the_train_step(client: TestClient, project) -> None:
    version = simple_version(client, project["id"])
    before = client.get(f"/api/dataset-versions/{version['id']}/review-status").json()
    assert before["unreviewed_count"] == 2
    r = client.post(
        f"/api/dataset-versions/{version['id']}/review",
        json={
            "decisions": [
                {"row_index": 1, "family": "tone", "question_type": "choice", "decision": "accept"},
                {"row_index": 3, "family": "tone", "question_type": "choice", "decision": "accept"},
            ],
            "raise_check": False,
        },
    )
    child = r.json()["version"]["id"]
    assert client.get(f"/api/dataset-versions/{child}/review-status").json()["unreviewed_count"] == 0


def test_a_reviewed_label_stays_out_of_the_queue(client: TestClient, project) -> None:
    version = simple_version(client, project["id"])
    client.post(
        f"/api/dataset-versions/{version['id']}/review",
        json={
            "decisions": [
                {"row_index": 1, "family": "tone", "question_type": "choice", "decision": "accept"}
            ],
            "raise_check": False,
        },
    )
    child = client.app.state.store.list_versions(project["id"])[-1]
    queued = {
        (i["row_index"], i["family"])
        for i in client.get(f"/api/dataset-versions/{child.id}/review-queue").json()["items"]
    }
    assert (1, "tone") not in queued
    assert (3, "tone") in queued


# -- the unreviewed Check -----------------------------------------------------


def test_unreviewed_labels_raise_a_warning_check_on_the_version(
    client: TestClient, project
) -> None:
    version = simple_version(client, project["id"])
    r = client.post(f"/api/dataset-versions/{version['id']}/review", json={"decisions": []})
    assert len(r.json()["checks_raised"]) == 1
    check = r.json()["checks_raised"][0]
    assert check["kind"] == UNREVIEWED_CHECK_KIND
    assert check["severity"] == "warning"
    assert check["details"]["unreviewed"] == 2
    assert check["details"]["threshold"] == 0.8
    # it is registered against the version that was reviewed
    assert check["subject_id"] == version["id"]


def test_a_clean_review_raises_no_check(client: TestClient, project) -> None:
    version = simple_version(client, project["id"])
    r = client.post(
        f"/api/dataset-versions/{version['id']}/review",
        json={
            "decisions": [
                {"row_index": 1, "family": "tone", "question_type": "choice", "decision": "accept"},
                {"row_index": 3, "family": "tone", "question_type": "choice", "decision": "accept"},
            ]
        },
    )
    assert r.json()["checks_raised"] == []


def test_the_check_is_raised_once(client: TestClient, project) -> None:
    version = simple_version(client, project["id"])
    client.post(f"/api/dataset-versions/{version['id']}/review", json={"decisions": []})
    again = client.post(f"/api/dataset-versions/{version['id']}/review", json={"decisions": []})
    assert again.json()["checks_raised"] == []


def test_raising_the_check_can_be_declined(client: TestClient, project) -> None:
    version = simple_version(client, project["id"])
    r = client.post(
        f"/api/dataset-versions/{version['id']}/review",
        json={"decisions": [], "raise_check": False},
    )
    assert r.json()["checks_raised"] == []


# -- refusals -----------------------------------------------------------------


def test_an_override_without_an_answer_is_refused(client: TestClient, project) -> None:
    version = simple_version(client, project["id"])
    r = client.post(
        f"/api/dataset-versions/{version['id']}/review",
        json={
            "decisions": [
                {"row_index": 1, "family": "tone", "question_type": "choice", "decision": "override"}
            ]
        },
    )
    assert r.status_code == 422
    assert "needs a new answer" in r.text


def test_an_unknown_decision_is_refused(client: TestClient, project) -> None:
    version = simple_version(client, project["id"])
    r = client.post(
        f"/api/dataset-versions/{version['id']}/review",
        json={"decisions": [{"row_index": 1, "family": "tone", "decision": "delete"}]},
    )
    assert r.status_code == 422
    assert "accept" in r.text and "override" in r.text


def test_reviewing_a_column_that_is_not_a_label_is_refused(client: TestClient, project) -> None:
    version = simple_version(client, project["id"])
    r = client.post(
        f"/api/dataset-versions/{version['id']}/review",
        json={
            "decisions": [
                {"row_index": 0, "family": "text", "question_type": "noul", "decision": "accept"}
            ]
        },
    )
    assert r.status_code == 422
    assert "not a Label Column" in r.text


def test_a_negative_row_index_is_refused(client: TestClient, project) -> None:
    version = simple_version(client, project["id"])
    r = client.post(
        f"/api/dataset-versions/{version['id']}/review",
        json={"decisions": [{"row_index": -1, "family": "tone", "decision": "accept"}]},
    )
    assert r.status_code == 422


def test_reviewing_a_version_with_no_label_columns_is_refused(
    client: TestClient, project
) -> None:
    r = client.post(
        f"/api/projects/{project['id']}/upload",
        files={"file": ("plain.csv", b"a,b\n1,2\n", "text/csv")},
    )
    plain = r.json()
    out = client.post(
        f"/api/dataset-versions/{plain['id']}/review", json={"decisions": []}
    )
    assert out.status_code == 422
    assert "no Label Columns" in out.text


def test_applying_to_an_unknown_version_is_404(client: TestClient) -> None:
    assert (
        client.post("/api/dataset-versions/nope/review", json={"decisions": []}).status_code == 404
    )
