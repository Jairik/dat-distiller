"""Cards: the exportable record of how a Dataset Version or Model was made.

The two acceptance criteria that matter are checked by *reading the output*, not
by trusting the code:

* **No API keys.** A Card is designed to be shared, and one that leaked a key
  would leak it to everyone it reached. The test plants a real-looking key in
  every place a value could hide — a top-level field, a nested dict, a list
  element, a free-text string, a Check's details — and then greps the rendered
  Markdown *and* the JSON for it.
* **Every Acknowledgement appears.** A Card that omitted the acknowledged Checks
  would hide the human decisions that shaped the data, which is the single most
  misleading thing a Card could do.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dat_distiller.cards import (
    REDACTED,
    REPRODUCIBILITY_CAVEAT,
    DatasetCard,
    ModelCard,
)
from dat_distiller.store import DatasetStore
from dat_distiller.store.provenance import PROVENANCE_COLUMN

pytest.importorskip("sklearn")

#: A key that looks exactly like the real thing, so a leak is unmistakable.
FAKE_KEY = "sk-or-v1-abcdef0123456789abcdef0123456789"


# -- fixtures -----------------------------------------------------------------


@pytest.fixture()
def project_id(client: TestClient) -> str:
    return client.post("/api/projects", json={"name": "card lab"}).json()["id"]


def labeled_frame(rows: int = 60, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, rows)
    spend = rng.normal(0, 1, rows).round(3)
    return pd.DataFrame(
        {
            "age": age,
            "spend": spend,
            "is_churn": pd.array((age < 45) & (spend > 0), dtype="boolean"),
            "is_churn__confidence": pd.array([0.95] * rows, dtype="Float64"),
            PROVENANCE_COLUMN: [
                json.dumps(
                    {
                        "row_origin": "synthetic",
                        "label_origins": {
                            "is_churn": {"origin": "jev", "confidence": 0.95}
                        },
                    }
                )
                for _ in range(rows)
            ],
        }
    )


def generated_frame(rows: int = 60, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, rows)
    return pd.DataFrame(
        {
            "age": age,
            "spend": rng.normal(0, 1, rows).round(3),
            PROVENANCE_COLUMN: [
                json.dumps({"row_origin": "synthetic", "mode": "hybrid", "provider": "openrouter"})
                for _ in range(rows)
            ],
        }
    )


def version(client: TestClient, project_id: str, frame: pd.DataFrame, origin: str, meta: dict | None = None) -> dict:
    store: DatasetStore = client.app.state.store
    return store.create_version(
        project_id, frame, origin=origin, meta=meta or {}, seed=17
    ).to_dict()


def check(
    check_id: str,
    kind: str,
    severity: str,
    message: str,
    subject_id: str,
    *,
    acknowledged: bool,
    details: dict | None = None,
) -> dict:
    return {
        "id": check_id,
        "kind": kind,
        "severity": severity,
        "message": message,
        "subject_type": "dataset_version",
        "subject_id": subject_id,
        "details": details or {},
        "acknowledged": acknowledged,
        "acknowledged_at": "2024-05-01T10:00:00Z" if acknowledged else None,
        "note": "reviewed by a person" if acknowledged else None,
    }


# -- AC1: no API keys --------------------------------------------------------


def test_a_card_never_prints_an_api_key(client: TestClient, project_id: str) -> None:
    """A Card is meant to be shared. A key in one would leak to everyone."""
    frame = labeled_frame()
    frame["leaked"] = f"my key is {FAKE_KEY} ok"
    stored = version(client, project_id, frame, "labeled")
    client.app.state.checks.register(
        kind="pii_found",
        severity="warning",
        message=f"found something near {FAKE_KEY}",
        subject_type="dataset_version",
        subject_id=stored["id"],
        details={
            "api_key": FAKE_KEY,
            "nested": {"openrouter_api_key": FAKE_KEY},
            "in_a_list": [FAKE_KEY, "harmless"],
            "free_text": f"Authorization: Bearer {FAKE_KEY}",
        },
    )

    for fmt in ("json", "markdown"):
        r = client.get(f"/api/dataset-versions/{stored['id']}/card", params={"format": fmt})
        assert r.status_code == 200, r.text
        assert FAKE_KEY not in r.text, f"the key leaked into the {fmt} Card"
        assert REDACTED in r.text, f"the {fmt} Card should say it redacted something"


def test_redaction_covers_nested_shapes_and_free_text() -> None:
    payload = {
        "top": FAKE_KEY,
        "api_key": "anything at all",
        "nested": {"deep": {"secret_token": "hunter2hunter2"}},
        "list": [{"key": "x"}, FAKE_KEY],
        "note": f"sent with Bearer {FAKE_KEY} yesterday",
    }
    text = json.dumps(DatasetCard("v1", "p1", 1, "uploaded", 0, meta={"x": payload}).to_dict())
    assert FAKE_KEY not in text
    assert "hunter2hunter2" not in text


def test_a_clean_card_says_nothing_was_redacted(client: TestClient, project_id: str) -> None:
    stored = version(client, project_id, labeled_frame(), "uploaded")
    body = client.get(f"/api/dataset-versions/{stored['id']}/card").json()
    # no redacted marker anywhere: nothing needed redacting
    assert REDACTED not in json.dumps(body)


# -- AC2: every Acknowledgement appears ---------------------------------------


def test_every_check_and_acknowledgement_appears_in_the_dataset_card(
    client: TestClient, project_id: str
) -> None:
    stored = version(client, project_id, labeled_frame(), "labeled")
    client.app.state.checks.register(
        kind="pii_found",
        severity="warning",
        message="Possible PII in column 'spend'",
        subject_type="dataset_version",
        subject_id=stored["id"],
        details={"column": "spend"},
    )
    client.app.state.checks.register(
        kind="class_imbalance",
        severity="warning",
        message="The Target is 4% of the rows",
        subject_type="dataset_version",
        subject_id=stored["id"],
    )
    checks = client.app.state.checks.list_for_subject("dataset_version", stored["id"])
    first = checks[0]
    r = client.post(
        f"/api/checks/{first.id}/acknowledge", json={"note": "synthetic, continuing"}
    )
    assert r.status_code == 200

    body = client.get(f"/api/dataset-versions/{stored['id']}/card").json()
    kinds = {c["kind"] for c in body["checks"]}
    assert kinds == {"pii_found", "class_imbalance"}

    acknowledged = [c for c in body["checks"] if c["acknowledged"]]
    assert len(acknowledged) == 1
    assert acknowledged[0]["kind"] == "pii_found"
    assert acknowledged[0]["note"] == "synthetic, continuing"
    assert acknowledged[0]["acknowledged_at"]

    markdown = client.get(
        f"/api/dataset-versions/{stored['id']}/card", params={"format": "markdown"}
    ).text
    for check_row in body["checks"]:
        assert check_row["kind"] in markdown
        assert check_row["message"] in markdown
    # the unacknowledged one is visibly unacknowledged
    assert markdown.count("**no**") == 1
    assert "reviewed by a person" in markdown or "synthetic, continuing" in markdown


def test_an_acknowledged_check_is_not_hidden_from_the_card(
    client: TestClient, project_id: str
) -> None:
    stored = version(client, project_id, labeled_frame(), "labeled")
    registered = client.app.state.checks.register(
        kind="pii_found",
        severity="warning",
        message="Possible PII",
        subject_type="dataset_version",
        subject_id=stored["id"],
    )
    client.post(f"/api/checks/{registered.id}/acknowledge", json={"note": "fine"})
    markdown = client.get(
        f"/api/dataset-versions/{stored['id']}/card", params={"format": "markdown"}
    ).text
    assert "pii_found" in markdown
    assert "fine" in markdown


# -- the Dataset Card's content ----------------------------------------------


def test_the_dataset_card_covers_lineage_provenance_questions_and_limitations(
    client: TestClient, project_id: str
) -> None:
    parent = version(client, project_id, generated_frame(), "generated")
    meta = {
        "generation": {"mode": "hybrid", "provider": "openrouter", "model": "x/y"},
        "fidelity": {
            "correlation_drift": 0.12,
            "near_copies": {"count": 2},
            "exact_duplicates": {"count": 1},
            "dropped_rows": 4,
            "provider_derived_profile": False,
        },
    }
    store = client.app.state.store
    child = store.create_version(
        project_id, labeled_frame(), parent_id=parent["id"], origin="generated", meta=meta
    )

    markdown = client.get(
        f"/api/dataset-versions/{child.id}/card", params={"format": "markdown"}
    ).text
    assert f"# Dataset Card — v{child.number}" in markdown
    assert "## Provenance" in markdown
    assert "## Columns" in markdown
    assert "## Generation" in markdown
    assert "## Fidelity Report" in markdown
    assert "hybrid" in markdown
    assert "## Known limitations" in markdown
    assert "Dataset Card" in markdown  # self-describing header


def test_the_jev_questions_reach_the_card(client: TestClient, project_id: str) -> None:
    stored = version(
        client,
        project_id,
        labeled_frame(),
        "labeled",
        meta={
            "labeling": {
                "questions": [
                    {
                        "type": "noul",
                        "name": "is_churn",
                        "instructions": "Answer yes when the customer left.",
                    }
                ],
                "state_columns": ["age"],
            }
        },
    )
    markdown = client.get(
        f"/api/dataset-versions/{stored['id']}/card", params={"format": "markdown"}
    ).text
    assert "## Jev Questions" in markdown
    assert "is_churn" in markdown
    assert "Answer yes when the customer left." in markdown
    assert "State columns" in markdown


def test_a_provider_derived_profile_is_flagged_in_the_card(
    client: TestClient, project_id: str
) -> None:
    stored = version(
        client,
        project_id,
        labeled_frame(),
        "generated",
        meta={"fidelity": {"provider_derived_profile": True, "correlation_drift": 0.0}},
    )
    markdown = client.get(
        f"/api/dataset-versions/{stored['id']}/card", params={"format": "markdown"}
    ).text
    assert "against a description, not reality" in markdown


def test_review_counts_reach_the_card(client: TestClient, project_id: str) -> None:
    parent = version(client, project_id, labeled_frame(), "labeled")
    store = client.app.state.store
    reviewed = store.create_version(
        project_id,
        labeled_frame(),
        parent_id=parent["id"],
        origin="reviewed",
        meta={"review": {"outcome": {"accepted": 5, "overridden": 2, "excluded_rows": [1]}}},
    )
    markdown = client.get(
        f"/api/dataset-versions/{reviewed.id}/card", params={"format": "markdown"}
    ).text
    assert "## Review" in markdown
    assert "accepted" in markdown
    assert "5" in markdown


def test_pii_handling_reaches_the_card(client: TestClient, project_id: str) -> None:
    parent = version(client, project_id, labeled_frame(), "labeled")
    store = client.app.state.store
    masked = store.create_version(
        project_id,
        labeled_frame(),
        parent_id=parent["id"],
        origin="redacted",
        meta={"pii": {"actions": {"spend": "mask"}, "summary": {"masked": ["spend"]}}},
    )
    markdown = client.get(
        f"/api/dataset-versions/{masked.id}/card", params={"format": "markdown"}
    ).text
    assert "## PII handling" in markdown
    assert "spend" in markdown


# -- reproducibility ---------------------------------------------------------


def test_a_generated_card_states_the_reproducibility_caveat(
    client: TestClient, project_id: str
) -> None:
    stored = version(client, project_id, generated_frame(), "generated")
    for fmt in ("json", "markdown"):
        body = client.get(
            f"/api/dataset-versions/{stored['id']}/card", params={"format": fmt}
        ).text
        assert "cannot be reproduced exactly" in body, fmt
        assert "snapshot, not a recipe" in body, fmt


def test_a_plain_upload_does_not_claim_to_need_the_caveat(
    client: TestClient, project_id: str
) -> None:
    stored = version(client, project_id, labeled_frame(), "uploaded")
    body = client.get(f"/api/dataset-versions/{stored['id']}/card").json()
    assert body["reproducibility"] == REPRODUCIBILITY_CAVEAT  # the field is always present
    markdown = client.get(
        f"/api/dataset-versions/{stored['id']}/card", params={"format": "markdown"}
    ).text
    # an uploaded file has no Provider in it, so it should not claim otherwise
    assert REPRODUCIBILITY_CAVEAT not in markdown


# -- the Model Card ----------------------------------------------------------


def run_training(client: TestClient, project_id: str) -> dict[str, Any]:
    frame = labeled_frame(80)
    frame["region"] = np.random.default_rng(2).choice(["north", "south"], 80)
    store = client.app.state.store
    v = store.create_version(project_id, frame, origin="labeled").id
    start = client.post(
        "/api/train/run",
        json={"version_id": v, "target": "is_churn", "models": ["logistic_regression"]},
    )
    assert start.status_code == 202, start.text
    for _ in range(6000):
        job = client.get(f"/api/jobs/{start.json()['id']}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.02)
    assert job["status"] == "completed", job.get("error")
    return job["result"]


def test_the_model_card_covers_target_features_metrics_and_limitations(
    client: TestClient, project_id: str
) -> None:
    run = run_training(client, project_id)
    run_id = run["training_run_id"]
    r = client.get(f"/api/train/runs/{run_id}/card", params={"format": "markdown"})
    assert r.status_code == 200
    markdown = r.text
    assert "# Model Card" in markdown
    assert "`is_churn`" in markdown
    assert "classification" in markdown
    assert "## Features" in markdown
    assert "## Split" in markdown
    assert "Tuning cross-validates on the training split only" in markdown
    assert "## Leaderboard" in markdown
    assert "logistic" in markdown.lower()
    assert "## Known limitations" in markdown
    assert "cannot be reproduced exactly" in markdown
    assert "within the noise of that split size" in markdown


def test_the_model_card_names_the_excluded_siblings(client: TestClient, project_id: str) -> None:
    run = run_training(client, project_id)
    markdown = client.get(
        f"/api/train/runs/{run['training_run_id']}/card", params={"format": "markdown"}
    ).text
    # the leakage guard dropped the confidence sibling, and the Card says so
    assert "excluded: is_churn__confidence" in markdown


def test_the_model_card_references_the_training_data_card(
    client: TestClient, project_id: str
) -> None:
    run = run_training(client, project_id)
    markdown = client.get(
        f"/api/train/runs/{run['training_run_id']}/card", params={"format": "markdown"}
    ).text
    assert "## Training data" in markdown
    assert run["version_id"] in markdown
    assert "Dataset Card v" in markdown


def test_the_model_card_carries_the_training_runs_checks(
    client: TestClient, project_id: str
) -> None:
    run = run_training(client, project_id)
    run_id = run["training_run_id"]
    client.app.state.checks.register(
        kind="class_imbalance",
        severity="warning",
        message="The Target is 4% of the rows",
        subject_type="training_run",
        subject_id=run_id,
    )
    body = client.get(f"/api/train/runs/{run_id}/card").json()
    kinds = [c["kind"] for c in body["checks"]]
    assert "class_imbalance" in kinds
    # and so is every other Check the setup already raised
    assert "label_sibling_exclusion" in kinds
    markdown = client.get(
        f"/api/train/runs/{run_id}/card", params={"format": "markdown"}
    ).text
    assert "class_imbalance" in markdown
    assert "**no**" in markdown  # not acknowledged yet, and it says so


def test_an_unranked_model_is_explained_not_dropped(client: TestClient, project_id: str) -> None:
    card = ModelCard(
        run={
            "training_run_id": "r1",
            "task_type": "classification",
            "leaderboard": [
                {
                    "model": "logistic_regression",
                    "label": "Logistic Regression",
                    "library": "sklearn",
                    "rank": 1,
                    "primary": {"value": 0.9},
                    "metrics": {"accuracy": {"value": 0.9}, "roc_auc": {"value": 0.95}},
                },
                {
                    "model": "svm",
                    "label": "SVM",
                    "library": "sklearn",
                    "rank": None,
                    "error": "this Model does not output class probabilities",
                    "primary": {"value": None},
                    "metrics": {"roc_auc": {"value": None, "reason": "no probabilities"}},
                },
            ],
        }
    )
    markdown = card.to_markdown()
    assert "Not ranked" in markdown
    assert "SVM" in markdown
    assert "no comparable metric" in markdown or "class probabilities" in markdown


def test_a_sensitive_attribute_is_described_by_whether_it_was_included() -> None:
    base = {
        "training_run_id": "r1",
        "version_id": "v1",
        "task_type": "classification",
        "primary_metric": "f1_macro",
        "primary_metric_higher_is_better": True,
        "setup": {"target": "y", "sensitive_attribute": "region"},
        "leaderboard": [],
    }
    excluded = ModelCard(run=base).to_markdown()
    assert "excluded from the features by default" in excluded
    included = ModelCard(run={**base, "setup": {**base["setup"], "include_sensitive_attribute": True}}).to_markdown()
    assert "included" in included
    assert "explicit opt-in" in included


# -- the endpoints ------------------------------------------------------------


def test_the_card_of_an_unknown_dataset_version_is_404(client: TestClient) -> None:
    assert client.get("/api/dataset-versions/nope/card").status_code == 404


def test_the_card_of_an_unknown_run_is_404(client: TestClient) -> None:
    assert client.get("/api/train/runs/nope/card").status_code == 404


def test_a_card_serves_as_a_download(client: TestClient, project_id: str) -> None:
    stored = version(client, project_id, labeled_frame(), "uploaded")
    for fmt, expected_type in (("markdown", "text/markdown"), ("json", "application/json")):
        r = client.get(f"/api/dataset-versions/{stored['id']}/card", params={"format": fmt})
        assert expected_type in r.headers["content-type"]
        assert "attachment" in r.headers["content-disposition"]
        suffix = "md" if fmt == "markdown" else "json"
    assert f"dataset-card-v{stored['number']}.{suffix}" in r.headers["content-disposition"]


def test_an_unknown_format_is_refused(client: TestClient, project_id: str) -> None:
    stored = version(client, project_id, labeled_frame(), "uploaded")
    assert (
        client.get(
            f"/api/dataset-versions/{stored['id']}/card", params={"format": "pdf"}
        ).status_code
        == 422
    )


def test_the_json_and_markdown_cards_agree(client: TestClient, project_id: str) -> None:
    run = run_training(client, project_id)
    run_id = run["training_run_id"]
    body = client.get(f"/api/train/runs/{run_id}/card").json()
    markdown = client.get(
        f"/api/train/runs/{run_id}/card", params={"format": "markdown"}
    ).text
    assert body["target"] in markdown
    assert body["primary_metric"] in markdown
    assert str(body["seed"]) in markdown
    for entry in body["leaderboard"]:
        assert entry["model"].replace("_", " ") in markdown.lower() or entry["model"] in markdown


def test_a_card_builds_even_when_a_pii_scan_cannot_run(
    client: TestClient, project_id: str, monkeypatch
) -> None:
    import dat_distiller.api.cards as cards_module

    stored = version(client, project_id, labeled_frame(), "uploaded")

    def boom(*args: Any, **kwargs: Any):
        raise RuntimeError("scan exploded")

    monkeypatch.setattr("dat_distiller.pii.scan", boom)
    r = client.get(f"/api/dataset-versions/{stored['id']}/card")
    assert r.status_code == 200
    assert "no PII scan was available" in r.text
    assert cards_module.router is not None
