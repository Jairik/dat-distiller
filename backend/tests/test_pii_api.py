"""PII over the API and on the two paths that create a Dataset Version.

The two rules under test: nothing destructive happens to the Version you looked
at (mask and drop always produce a *child*), and the `warn` action is exactly
"raise a Check and change nothing".
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dat_distiller.pii import PII_CHECK_KIND
from dat_distiller.store.provenance import PROVENANCE_COLUMN

CSV = b"""email,note,age
jane@example.com,"call 555-123-4567",30
bob@example.org,all clean,40
"""


def upload_csv(client: TestClient, project_id: str = "p1", body: bytes = CSV) -> dict:
    r = client.post(
        f"/api/projects/{project_id}/upload",
        files={"file": ("people.csv", body, "text/csv")},
    )
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture()
def project(client: TestClient) -> dict:
    r = client.post("/api/projects", json={"name": "PII Lab"})
    assert r.status_code == 201, r.text
    return r.json()


def checks_for(client: TestClient, version_id: str) -> list[dict]:
    r = client.get(
        "/api/checks", params={"subject_type": "dataset_version", "subject_id": version_id}
    )
    assert r.status_code == 200, r.text
    return [c for c in r.json()["checks"] if c["kind"] == PII_CHECK_KIND]


# -- the upload hook ----------------------------------------------------------


def test_upload_raises_a_warning_check_per_affected_column(client: TestClient, project) -> None:
    version = upload_csv(client, project["id"])
    assert version["pii"]["columns"] == ["email", "note"]
    assert version["pii"]["total_findings"] == 3  # 2 emails + 1 phone; "all clean" matches nothing

    found = checks_for(client, version["id"])
    assert [c["details"]["column"] for c in found] == ["email", "note"]
    assert all(c["severity"] == "warning" for c in found)
    # warn requires an Acknowledgement; nothing is blocked
    assert all(c["acknowledged"] is False for c in found)
    assert client.get("/api/checks", params={"subject_type": "dataset_version", "subject_id": version["id"]}).json()[
        "unacknowledged_warnings"
    ] == 2


def test_upload_check_details_never_carry_raw_pii(client: TestClient, project) -> None:
    version = upload_csv(client, project["id"])
    body = client.get(
        "/api/checks", params={"subject_type": "dataset_version", "subject_id": version["id"]}
    ).text
    assert "jane@example.com" not in body
    assert "555-123-4567" not in body


def test_a_clean_upload_raises_no_check(client: TestClient, project) -> None:
    version = upload_csv(
        client, project["id"], b"email,note\nnone here,all clean\n"
    )
    assert version["pii"]["total_findings"] == 0
    assert version["pii_checks"] == []
    assert checks_for(client, version["id"]) == []


def test_re_uploading_the_same_file_does_not_duplicate_checks(client: TestClient, project) -> None:
    first = upload_csv(client, project["id"])
    second = upload_csv(client, project["id"])
    assert len(checks_for(client, first["id"])) == 2
    assert len(checks_for(client, second["id"])) == 2  # per-version, not per-project


# -- scan is read-only --------------------------------------------------------


def test_scan_reports_findings_without_registering_a_check(client: TestClient, project) -> None:
    version = upload_csv(client, project["id"])
    before = len(checks_for(client, version["id"]))

    r = client.get(f"/api/dataset-versions/{version['id']}/pii")
    assert r.status_code == 200
    body = r.json()
    assert body["version_id"] == version["id"]
    assert {(f["column"], f["detector"]) for f in body["findings"]} == {
        ("email", "email"),
        ("note", "phone"),
    }
    assert body["actions"] == ["mask", "drop"]
    assert len(checks_for(client, version["id"])) == before


def test_scan_of_an_unknown_version_is_404(client: TestClient) -> None:
    assert client.get("/api/dataset-versions/nope/pii").status_code == 404


# -- warn ---------------------------------------------------------------------


def test_warn_raises_checks_and_changes_nothing(client: TestClient, project) -> None:
    version = upload_csv(client, project["id"])
    df_before = client.get(f"/api/dataset-versions/{version['id']}/download").text

    r = client.post(f"/api/dataset-versions/{version['id']}/pii/warn")
    assert r.status_code == 200
    assert r.json()["check_kind"] == PII_CHECK_KIND
    assert client.get(f"/api/dataset-versions/{version['id']}/download").text == df_before
    # the version count is unchanged — warn is not a step
    assert len(client.get(f"/api/projects/{project['id']}/dataset_versions").json()) == 1


def test_warn_deduplicates_repeated_calls(client: TestClient, project) -> None:
    version = upload_csv(client, project["id"])
    client.post(f"/api/dataset-versions/{version['id']}/pii/warn")
    r = client.post(f"/api/dataset-versions/{version['id']}/pii/warn")
    assert r.json()["checks"] == []
    assert len(checks_for(client, version["id"])) == 2


def test_acknowledging_a_pii_check_unblocks_the_gate(client: TestClient, project) -> None:
    version = upload_csv(client, project["id"])
    found = checks_for(client, version["id"])
    for check in found:
        r = client.post(f"/api/checks/{check['id']}/acknowledge", json={"note": "synthetic"})
        assert r.status_code == 200
    gate = client.get(
        "/api/checks", params={"subject_type": "dataset_version", "subject_id": version["id"]}
    ).json()
    assert gate["unacknowledged_warnings"] == 0


# -- mask and drop ------------------------------------------------------------


def test_mask_creates_a_child_version_and_leaves_the_parent_alone(
    client: TestClient, project
) -> None:
    version = upload_csv(client, project["id"])
    r = client.post(
        f"/api/dataset-versions/{version['id']}/pii/actions", json={"actions": {"email": "mask"}}
    )
    assert r.status_code == 201, r.text
    child = r.json()["version"]
    assert child["parent_id"] == version["id"]
    assert child["origin"] == "redacted"
    assert "email" in [c["name"] for c in child["columns"]]

    # the parent still holds the PII — it is immutable
    assert "jane@example.com" in client.get(
        f"/api/dataset-versions/{version['id']}/download"
    ).text
    # the child does not
    assert "jane@example.com" not in client.get(
        f"/api/dataset-versions/{child['id']}/download"
    ).text
    assert "555-123-4567" in client.get(
        f"/api/dataset-versions/{child['id']}/download"
    ).text  # only `email` was masked


def test_drop_creates_a_child_without_the_column(client: TestClient, project) -> None:
    version = upload_csv(client, project["id"])
    r = client.post(
        f"/api/dataset-versions/{version['id']}/pii/actions", json={"actions": {"email": "drop"}}
    )
    assert r.status_code == 201
    child = r.json()["version"]
    assert "email" not in [c["name"] for c in child["columns"]]
    assert "jane@example.com" not in client.get(
        f"/api/dataset-versions/{child['id']}/download"
    ).text


def test_mask_and_drop_together(client: TestClient, project) -> None:
    version = upload_csv(client, project["id"])
    r = client.post(
        f"/api/dataset-versions/{version['id']}/pii/actions",
        json={"actions": {"email": "mask", "note": "drop"}},
    )
    assert r.status_code == 201
    child = r.json()["version"]
    names = [c["name"] for c in child["columns"]]
    assert "email" in names and "note" not in names


def test_the_child_records_what_was_done_for_the_dataset_card(
    client: TestClient, project
) -> None:
    version = upload_csv(client, project["id"])
    client.post(
        f"/api/dataset-versions/{version['id']}/pii/actions",
        json={"actions": {"email": "mask"}},
    )
    meta = client.get(f"/api/dataset-versions/{version['id']}/pii").json()  # still scannable
    assert meta["summary"]["columns"] == ["email", "note"]  # parent untouched

    versions = client.get(f"/api/projects/{project['id']}/dataset_versions").json()
    child = next(v for v in versions if v["parent_id"] == version["id"])
    assert child["meta"]["pii"]["actions"] == {"email": "mask"}
    assert child["meta"]["pii"]["summary"]["columns"]["email"]["replacements"] == 2
    assert child["meta"]["pii"]["parent_version_id"] == version["id"]


def test_provenance_survives_a_mask(client: TestClient, project) -> None:
    version = upload_csv(client, project["id"])
    r = client.post(
        f"/api/dataset-versions/{version['id']}/pii/actions", json={"actions": {"email": "mask"}}
    )
    child = r.json()["version"]
    body = client.get(
        f"/api/dataset-versions/{child['id']}/download",
        params={"include_provenance": True},
    ).text
    assert PROVENANCE_COLUMN in body or "uploaded" in body


def test_an_unknown_action_is_refused(client: TestClient, project) -> None:
    version = upload_csv(client, project["id"])
    r = client.post(
        f"/api/dataset-versions/{version['id']}/pii/actions", json={"actions": {"email": "hash"}}
    )
    assert r.status_code == 422
    assert "mask" in r.text and "drop" in r.text


def test_an_empty_action_set_is_a_harmless_child(client: TestClient, project) -> None:
    version = upload_csv(client, project["id"])
    r = client.post(f"/api/dataset-versions/{version['id']}/pii/actions", json={"actions": {}})
    assert r.status_code == 201
    assert r.json()["summary"]["columns"] == {}


def test_actions_on_an_unknown_version_is_404(client: TestClient) -> None:
    r = client.post("/api/dataset-versions/nope/pii/actions", json={"actions": {}})
    assert r.status_code == 404


# -- the Generation hook ------------------------------------------------------


def test_generation_scans_provider_output(client: TestClient, project, monkeypatch) -> None:
    """A Provider that invents a real-looking email must still be caught."""
    from dat_distiller.generate import run as gen_run
    from dat_distiller.generate.modes import ModeResult

    def leaky(provider, mode, profile, description, count, **kwargs):
        return ModeResult(
            rows=pd.DataFrame(
                {"contact": ["jane@example.com"] * count, "amount": [10.0] * count}
            ),
            profile=profile,
        )

    monkeypatch.setattr(gen_run, "generate_with_mode", leaky)
    seed_version = upload_csv(client, project["id"], b"contact,amount\nseed@example.com,1.0\n")
    r = client.post(
        "/api/generate/run",
        json={
            "project_id": project["id"],
            "mode": "statistical",
            "count": 5,
            "sample_version_id": seed_version["id"],
            "description": "leaky provider output",
        },
    )
    assert r.status_code == 202, r.text
    result = _finish(client, r.json()["id"])
    assert result["pii"]["columns"] == ["contact"]
    assert result["pii"]["detectors"] == {"email": 5}
    assert checks_for(client, result["version_id"])


def _finish(client: TestClient, job_id: str) -> dict:
    import time

    for _ in range(200):
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            assert body["status"] == "completed", body.get("error")
            return body["result"]
        time.sleep(0.02)
    raise AssertionError("generation job never finished")
