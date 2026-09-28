"""Checks + Acknowledgements: registration, the warning gate, Cards queries."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dat_distiller.checks import CheckStore
from dat_distiller.store import DatasetStore
from dat_distiller.store.paths import AppPaths


@pytest.fixture
def store(isolated_data_dir) -> DatasetStore:
    return DatasetStore(AppPaths(root=isolated_data_dir))


@pytest.fixture
def checks(store: DatasetStore) -> CheckStore:
    return CheckStore(store.db)


def register(checks: CheckStore, severity="warning", kind="pii_found", subject="dv-1"):
    return checks.register(
        kind=kind,
        severity=severity,
        message="Email addresses found in 1 column",
        subject_type="dataset_version",
        subject_id=subject,
        details={"columns": ["email"]},
        step="upload",
    )


def test_registration_and_listing(checks: CheckStore) -> None:
    info = register(checks, severity="info", kind="imbalance")
    warn = register(checks)
    listed = checks.list_for_subject("dataset_version", "dv-1")
    assert [c.id for c in listed] == [info.id, warn.id]
    assert listed[0].acknowledged is False
    assert listed[1].details == {"columns": ["email"]}
    assert checks.list_for_subject("dataset_version", "other") == []


def test_warning_gate_and_acknowledgement_persist(checks: CheckStore) -> None:
    info = register(checks, severity="info", kind="imbalance")
    warn_a = register(checks, kind="pii_found")
    warn_b = register(checks, kind="near_copies")

    assert [c.id for c in checks.unacknowledged_warnings("dataset_version", "dv-1")] == [
        warn_a.id,
        warn_b.id,
    ]
    # acknowledging an *info* check does not satisfy a warning's absence; they
    # are independent, and info checks never appear in the gate anyway
    checks.acknowledge(info.id, note="fine")
    assert len(checks.unacknowledged_warnings("dataset_version", "dv-1")) == 2

    acked = checks.acknowledge(warn_a.id, acknowledged_by="jj", note="known, sample data")
    assert acked.acknowledged and acked.acknowledged_by == "jj" and acked.note is not None
    assert [c.id for c in checks.unacknowledged_warnings("dataset_version", "dv-1")] == [
        warn_b.id
    ]

    # Idempotent: second acknowledge keeps the first record.
    again = checks.acknowledge(warn_a.id, note="second note")
    assert again.acknowledged_at == acked.acknowledged_at and again.note == acked.note


def test_acknowledgements_queryable_for_cards(checks: CheckStore) -> None:
    warn = register(checks)
    checks.acknowledge(warn.id, note="accepted risk")
    listed = checks.list_for_subject("dataset_version", "dv-1")
    card_view = [c.to_dict() for c in listed]
    assert card_view[0]["acknowledged"] is True
    assert card_view[0]["note"] == "accepted risk"
    # and they are gone from an unacknowledged-only query
    assert checks.list_for_subject("dataset_version", "dv-1", include_acknowledged=False) == []


def test_has_check_for_once_per_subject(checks: CheckStore) -> None:
    assert not checks.has_check("cli_no_tools", "project", "p-1")
    checks.register(
        kind="cli_no_tools",
        severity="warning",
        message="claude could not be locked down",
        subject_type="project",
        subject_id="p-1",
    )
    assert checks.has_check("cli_no_tools", "project", "p-1")
    assert not checks.has_check("cli_no_tools", "project", "p-2")


def test_validation_errors(checks: CheckStore) -> None:
    with pytest.raises(ValueError):
        checks.register(
            kind="k", severity="error", message="m", subject_type="project", subject_id="p"
        )
    with pytest.raises(ValueError):
        checks.register(
            kind="k", severity="info", message="m", subject_type="widget", subject_id="p"
        )


# -- API ---------------------------------------------------------------------


def test_checks_api(client: TestClient, store: DatasetStore, checks: CheckStore) -> None:
    version = store.create_version(
        store.create_project("Checks").id, pd.DataFrame({"a": [1]})
    )
    warn = register(checks, subject=version.id)
    register(checks, severity="info", kind="imbalance", subject=version.id)

    body = client.get(
        "/api/checks", params={"subject_type": "dataset_version", "subject_id": version.id}
    ).json()
    assert body["unacknowledged_warnings"] == 1
    assert [c["kind"] for c in body["checks"]] == ["pii_found", "imbalance"]

    assert client.post(f"/api/checks/{warn.id}/acknowledge", json={"note": "ok"}).json()[
        "acknowledged"
    ] is True
    body = client.get(
        "/api/checks", params={"subject_type": "dataset_version", "subject_id": version.id}
    ).json()
    assert body["unacknowledged_warnings"] == 0

    assert client.post("/api/checks/missing/acknowledge", json={}).status_code == 404
    assert (
        client.get(
            "/api/checks", params={"subject_type": "widget", "subject_id": "x"}
        ).status_code
        == 422
    )
