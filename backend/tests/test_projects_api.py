"""Projects / Dataset Versions REST API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def project(client: TestClient) -> dict:
    response = client.post("/api/projects", json={"name": "Demo"})
    assert response.status_code == 201
    return response.json()


def make_version(client: TestClient, project_id: str, **overrides) -> dict:
    body = {
        "columns": ["name", "price", "spam"],
        "rows": [
            ["widget", 9.5, True],
            ["gadget", 3.0, False],
            ["thing", None, True],
        ],
        **overrides,
    }
    response = client.post(f"/api/projects/{project_id}/dataset_versions", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def test_project_lifecycle(client: TestClient) -> None:
    created = client.post("/api/projects", json={"name": "One"}).json()
    assert created["version_count"] == 0 and created["latest_version_id"] is None
    assert client.get("/api/projects").json() == [created]
    assert client.get(f"/api/projects/{created['id']}").json() == created
    assert client.delete(f"/api/projects/{created['id']}").status_code == 204
    assert client.get(f"/api/projects/{created['id']}").status_code == 404
    assert client.delete("/api/projects/missing").status_code == 404


def test_duplicate_and_empty_names(client: TestClient) -> None:
    assert client.post("/api/projects", json={"name": "Dup"}).status_code == 201
    assert client.post("/api/projects", json={"name": "Dup"}).status_code == 409
    assert client.post("/api/projects", json={"name": "  "}).status_code == 422


def test_version_tree_exposed_as_flat_list_with_parent_ids(
    client: TestClient, project: dict
) -> None:
    root = make_version(client, project["id"])
    branch_a = make_version(client, project["id"], parent_id=root["id"])
    branch_b = make_version(client, project["id"], parent_id=root["id"])
    versions = client.get(f"/api/projects/{project['id']}/dataset_versions").json()
    assert [v["parent_id"] for v in versions] == [None, root["id"], root["id"]]
    assert [v["number"] for v in versions] == [1, 2, 3]
    updated_project = client.get(f"/api/projects/{project['id']}").json()
    assert updated_project["version_count"] == 3
    assert updated_project["latest_version_id"] == branch_b["id"]
    assert branch_a["origin"] == "uploaded"
    # uploaded rows: provenance breakdown is all uploaded
    assert root["provenance_summary"] == {
        "uploaded": 3,
        "synthetic": 0,
        "jev": 0,
        "human_reviewed": 0,
    }


def test_version_columns_inferred_on_creation(client: TestClient, project: dict) -> None:
    version = make_version(client, project["id"])
    kinds = {c["name"]: c["kind"] for c in version["columns"]}
    assert kinds == {"name": "categorical", "price": "number", "spam": "bool"}
    fetched = client.get(f"/api/dataset-versions/{version['id']}").json()
    assert fetched["row_count"] == 3
    assert client.get("/api/dataset-versions/missing").status_code == 404
    assert (
        client.post(
            f"/api/projects/{project['id']}/dataset_versions",
            json={"columns": ["a"], "rows": [[1]], "parent_id": "nope"},
        ).status_code
        == 404
    )


def test_preview_paginates_and_hides_provenance(client: TestClient, project: dict) -> None:
    version = make_version(client, project["id"])
    page0 = client.get(
        f"/api/dataset-versions/{version['id']}/preview?page=0&page_size=2"
    ).json()
    page1 = client.get(
        f"/api/dataset-versions/{version['id']}/preview?page=1&page_size=2"
    ).json()
    assert page0["total_rows"] == page1["total_rows"] == 3
    assert [r[0] for r in page0["rows"]] == ["widget", "gadget"]
    assert [r[0] for r in page1["rows"]] == ["thing"]
    assert page1["rows"][0][1] is None  # nulls come back as JSON null
    assert page1["rows"][0][2] is True
    # the reserved Provenance column is not part of the visible table
    assert all("provenance" not in str(c).lower() for c in page0["columns"])
    assert [c["name"] for c in page0["columns"]] == ["name", "price", "spam"]
    beyond = client.get(
        f"/api/dataset-versions/{version['id']}/preview?page=99&page_size=50"
    ).json()
    assert beyond["rows"] == [] and beyond["total_rows"] == 3


def test_preview_can_return_one_row_by_position(client: TestClient, project: dict) -> None:
    """`?row=` addresses a row directly, for the Review Queue's State.

    The Review Queue needs the State of the row it is showing. Paging to reach
    one row means either fetching up to 1000 rows or computing which page holds
    it, and both quietly return a *different* row than the one asked for — which
    is how every card came to display row 0's State above its own row number.
    """
    version = make_version(client, project["id"])
    base = f"/api/dataset-versions/{version['id']}/preview"
    page1 = client.get(f"{base}?page=1&page_size=1").json()
    assert [r[0] for r in page1["rows"]] == ["gadget"]

    for index, expected in enumerate(["widget", "gadget", "thing"]):
        body = client.get(f"{base}?row={index}").json()
        assert [r[0] for r in body["rows"]] == [expected]
        assert body["total_rows"] == 3
        # Same shape as a page, so the same parser works either way.
        assert [c["name"] for c in body["columns"]] == ["name", "price", "spam"]

    # A row past the end is a plain refusal, not an empty list that reads as a
    # row with no values.
    past = client.get(f"{base}?row=99")
    assert past.status_code == 422
    assert "past the end" in past.json()["detail"]
    assert client.get(f"{base}?row=-1").status_code == 422
