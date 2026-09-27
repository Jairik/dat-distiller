"""Upload CSV/Parquet/JSONL, download CSV/Parquet, Provenance opt-in."""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

SAMPLE = pd.DataFrame(
    {
        "name": ["widget", "gadget", "thing"],
        "price": [9.5, 3.0, None],
        "spam": [True, False, True],
    }
)


@pytest.fixture
def project(client: TestClient) -> dict:
    return client.post("/api/projects", json={"name": "Uploads"}).json()


def upload(client: TestClient, project_id: str, filename: str, content: bytes, **data):
    return client.post(
        f"/api/projects/{project_id}/upload",
        files={"file": (filename, content)},
        data=data,
    )


def sample_bytes(fmt: str) -> bytes:
    if fmt == "csv":
        return SAMPLE.to_csv(index=False).encode()
    if fmt == "parquet":
        buffer = io.BytesIO()
        SAMPLE.to_parquet(buffer, index=False)
        return buffer.getvalue()
    return "\n".join(SAMPLE.to_dict(orient="records") and json.dumps(r) for r in SAMPLE.to_dict(orient="records")).encode()


@pytest.mark.parametrize("fmt", ["csv", "parquet", "jsonl"])
def test_all_three_formats_round_trip(client: TestClient, project: dict, fmt: str) -> None:
    response = upload(client, project["id"], f"sample.{fmt}", sample_bytes(fmt))
    assert response.status_code == 201, response.text
    version = response.json()
    assert version["origin"] == "uploaded"
    assert version["row_count"] == 3
    assert version["provenance_summary"]["uploaded"] == 3

    downloaded = client.get(f"/api/dataset-versions/{version['id']}/download")
    back = pd.read_csv(io.BytesIO(downloaded.content))
    assert list(back.columns) == list(SAMPLE.columns)
    assert back["name"].tolist() == SAMPLE["name"].tolist()
    assert back["price"].isna().tolist() == SAMPLE["price"].isna().tolist()
    assert back["spam"].astype(bool).tolist() == SAMPLE["spam"].tolist()


def test_download_defaults_to_csv_parquet_on_request(
    client: TestClient, project: dict
) -> None:
    version = upload(
        client, project["id"], "sample.csv", sample_bytes("csv")
    ).json()
    url = f"/api/dataset-versions/{version['id']}/download"

    default = client.get(url)
    assert default.headers["content-type"].startswith("text/csv")
    assert 'filename="dataset-v1.csv"' in default.headers["content-disposition"]

    parquet = client.get(url, params={"format": "parquet"})
    assert parquet.status_code == 200
    assert parquet.content[:4] == b"PAR1"
    back = pd.read_parquet(io.BytesIO(parquet.content))
    assert len(back) == 3


def test_provenance_only_included_when_requested(client: TestClient, project: dict) -> None:
    version = upload(client, project["id"], "sample.csv", sample_bytes("csv")).json()
    url = f"/api/dataset-versions/{version['id']}/download"
    default_csv = client.get(url).text
    assert "__provenance__" not in default_csv
    with_prov = client.get(url, params={"include_provenance": "true"})
    assert "__provenance__" in with_prov.text
    # re-read with pandas and decode the column
    frame = pd.read_csv(io.BytesIO(with_prov.content))
    row_prov = json.loads(frame["__provenance__"].iloc[0])
    assert row_prov["row_origin"] == "uploaded"


def test_upload_with_parent_creates_child(client: TestClient, project: dict) -> None:
    root = upload(client, project["id"], "a.csv", sample_bytes("csv")).json()
    child = upload(
        client, project["id"], "b.csv", sample_bytes("csv"), parent_id=root["id"]
    ).json()
    assert child["parent_id"] == root["id"]
    assert (
        upload(
            client, project["id"], "c.csv", sample_bytes("csv"), parent_id="nope"
        ).status_code
        == 404
    )


def test_bad_uploads_rejected_clearly(client: TestClient, project: dict) -> None:
    assert upload(client, project["id"], "notes.txt", b"hello").status_code == 415
    broken = upload(client, project["id"], "broken.csv", b"\x80\x81\x82\x83")
    assert broken.status_code == 422  # undecodable bytes
    assert "could not be parsed" in broken.text
    assert upload(client, project["id"], "empty.csv", b"").status_code == 422
    bad_jsonl = upload(client, project["id"], "x.jsonl", b"{not json}\n")
    assert bad_jsonl.status_code == 422
