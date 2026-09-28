"""The Fidelity Report endpoint: the stored report plus the series it charts.

The report Generation already stored has the statistics (KS, chi-square,
duplicates, balance). What it does not have is the *distributions* — a p-value
tells you two columns differ, not how. Charting the preview endpoint would be
charting a sample of a sample, so this endpoint sends real binned series and
the actual correlation matrices instead. A chart here is never quietly wrong
because it only saw the first 200 rows.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dat_distiller.api.fidelity import distributions

# -- a real report, shaped exactly as build_fidelity_report writes it ----------

REPORT = {
    "ks_tests": {"age": {"statistic": 0.12, "p_value": 0.81}},
    "chi2_tests": {"plan": {"statistic": 0.4, "p_value": 0.93}},
    "correlation_drift": 0.0312,
    "exact_duplicates": {"count": 3, "fraction": 0.03},
    "near_copies": {"count": 2, "threshold": 0.05, "examples": [1, 4], "scanned": 100},
    "dropped_rows": 12,
    "dropped_ratio": 0.02,
    "balance": {"plan": {"pro": {"requested": 0.5, "actual": 0.49}}},
    "provider_derived_profile": False,
    "generated_rows": 100,
    "warnings": ["near_copies"],
}

PROFILE_DERIVED = {**REPORT, "provider_derived_profile": True, "warnings": ["provider_derived_profile"]}


@pytest.fixture()
def project(client: TestClient) -> dict:
    r = client.post("/api/projects", json={"name": "Fidelity Lab"})
    assert r.status_code == 201
    return r.json()


def upload(client: TestClient, project_id: str, body: bytes, parent_id: str | None = None) -> dict:
    r = client.post(
        f"/api/projects/{project_id}/upload",
        files={"file": ("data.csv", body, "text/csv")},
        data={"parent_id": parent_id} if parent_id else None,
    )
    assert r.status_code == 201, r.text
    return r.json()


SAMPLE_CSV = b"""age,plan,bio
20,pro,loves hiking
30,basic,hates rain
40,pro,loves hiking
50,basic,hates rain
60,pro,loves hiking
"""

GENERATED_CSV = b"""age,plan,bio
22,pro,loves hiking
33,basic,hates rain
44,pro,loves hiking
55,basic,hates rain
66,pro,loves hiking
"""


def with_report(client: TestClient, store_version: dict, report: dict = REPORT) -> None:
    """Stamp a Fidelity Report onto a version, the way Generation does."""
    store = client.app.state.store
    with store.db.connect() as conn:
        conn.execute(
            "UPDATE dataset_versions SET meta_json = ? WHERE id = ?",
            (json.dumps({"fidelity": report}), store_version["id"]),
        )


# -- the endpoint -------------------------------------------------------------


def test_returns_the_stored_report_plus_chart_series(client: TestClient, project) -> None:
    sample = upload(client, project["id"], SAMPLE_CSV)
    generated = upload(client, project["id"], GENERATED_CSV, parent_id=sample["id"])
    with_report(client, generated)

    r = client.get(f"/api/dataset-versions/{generated['id']}/fidelity")
    assert r.status_code == 200
    body = r.json()
    assert body["report"] == REPORT
    assert body["sample_version_id"] == sample["id"]

    by_column = {d["column"]: d for d in body["distributions"]}
    # numeric -> a shared-bin histogram, with edges that cover BOTH series
    age = by_column["age"]
    assert age["chart"] == "histogram"
    assert len(age["edges"]) == len(age["sample"]) + 1
    assert len(age["generated"]) == len(age["sample"])
    # the generated range (22..66) is wider, so the shared edges must span it
    assert min(age["edges"]) <= 20 and max(age["edges"]) >= 66

    # categorical -> shares that add to ~1
    plan = by_column["plan"]
    assert plan["chart"] == "bar"
    assert plan["categories"] == ["basic", "pro"]
    assert sum(plan["sample"]) == pytest.approx(1.0)
    assert sum(plan["generated"]) == pytest.approx(1.0)

    # text still gets a comparison, as category shares
    assert by_column["bio"]["chart"] == "bar"


def test_the_correlation_matrices_come_back_with_the_drift_heatmap_data(
    client: TestClient, project
) -> None:
    sample = upload(client, project["id"], b"age,spend\n20,5\n30,7\n40,9\n50,11\n")
    generated = upload(client, project["id"], b"age,spend\n22,6\n33,8\n44,10\n55,12\n", sample["id"])
    with_report(client, generated)

    corr = client.get(f"/api/dataset-versions/{generated['id']}/fidelity").json()["correlation"]
    assert corr["columns"] == ["age", "spend"]
    # both series are perfectly correlated, so the drift matrix is all zeros
    assert corr["drift"] == [[0.0, 0.0], [0.0, 0.0]]
    assert corr["max_drift"] == 0.0
    assert corr["sample"][0][1] == pytest.approx(1.0)


def test_drift_is_where_the_relationships_moved(client: TestClient, project) -> None:
    # age and spend move together in the sample, independently in the generated
    sample = upload(client, project["id"], b"age,spend\n20,5\n30,7\n40,9\n50,11\n")
    generated = upload(client, project["id"], b"age,spend\n20,50\n30,10\n40,30\n50,20\n", sample["id"])
    with_report(client, generated)

    corr = client.get(f"/api/dataset-versions/{generated['id']}/fidelity").json()["correlation"]
    assert corr["max_drift"] > 0.3
    assert any(any(cell > 0.3 for cell in row) for row in corr["drift"])


def test_a_version_without_a_report_is_404_not_an_empty_chart(client: TestClient, project) -> None:
    uploaded = upload(client, project["id"], SAMPLE_CSV)
    r = client.get(f"/api/dataset-versions/{uploaded['id']}/fidelity")
    assert r.status_code == 404
    assert "no Fidelity Report" in r.text


def test_an_unknown_version_is_404(client: TestClient) -> None:
    assert client.get("/api/dataset-versions/nope/fidelity").status_code == 404


# -- the series builder, on its own -------------------------------------------


def test_a_constant_column_charts_as_one_visible_bar() -> None:
    frame = pd.DataFrame({"plan": ["pro"] * 5, "n": [7] * 5})
    out = {d["column"]: d for d in distributions(frame, frame)}
    # a constant column would divide by a zero-width range; the edges are widened
    # by one instead, so every row lands in the first bin and the rest are empty
    assert out["n"]["edges"][:2] == [7.0, 7.05]
    assert out["n"]["sample"][0] == 5
    assert set(out["n"]["sample"][1:]) == {0}


def test_a_column_only_in_one_side_is_still_listed() -> None:
    sample = pd.DataFrame({"age": [1, 2, 3], "legacy": ["a", "b", "c"]})
    generated = pd.DataFrame({"age": [1, 2, 3], "invented": ["x", "y", "z"]})
    out = {d["column"]: d for d in distributions(sample, generated)}
    assert out["legacy"]["only_in"] == "sample"
    assert out["invented"]["only_in"] == "generated"
    assert out["age"]["only_in"] is None
    # one-sided columns still draw: the absent side is all-zero bars, and
    # `only_in` is what tells the UI the column is not comparable
    assert out["legacy"]["categories"] == ["a", "b", "c"]
    assert out["legacy"]["generated"] == [0.0, 0.0, 0.0]
    assert out["invented"]["sample"] == [0.0, 0.0, 0.0]


def test_a_long_categorical_tail_is_folded_into_other() -> None:
    sample = pd.DataFrame({"c": [f"v{i}" for i in range(40)]})
    out = {d["column"]: d for d in distributions(sample, sample)}["c"]
    assert "other" in out["categories"]
    assert len(out["categories"]) <= 13  # 12 kept + "other"
    assert sum(out["sample"]) == pytest.approx(1.0)


def test_all_missing_values_do_not_crash_the_histogram() -> None:
    frame = pd.DataFrame({"empty": [None, None], "ok": [1, 2]})
    out = {d["column"]: d for d in distributions(frame, frame)}
    # an all-missing column has nothing to chart, and says so with empty series
    assert out["empty"]["categories"] == []
    assert out["ok"]["edges"]


def test_provenance_is_never_charted() -> None:
    frame = pd.DataFrame({"age": [1, 2], "__provenance__": ["x", "y"]})
    assert [d["column"] for d in distributions(frame, frame)] == ["age"]


def test_no_sample_gives_one_sided_series(client: TestClient, project) -> None:
    # a spec-only Generation has no parent to compare against; the chart must
    # still render the generated side rather than 404 or draw nothing
    version = upload(client, project["id"], SAMPLE_CSV)
    with_report(client, version)
    body = client.get(f"/api/dataset-versions/{version['id']}/fidelity").json()
    assert body["sample_version_id"] is None
    assert all(d["only_in"] in ("generated", None) for d in body["distributions"])


def test_bin_count_is_configurable(client: TestClient, project) -> None:
    version = upload(client, project["id"], b"age\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n")
    with_report(client, version)
    body = client.get(f"/api/dataset-versions/{version['id']}/fidelity", params={"bins": 4}).json()
    age = {d["column"]: d for d in body["distributions"]}["age"]
    assert len(age["edges"]) == 5
