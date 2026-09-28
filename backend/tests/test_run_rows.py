"""A finished Training Run's rows, rebuilt by `training.run_rows`.

A run does not train on the Dataset Version as stored. `build_setup` drops rows
with no Target value, and rows whose labelling is still below the Review
threshold, and the split it records addresses what is left. So anything acting on
a finished run has to rebuild that filtered frame — and reading the stored
version instead is silent, because the indices are all still in range.

These tests pin the guarantee in one place, and then the two consequences that
matter to a person: the plots do not 500, and the Model Bundle is the Model the
leaderboard ranked.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dat_distiller.store import DatasetStore
from dat_distiller.store.provenance import PROVENANCE_COLUMN
from dat_distiller.training.preprocess import transform_with_spec
from dat_distiller.training.run_rows import resolve_run

pytest.importorskip("sklearn")

MODEL = "logistic_regression"


# -- data --------------------------------------------------------------------


def frame_with_a_null_target(rows: int = 40) -> pd.DataFrame:
    """A learnable frame with one row that has no Target value at all.

    The Target alternates row by row, so reading it one row off is visible
    rather than plausible-looking. Row 0 has no Target, so `build_setup` drops it
    and *every* kept index shifts by one relative to the stored version — which
    is the whole point: an index that is off by one still looks like an index.
    """
    kinds = ["b" if position % 2 == 0 else "c" for position in range(rows)]
    frame = pd.DataFrame(
        {
            "feature": [float(position % 2) for position in range(rows)],
            "kind": pd.array(kinds, dtype="object"),
            PROVENANCE_COLUMN: [json.dumps({"row_origin": "uploaded"}) for _ in kinds],
        }
    )
    frame.loc[0, "kind"] = None
    return frame


# -- helpers -----------------------------------------------------------------


def wait_for_job(client: TestClient, job_id: str, timeout: float = 240.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.02)
    raise AssertionError("the training job never finished")


def train(client: TestClient, version_id: str, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "version_id": version_id,
        "target": "kind",
        "models": [MODEL],
        "seed": 5,
        **extra,
    }
    start = client.post("/api/train/run", json=body)
    assert start.status_code == 202, start.text
    job = wait_for_job(client, start.json()["id"])
    assert job["status"] == "completed", job.get("error")
    return job["result"]


@pytest.fixture()
def run(client: TestClient) -> dict[str, Any]:
    project = client.post("/api/projects", json={"name": "row space"}).json()["id"]
    store: DatasetStore = client.app.state.store
    version = store.create_version(
        project, frame_with_a_null_target(), origin="uploaded"
    ).id
    payload = train(client, version)
    return payload


# -- the guarantee -----------------------------------------------------------


def test_the_rows_a_run_worked_on_are_the_filtered_rows_not_the_stored_version(
    client: TestClient, run: dict[str, Any]
) -> None:
    """The run's indices address the frame `build_setup` produced.

    This is the invariant the whole module exists for, so it is asserted
    directly: a row index recorded by the run must select the same Target value
    whether it is read through the stored version or through the resolved frame,
    and here they must *not* — that is what makes the test meaningful.
    """
    job = client.app.state.jobs.get(run["training_run_id"])
    context = resolve_run(client.app.state.store, job)

    assert context.setup.dropped_null_target == 1
    assert len(context.frame) == len(context.source_frame) - 1

    # The held-out Target, as the run scored it.
    honest = context.test_frame["kind"].tolist()
    # The same indices read against the stored version — one row off, throughout.
    shifted = context.source_frame["kind"].iloc[context.test_indices].tolist()
    assert honest != shifted, "this test needs the index spaces to disagree"
    assert honest == context.frame["kind"].iloc[context.test_indices].tolist()

    # And the bookkeeping column is not part of what a caller has to supply.
    assert PROVENANCE_COLUMN in context.frame.columns
    assert PROVENANCE_COLUMN not in context.data_frame.columns
    assert list(context.data_frame.columns) == ["feature", "kind"]


def test_a_run_whose_split_cannot_be_reproduced_is_refused_rather_than_mismeasured(
    client: TestClient, run: dict[str, Any]
) -> None:
    """A split that no longer rebuilds is an error, not a plausible number.

    The indices are the only record of which rows a run used. If they cannot be
    reproduced from the run's own request and seed, then anything computed
    against them is a guess — so the resolver refuses, and says what to do.
    """
    job = client.app.state.jobs.get(run["training_run_id"])
    tampered = dict(job.checkpoint["run"])
    held_out = list(tampered["test_split"]["indices"])
    held_out[0] = held_out[0] + 1  # a different split, still a plausible one
    tampered["test_split"] = {**tampered["test_split"], "indices": held_out}
    job.checkpoint = {**job.checkpoint, "run": tampered}

    with pytest.raises(ValueError, match="cannot be reproduced"):
        resolve_run(client.app.state.store, job)


# -- what a person sees ------------------------------------------------------


def test_a_confusion_matrix_plot_reads_the_runs_own_rows(
    client: TestClient, run: dict[str, Any]
) -> None:
    """The plot is drawn from the run's Target, not from the version's.

    Reading the stored version gave the plot a Target value the Model had never
    seen, and the resulting `KeyError` escaped as an HTTP 500 rather than a
    refusal.
    """
    response = client.get(
        f"/api/train/runs/{run['training_run_id']}/plots",
        params={"model": MODEL, "plot": "confusion_matrix"},
    )
    assert response.status_code == 200, response.text
    plot = response.json()
    assert "error" not in plot, plot
    # Every held-out row is accounted for, and each is a class the run knows.
    matrix = np.array(plot["matrix"])
    assert int(matrix.sum()) == len(run["test_split"]["indices"])
    assert [str(c) for c in plot["labels"]] == ["str:b", "str:c"]


def test_the_bundle_ships_the_model_the_leaderboard_ranked(
    client: TestClient, run: dict[str, Any]
) -> None:
    """The exported Model is fitted on the run's training rows, not the version's.

    Before, the bundle refitted on the stored version's rows. The null-Target
    row came along as a spurious third class, the two real classes shifted
    underneath it, and the exported Model scored 0.0 on the same held-out rows
    the leaderboard had scored 1.0. A user downloading the bundle was getting a
    different Model from the one they had been shown.
    """
    response = client.get(
        f"/api/train/runs/{run['training_run_id']}/bundle",
        params={"model": MODEL, "onnx": "false"},
    )
    assert response.status_code == 200, response.text
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    metadata = json.loads(archive.read("metadata.json"))

    import joblib

    estimator = joblib.load(archive.open("model.joblib"))
    classes = [str(c) for c in next(
        e for e in run["leaderboard"] if e["model"] == MODEL
    )["classes"]]

    # Same vocabulary as the run: no third class conjured out of a null Target.
    assert len(estimator.classes_) == len(classes)
    assert metadata["classes"] == classes

    # And the same held-out accuracy, recomputed here from the run's own rows.
    job = client.app.state.jobs.get(run["training_run_id"])
    context = resolve_run(client.app.state.store, job)
    X = transform_with_spec(context.data_frame, run["preprocessing"])
    held_out = context.test_indices
    truth = np.array([classes.index(f"str:{value}") for value in context.test_frame["kind"]])
    assert (estimator.predict(X[held_out]) == truth).mean() == pytest.approx(
        metadata["primary_value"]
    )
    assert metadata["primary_value"] == run["leaderboard"][0]["primary"]["value"]


def test_the_bundles_input_schema_comes_from_the_runs_own_rows(
    client: TestClient, run: dict[str, Any]
) -> None:
    """The example row in the schema is a row the Model was actually trained on.

    The schema is the one part of the bundle a person reads before trusting it,
    so its example values have to come from the right rows and must not ask the
    caller for a provenance blob.
    """
    response = client.get(
        f"/api/train/runs/{run['training_run_id']}/bundle",
        params={"model": MODEL, "onnx": "false"},
    )
    assert response.status_code == 200, response.text
    schema = json.loads(zipfile.ZipFile(io.BytesIO(response.content)).read("input_schema.json"))

    assert [c["name"] for c in schema["columns"]] == ["feature"]
    job = client.app.state.jobs.get(run["training_run_id"])
    context = resolve_run(client.app.state.store, job)
    first = context.frame["feature"].iloc[0]
    assert float(schema["columns"][0]["example"]) == float(first)
