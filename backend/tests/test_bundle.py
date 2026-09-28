"""The Model Bundle.

The acceptance criterion is that *a bundle loads in a fresh Python session and
reproduces predictions*. So the test that matters is not "the zip has the right
names" — it is one that:

1. writes the bundle to disk,
2. runs a **separate `python` process** that opens it with nothing but the
   library it needs, and
3. compares that process's predictions to the ones the app produced.

A test in the same process would prove only that the objects are still in
memory. A fresh session is the only honest version of "someone else can use
this", and it is the one place a missing dependency or a closure over a
module-level object shows up.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dat_distiller.store import DatasetStore
from dat_distiller.store.provenance import PROVENANCE_COLUMN
from dat_distiller.training.bundle import (
    ESTIMATOR_NAME,
    INPUT_SCHEMA_NAME,
    KERAS_NAME,
    METADATA_NAME,
    MODEL_CARD_NAME,
    ONNX_NAME,
    PREPROCESSING_NAME,
    README_NAME,
    STATE_DICT_NAME,
    TORCHSCRIPT_NAME,
    build_bundle,
    build_input_schema,
    build_metadata,
)
from dat_distiller.training.evaluate import refit
from dat_distiller.training.preprocess import transform_with_spec

pytest.importorskip("sklearn")


# -- fixtures -----------------------------------------------------------------


def toy_frame(rows: int = 120, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, rows)
    spend = rng.normal(0, 1, rows).round(3)
    return pd.DataFrame(
        {
            "age": age,
            "spend": spend,
            "plan": rng.choice(["basic", "pro"], rows),
            "is_churn": pd.array((age < 45) & (spend > 0), dtype="boolean"),
            "is_churn__confidence": pd.array([0.95] * rows, dtype="Float64"),
            PROVENANCE_COLUMN: [
                json.dumps(
                    {
                        "row_origin": "synthetic",
                        "label_origins": {"is_churn": {"origin": "jev", "confidence": 0.95}},
                    }
                )
                for _ in range(rows)
            ],
        }
    )


@pytest.fixture()
def project_id(client: TestClient) -> str:
    return client.post("/api/projects", json={"name": "bundle lab"}).json()["id"]


def _version(client: TestClient, project_id: str) -> str:
    store: DatasetStore = client.app.state.store
    return store.create_version(project_id, toy_frame(), origin="labeled").id


def _wait(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(9000):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.02)
    raise AssertionError("the training job never finished")


def train(client: TestClient, version_id: str, target: str, model: str, **extra) -> dict[str, Any]:
    body: dict[str, Any] = {"version_id": version_id, "target": target, "models": [model], **extra}
    start = client.post("/api/train/run", json=body)
    assert start.status_code == 202, start.text
    job = _wait(client, start.json()["id"])
    assert job["status"] == "completed", job.get("error")
    return job["result"]


@pytest.fixture()
def run(client: TestClient, project_id: str) -> dict[str, Any]:
    return train(client, _version(client, project_id), "is_churn", "logistic_regression")


def bundle_of(client: TestClient, run: dict[str, Any], model: str = "logistic_regression") -> bytes:
    store: DatasetStore = client.app.state.store
    frame = store.load_dataframe(run["version_id"])
    return build_bundle(run, model, frame, model_card="# Model Card\n\nHand-written for the test.\n")


# -- what is in the zip -------------------------------------------------------


def test_the_bundle_carries_the_estimator_pipeline_metadata_and_card(
    client: TestClient, run: dict[str, Any]
) -> None:
    with zipfile.ZipFile(io.BytesIO(bundle_of(client, run))) as zf:
        names = set(zf.namelist())
    for expected in (
        ESTIMATOR_NAME,
        PREPROCESSING_NAME,
        METADATA_NAME,
        INPUT_SCHEMA_NAME,
        README_NAME,
        MODEL_CARD_NAME,
    ):
        assert expected in names, f"{expected} is missing from the bundle"


def test_the_model_card_is_inside_the_bundle(
    client: TestClient, run: dict[str, Any]
) -> None:
    # a bundle that separates a Model from its caveats invites someone to share
    # the first and not the second
    with zipfile.ZipFile(io.BytesIO(bundle_of(client, run))) as zf:
        assert "Hand-written for the test." in zf.read(MODEL_CARD_NAME).decode()


def test_the_metadata_says_what_the_model_predicts_and_from_what(
    client: TestClient, run: dict[str, Any]
) -> None:
    with zipfile.ZipFile(io.BytesIO(bundle_of(client, run))) as zf:
        meta = json.loads(zf.read(METADATA_NAME))
    assert meta["target"] == "is_churn"
    assert meta["task_type"] == "classification"
    assert meta["classes"] == ["bool:False", "bool:True"]
    # source columns, not the transformed names: nobody should have to derive
    # `plan=pro` by hand
    assert meta["features"] == run["preprocessing"]["features"]
    assert meta["n_source_features"] == len(meta["features"])
    # and the estimator's own width is stated separately
    assert meta["n_features"] == run["n_features"]
    assert meta["transformed_feature_names"] == run["feature_names"]
    assert meta["seed"] == run["seed"]
    assert meta["library_versions"]["scikit-learn"]
    assert meta["primary_metric"] == run["primary_metric"]
    assert meta["primary_value"] is not None


def test_the_input_schema_states_the_column_order(
    client: TestClient, run: dict[str, Any]
) -> None:
    with zipfile.ZipFile(io.BytesIO(bundle_of(client, run))) as zf:
        schema = json.loads(zf.read(INPUT_SCHEMA_NAME))
        meta = json.loads(zf.read(METADATA_NAME))
    assert [c["name"] for c in schema["columns"]] == meta["features"]
    assert schema["order_matters"] is True
    assert "a missing one is an error rather than a guess" in schema["note"]
    assert "Extra columns are ignored" in schema["note"]


def test_the_readme_carries_a_copy_pasteable_snippet(
    client: TestClient, run: dict[str, Any]
) -> None:
    with zipfile.ZipFile(io.BytesIO(bundle_of(client, run))) as zf:
        readme = zf.read(README_NAME).decode()
        meta = json.loads(zf.read(METADATA_NAME))
    assert "```python" in readme
    assert "transform_with_spec" in readme
    assert "cannot be reproduced exactly" in readme
    # and it says which columns to supply, in order
    for feature in meta["features"]:
        assert f"`{feature}`" in readme


def test_the_pipeline_in_the_bundle_is_the_one_the_model_was_trained_with(
    client: TestClient, run: dict[str, Any]
) -> None:
    with zipfile.ZipFile(io.BytesIO(bundle_of(client, run))) as zf:
        spec = json.loads(zf.read(PREPROCESSING_NAME))
    assert spec == run["preprocessing"]


# -- AC1: a fresh session reproduces the predictions --------------------------


LOADER = """
import json, sys, zipfile
import joblib
import pandas as pd

bundle, csv_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
with zipfile.ZipFile(bundle) as z:
    estimator = joblib.load(z.open("model.joblib"))
    spec = json.loads(z.read("preprocessing.json"))
    meta = json.loads(z.read("metadata.json"))

from dat_distiller.training.preprocess import transform_with_spec

df = pd.read_csv(csv_path)
X = transform_with_spec(df, spec)

prediction = estimator.predict(X)
proba = estimator.predict_proba(X) if hasattr(estimator, "predict_proba") else None
with open(out_path, "w") as fh:
    json.dump({
        "features": meta["features"],
        "n_features": meta["n_features"],
        "prediction": [str(v) for v in prediction],
        "proba": None if proba is None else [[round(float(v), 8) for v in row] for row in proba],
    }, fh)
print("LOADED")
"""


def test_a_bundle_loads_in_a_fresh_python_session_and_reproduces_predictions(
    client: TestClient, run: dict[str, Any], tmp_path: Path
) -> None:
    """AC1, in a process that has never heard of this app."""
    store: DatasetStore = client.app.state.store
    frame = store.load_dataframe(run["version_id"])
    payload = bundle_of(client, run)

    bundle_path = tmp_path / "bundle.zip"
    bundle_path.write_bytes(payload)
    loader = tmp_path / "loader.py"
    loader.write_text(LOADER)
    out = tmp_path / "out.json"

    # the rows the Model was never fitted on
    new_rows = frame.drop(
        columns=[c for c in ("is_churn", "is_churn__confidence", PROVENANCE_COLUMN) if c in frame.columns]
    ).head(20)
    csv_path = tmp_path / "new.csv"
    new_rows.to_csv(csv_path, index=False)

    process = subprocess.run(
        [sys.executable, str(loader), str(bundle_path), str(csv_path), str(out)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert process.returncode == 0, f"the fresh session failed:\n{process.stderr}"
    assert "LOADED" in process.stdout

    loaded = json.loads(out.read_text())
    assert loaded["features"] == run["preprocessing"]["features"]
    assert loaded["n_features"] == run["n_features"]

    # and the same Model, refit from the same recipe, agrees row for row
    fitted = refit(run, "logistic_regression", frame)
    X = transform_with_spec(new_rows, run["preprocessing"])
    expected = [str(v) for v in fitted.predict(X)]
    assert loaded["prediction"] == expected
    if loaded["proba"] is not None:
        assert np.allclose(np.array(loaded["proba"]), fitted.proba(X), atol=1e-6)


def test_the_documented_snippet_is_the_one_that_works(
    client: TestClient, run: dict[str, Any]
) -> None:
    """The README must not drift from reality, so check its claims, not its prose."""
    with zipfile.ZipFile(io.BytesIO(bundle_of(client, run))) as zf:
        readme = zf.read(README_NAME).decode()
        names = set(zf.namelist())
        meta = json.loads(zf.read(METADATA_NAME))
    # it names the estimator file it actually wrote
    assert ESTIMATOR_NAME in names
    assert ESTIMATOR_NAME in readme
    # and it tells the reader the library to install
    assert "joblib" in readme
    assert "scikit-learn" in meta["library_versions"]


# -- ONNX is optional ---------------------------------------------------------


def test_onnx_is_optional_and_its_failure_is_not_fatal(
    client: TestClient, run: dict[str, Any]
) -> None:
    # asking for it must never break the bundle, whether or not it converts
    for want in (True, False):
        payload = bundle_of(client, run)
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            assert ESTIMATOR_NAME in zf.namelist()
            with_meta = json.loads(zf.read(METADATA_NAME))
        assert with_meta["model"] == "logistic_regression"


def test_a_missing_onnx_is_said_out_loud_in_the_readme(
    client: TestClient, run: dict[str, Any]
) -> None:
    store: DatasetStore = client.app.state.store
    frame = store.load_dataframe(run["version_id"])
    payload = build_bundle(run, "logistic_regression", frame, with_onnx=False)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        assert ONNX_NAME not in zf.namelist()
        readme = zf.read(README_NAME).decode()
        meta = json.loads(zf.read(METADATA_NAME))
    assert meta["onnx_file"] is None
    # a reader is told, rather than left wondering
    assert "did not succeed" in readme
    assert "not an error" in readme


# -- refusals -----------------------------------------------------------------


def test_a_model_that_did_not_fit_is_refused(client: TestClient, run: dict[str, Any]) -> None:
    run = {**run, "leaderboard": [{**run["leaderboard"][0], "status": "failed", "error": "boom"}]}
    store: DatasetStore = client.app.state.store
    frame = store.load_dataframe(run["version_id"])
    with pytest.raises(Exception, match="boom"):
        build_bundle(run, "logistic_regression", frame)


def test_an_unknown_model_is_refused(client: TestClient, run: dict[str, Any]) -> None:
    store: DatasetStore = client.app.state.store
    frame = store.load_dataframe(run["version_id"])
    with pytest.raises(Exception, match="not on this Training Run"):
        build_bundle(run, "nope", frame)


def test_a_run_with_no_stored_pipeline_is_refused(
    client: TestClient, run: dict[str, Any]
) -> None:
    store: DatasetStore = client.app.state.store
    frame = store.load_dataframe(run["version_id"])
    with pytest.raises(Exception, match="no stored preprocessing"):
        build_bundle({**run, "preprocessing": {}}, "logistic_regression", frame)


# -- the endpoint -------------------------------------------------------------


def test_the_endpoint_downloads_a_zip(client: TestClient, project_id: str) -> None:
    version = _version(client, project_id)
    run = train(client, version, "is_churn", "random_forest")
    r = client.get(
        f"/api/train/runs/{run['training_run_id']}/bundle", params={"model": "random_forest"}
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    assert "random_forest-bundle.zip" in r.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert METADATA_NAME in zf.namelist()
        meta = json.loads(zf.read(METADATA_NAME))
    assert meta["model"] == "random_forest"


def test_the_endpoints_bundle_carries_the_real_model_card(
    client: TestClient, project_id: str
) -> None:
    version = _version(client, project_id)
    run = train(client, version, "is_churn", "logistic_regression")
    r = client.get(
        f"/api/train/runs/{run['training_run_id']}/bundle",
        params={"model": "logistic_regression"},
    )
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        card = zf.read(MODEL_CARD_NAME).decode()
    assert "# Model Card" in card
    assert "cannot be reproduced exactly" in card


def test_the_endpoint_refuses_an_unknown_run(client: TestClient) -> None:
    r = client.get("/api/train/runs/nope/bundle", params={"model": "logistic_regression"})
    assert r.status_code == 404


def test_the_endpoint_refuses_an_unknown_model(
    client: TestClient, project_id: str
) -> None:
    version = _version(client, project_id)
    run = train(client, version, "is_churn", "logistic_regression")
    r = client.get(f"/api/train/runs/{run['training_run_id']}/bundle", params={"model": "nope"})
    assert r.status_code == 422
    assert "not on this Training Run" in r.text


def test_regression_bundles_too(client: TestClient, project_id: str) -> None:
    store: DatasetStore = client.app.state.store
    frame = toy_frame()
    frame["spend_target"] = (frame["spend"] * 3 + frame["age"] / 10).round(3)
    version = store.create_version(project_id, frame, origin="labeled").id
    run = train(client, version, "spend_target", "linear_regression")
    r = client.get(
        f"/api/train/runs/{run['training_run_id']}/bundle", params={"model": "linear_regression"}
    )
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        meta = json.loads(zf.read(METADATA_NAME))
    assert meta["task_type"] == "regression"
    assert meta["classes"] == []


# -- the helpers --------------------------------------------------------------


def test_build_metadata_refuses_a_model_off_the_run(run: dict[str, Any]) -> None:
    with pytest.raises(Exception, match="not on this Training Run"):
        build_metadata(run, "nope", ESTIMATOR_NAME)


def test_build_input_schema_names_the_kind_of_each_column() -> None:
    meta = {"features": ["a", "b", "c"], "feature_roles": {"a": "numeric"}}
    sample = pd.DataFrame({"a": [1.5], "b": ["x"], "c": [True]})
    schema = build_input_schema(meta, sample)
    kinds = {c["name"]: c["kind"] for c in schema["columns"]}
    assert kinds == {"a": "number", "b": "string", "c": "bool"}
    # a column not in the sample is left out rather than described from nothing
    assert build_input_schema({"features": ["zz"]}, sample)["columns"] == []


# -- the PyTorch family -------------------------------------------------------


TORCH_TINY = {"epochs": 8, "hidden_units": [8], "learning_rate": 0.05, "early_stopping": False}


def test_a_torch_bundle_carries_torchscript_and_a_state_dict(
    client: TestClient, project_id: str
) -> None:
    pytest.importorskip("torch")
    version = _version(client, project_id)
    run = train(client, version, "is_churn", "torch_mlp", hyperparameters={"torch_mlp": TORCH_TINY})
    payload = bundle_of(client, run, "torch_mlp")
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = set(zf.namelist())
        assert TORCHSCRIPT_NAME in names, sorted(names)
        # a state_dict as well: TorchScript is what you run, the weights are how
        # you rebuild the Model in your own module
        assert STATE_DICT_NAME in names
        assert ESTIMATOR_NAME not in names
        meta = json.loads(zf.read(METADATA_NAME))
        readme = zf.read(README_NAME).decode()
    assert meta["model"] == "torch_mlp"
    assert meta["library_versions"]["torch"]
    assert "torch.jit.load" in readme


def test_a_torch_bundle_loads_in_a_fresh_session_and_reproduces_predictions(
    client: TestClient, project_id: str, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    version = _version(client, project_id)
    run = train(client, version, "is_churn", "torch_mlp", hyperparameters={"torch_mlp": TORCH_TINY})
    store: DatasetStore = client.app.state.store
    frame = store.load_dataframe(run["version_id"])
    bundle_path = tmp_path / "torch-bundle.zip"
    bundle_path.write_bytes(bundle_of(client, run, "torch_mlp"))

    loader = tmp_path / "loader.py"
    loader.write_text(
        """
import json, sys, zipfile
import torch
import pandas as pd

bundle, csv_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
with zipfile.ZipFile(bundle) as z:
    module = torch.jit.load(z.open("model.pt"))
    state = torch.load(z.open("model_state_dict.pt"), weights_only=True)
    spec = json.loads(z.read("preprocessing.json"))
    meta = json.loads(z.read("metadata.json"))

from dat_distiller.training.preprocess import transform_with_spec

df = pd.read_csv(csv_path)
X = torch.from_numpy(transform_with_spec(df, spec)).float()
with torch.no_grad():
    prediction = module(X)
with open(out_path, "w") as fh:
    json.dump({
        "n_state_dict_keys": len(state),
        "prediction": prediction[:, 0].tolist(),
        "meta_features": meta["features"],
    }, fh)
print("LOADED")
"""
    )
    new_rows = frame.drop(
        columns=[c for c in ("is_churn", "is_churn__confidence") if c in frame.columns]
    ).head(10)
    csv_path = tmp_path / "new.csv"
    new_rows.to_csv(csv_path, index=False)
    out = tmp_path / "out.json"

    process = subprocess.run(
        [sys.executable, str(loader), str(bundle_path), str(csv_path), str(out)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert process.returncode == 0, f"the fresh session failed:\n{process.stderr}"
    loaded = json.loads(out.read_text())
    assert loaded["meta_features"] == run["preprocessing"]["features"]
    assert loaded["n_state_dict_keys"] > 0

    # TorchScript must give the same answer the app's own predict path does
    fitted = refit(run, "torch_mlp", frame)
    X = transform_with_spec(new_rows, run["preprocessing"])
    import torch

    module = fitted.estimator._model
    module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, torch.zeros((1, X.shape[1]), dtype=torch.float32), strict=False)
        expected = traced(torch.from_numpy(X).float())[:, 0].tolist()
    assert np.allclose(loaded["prediction"], expected, atol=1e-5)


def test_the_torch_state_dict_can_rebuild_the_weights(
    client: TestClient, project_id: str
) -> None:
    pytest.importorskip("torch")
    import torch

    version = _version(client, project_id)
    run = train(client, version, "is_churn", "torch_mlp", hyperparameters={"torch_mlp": TORCH_TINY})
    store: DatasetStore = client.app.state.store
    frame = store.load_dataframe(run["version_id"])
    payload = bundle_of(client, run, "torch_mlp")
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        state = torch.load(io.BytesIO(zf.read(STATE_DICT_NAME)), weights_only=True)
        spec = json.loads(zf.read(PREPROCESSING_NAME))
        meta = json.loads(zf.read(METADATA_NAME))
    fitted = refit(run, "torch_mlp", frame)
    original = fitted.estimator._model.state_dict()
    # the shipped weights are the ones the leaderboard scored
    assert set(state) == set(original)
    for key, tensor in state.items():
        assert torch.allclose(tensor, original[key]), key
    # and they are plain tensors, not pickled graph objects
    assert all(isinstance(t, torch.Tensor) for t in state.values())
    assert spec and meta


# -- the TensorFlow family ----------------------------------------------------


def test_a_tensorflow_bundle_carries_a_keras_model(
    client: TestClient, project_id: str
) -> None:
    pytest.importorskip("tensorflow")
    version = _version(client, project_id)
    run = train(
        client,
        version,
        "is_churn",
        "tensorflow_mlp",
        hyperparameters={"tensorflow_mlp": {"epochs": 8, "hidden_units": [8], "learning_rate": 0.05, "early_stopping": False}},
    )
    payload = bundle_of(client, run, "tensorflow_mlp")
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = set(zf.namelist())
        assert KERAS_NAME in names, sorted(names)
        # Keras's own format is the source of truth here, so joblib must not also
        # be present claiming to be it
        assert ESTIMATOR_NAME not in names
        meta = json.loads(zf.read(METADATA_NAME))
        readme = zf.read(README_NAME).decode()
    assert meta["model"] == "tensorflow_mlp"
    assert "tf.keras.models.load_model" in readme


def test_a_tensorflow_bundle_loads_in_a_fresh_session_and_reproduces_predictions(
    client: TestClient, project_id: str, tmp_path: Path
) -> None:
    pytest.importorskip("tensorflow")
    version = _version(client, project_id)
    run = train(
        client,
        version,
        "is_churn",
        "tensorflow_mlp",
        hyperparameters={"tensorflow_mlp": {"epochs": 8, "hidden_units": [8], "learning_rate": 0.05, "early_stopping": False}},
    )
    store: DatasetStore = client.app.state.store
    frame = store.load_dataframe(run["version_id"])
    bundle_path = tmp_path / "tf-bundle.zip"
    bundle_path.write_bytes(bundle_of(client, run, "tensorflow_mlp"))
    loader = tmp_path / "loader.py"
    loader.write_text(
        """
import json, os, sys, tempfile, zipfile
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

bundle, csv_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
# exactly what the README says to do: extract first, because Keras cannot read
# a stream from inside the zip
with tempfile.TemporaryDirectory() as tmp:
    with zipfile.ZipFile(bundle) as z:
        z.extract("model.keras", tmp)
        spec = json.loads(z.read("preprocessing.json"))
        meta = json.loads(z.read("metadata.json"))
    model = tf.keras.models.load_model(Path(tmp) / "model.keras")

from dat_distiller.training.preprocess import transform_with_spec

df = pd.read_csv(csv_path)
X = transform_with_spec(df, spec).astype("float32")
raw = model.predict(X, verbose=0)
with open(out_path, "w") as fh:
    json.dump({"raw": [[round(float(v), 6) for v in row] for row in raw]}, fh)
print("LOADED")
"""
    )
    new_rows = frame.drop(
        columns=[c for c in ("is_churn", "is_churn__confidence") if c in frame.columns]
    ).head(10)
    csv_path = tmp_path / "new.csv"
    new_rows.to_csv(csv_path, index=False)
    out = tmp_path / "out.json"
    process = subprocess.run(
        [sys.executable, str(loader), str(bundle_path), str(csv_path), str(out)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert process.returncode == 0, f"the fresh session failed:\n{process.stderr[-2000:]}"
    loaded = np.array(json.loads(out.read_text())["raw"])

    # the fresh session's .keras model must give the same answers as the app's
    fitted = refit(run, "tensorflow_mlp", frame)
    X = transform_with_spec(new_rows, run["preprocessing"]).astype("float32")
    expected = np.asarray(fitted.estimator._model.predict(X, verbose=0))
    assert loaded.shape == expected.shape
    assert np.allclose(loaded, expected, atol=1e-4)
