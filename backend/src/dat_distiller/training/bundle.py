"""Model Bundle: one zip that a fresh Python session can load and predict with.

A bundle has to answer three questions for someone who has never met this app:

1. **What does it predict, from what?** ``metadata.json`` — the Target, the Task
   Type, the feature list in the order the matrix expects, the class mapping, the
   seed, and the library versions.
2. **How do I get it to predict?** ``README.md`` — a copy-pasteable snippet, one
   per Model family, that works without this repository.
3. **Is this the Model I think it is?** ``model_card.md`` — the Model Card, with
   its limitations, shipped inside the bundle rather than beside it. A bundle
   that separates a Model from its caveats invites someone to share the first
   and not the second.

The preprocessing pipeline is a real member of the bundle, not something the
loader has to reconstruct: it is stored as plain JSON (``preprocessing.json``) and
applied by the snippet. Re-deriving it from the request would let the column set
drift and quietly produce predictions from a differently-transformed matrix.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import EvaluationError, NotOnRunError, load_model, refit

#: The file names inside a bundle, so a test and the README agree with reality.
README_NAME = "README.md"
METADATA_NAME = "metadata.json"
PREPROCESSING_NAME = "preprocessing.json"
MODEL_CARD_NAME = "model_card.md"
ESTIMATOR_NAME = "model.joblib"
KERAS_NAME = "model.keras"
TORCHSCRIPT_NAME = "model.pt"
STATE_DICT_NAME = "model_state_dict.pt"
ONNX_NAME = "model.onnx"
INPUT_SCHEMA_NAME = "input_schema.json"


class BundleError(EvaluationError):
    """A bundle cannot be produced, or produced faithfully."""


def _frame(run: Mapping[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """The matrix the Model was fitted on, rebuilt from the recorded split."""
    rows = (run.get("training_split") or {}).get("rows")
    if rows is None:
        raise BundleError("this Training Run does not record its training split")
    indices = [int(i) for i in rows]
    if not indices:
        raise BundleError("this Training Run records an empty training split")
    if max(indices) >= len(frame):
        raise BundleError("the stored training split does not fit the data")
    return frame.iloc[indices]


def build_metadata(
    run: Mapping[str, Any],
    model_name: str,
    estimator_name: str,
    *,
    extras: Mapping[str, str] | None = None,
    onnx: str | None = None,
) -> dict[str, Any]:
    """Everything needed to predict, and nothing that is not."""
    setup = run.get("setup") or {}
    entry = next(
        (e for e in run.get("leaderboard", []) if e.get("model") == model_name), None
    )
    if entry is None:
        raise NotOnRunError(f"Model {model_name!r} is not on this Training Run")
    pipeline = run.get("preprocessing") or {}
    # The columns a reader must supply are the pipeline's *source* features. The
    # transformed names (`plan=pro`, `age_scaled`) are an internal detail, and
    # asking someone to derive them by hand is exactly the mistake that produces
    # silently wrong predictions.
    features = list(pipeline.get("features") or [])
    output_names = list(run.get("feature_names") or pipeline.get("output_names") or [])
    return {
        "format_version": 1,
        "name": entry.get("label") or model_name,
        "model": model_name,
        "estimator_file": estimator_name,
        "target": setup.get("target") or run.get("target"),
        "task_type": entry.get("task_type") or run.get("task_type"),
        "classes": list(entry.get("classes") or []),
        "features": features,
        "feature_names": features,
        "feature_roles": dict(pipeline.get("roles") or {}),
        "n_source_features": len(features),
        "n_features": int(run.get("n_features") or len(output_names)),
        "transformed_feature_names": output_names,
        "seed": run.get("seed"),
        "seeds": dict(run.get("seeds") or {}),
        "hyperparameters": dict(entry.get("hyperparameters") or {}),
        "primary_metric": run.get("primary_metric"),
        "primary_value": (entry.get("primary") or {}).get("value"),
        "metrics": _metric_values(entry.get("metrics") or {}),
        "library_versions": dict(run.get("library_versions") or {}),
        "dataset_version_id": run.get("version_id"),
        "training_run_id": run.get("training_run_id"),
        "test_split": {
            "rows": (run.get("test_split") or {}).get("rows"),
            "size": (run.get("test_split") or {}).get("test_size"),
            "stratified": (run.get("test_split") or {}).get("stratified"),
        },
        "onnx_file": onnx,
        "optional_extras": dict(extras or {}),
    }


def build_input_schema(
    metadata: Mapping[str, Any], sample: pd.DataFrame
) -> dict[str, Any]:
    """A CSV author's contract: which columns, and what each one looks like.

    Predicting with a DataFrame whose columns are in the wrong order silently
    produces nonsense, so the order is stated rather than assumed.
    """
    columns = []
    for name in metadata.get("features", []):
        if name not in sample.columns:
            continue
        series = sample[name]
        kind = "number"
        if pd.api.types.is_bool_dtype(series):
            kind = "bool"
        elif pd.api.types.is_numeric_dtype(series):
            kind = "number"
        else:
            kind = "string"
        columns.append(
            {
                "name": name,
                "kind": kind,
                "role": (metadata.get("feature_roles") or {}).get(name),
                "example": _plain(series.iloc[0]) if len(series) else None,
            }
        )
    return {
        "columns": columns,
        "order_matters": True,
        "note": (
            "Give the Model these columns, in this order. Extra columns are ignored; "
            "a missing one is an error rather than a guess."
        ),
    }


def _metric_values(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """The bare number for each metric that has one.

    Metrics arrive as `{value, reason}` but a leaderboard also carries scalar
    extras alongside them (a confusion matrix, support counts), so the shape is
    checked rather than assumed — otherwise exporting a bundle crashes on a
    perfectly ordinary leaderboard row.
    """
    out: dict[str, Any] = {}
    for name, value in metrics.items():
        if name == "confusion_matrix":
            continue
        out[name] = value.get("value") if isinstance(value, Mapping) else value
    return out


def _plain(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


# -- estimator serialisation --------------------------------------------------


def _serialise_estimator(
    model_name: str,
    fitted: Any,
    buffer: io.BytesIO,
    n_features: int,
) -> str:
    """Write the fitted Model and return the file name it landed under.

    The format is chosen by the Model family, because that is what the reader
    needs: joblib for scikit-learn and LightGBM, ``.keras`` for TensorFlow,
    TorchScript plus a ``state_dict`` for PyTorch.
    """
    estimator = fitted.estimator
    # `spec.library` is the display name ("scikit-learn") and `spec.extra` is the
    # canonical key ("sklearn"); dispatch on the extra so a rename of the display
    # name cannot quietly change which format an estimator is written in.
    extra = str(fitted.spec.extra or fitted.spec.library)
    if extra == "torch":
        return _serialise_torch(estimator, buffer, n_features)
    if extra in {"tensorflow", "keras"}:
        return _serialise_keras(estimator, buffer)
    import joblib

    # scikit-learn and LightGBM both round-trip through joblib, and LightGBM's
    # own text format does not carry the fitted booster's parameters as reliably
    joblib.dump(estimator, buffer)
    return ESTIMATOR_NAME


def _serialise_torch(estimator: Any, buffer: io.BytesIO, n_features: int) -> str:
    """TorchScript, plus the ``state_dict`` as a second way back in.

    TorchScript is what you *run*; the ``state_dict`` is what you *rebuild with*
    if you would rather write your own module. Shipping only one of them leaves
    someone unable to get their Model back.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - guarded by the registry
        raise BundleError("this Model needs the optional extra 'torch'") from exc
    module = getattr(estimator, "_model", None)
    if module is None:
        raise BundleError("this Model was never fitted, so there is nothing to export")
    module.eval()
    example = torch.zeros((1, n_features), dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(module, example, strict=False)
    # TorchScript is deprecated in torch 2.x in favour of torch.export, but it is
    # what a `torch.jit.load` reader in any environment can still open, and the
    # state_dict is shipped alongside for anyone who would rather not rely on it.
    # Switching to torch.export is a deliberate change of the bundle's contract,
    # not a cleanup.
    torch.jit.save(traced, buffer)
    return TORCHSCRIPT_NAME


def _state_dict(estimator: Any, buffer: io.BytesIO) -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover - guarded by the registry
        return
    module = getattr(estimator, "_model", None)
    if module is not None:
        torch.save(module.state_dict(), buffer)


def _serialise_keras(estimator: Any, buffer: io.BytesIO) -> str:
    model = getattr(estimator, "_model", None)
    if model is None:
        raise BundleError("this Model was never fitted, so there is nothing to export")
    import tempfile
    from pathlib import Path

    # Keras insists on a path; write to a temp file and read the bytes back
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.keras"
        model.save(path)
        buffer.write(path.read_bytes())
    return KERAS_NAME


def _try_onnx(estimator: Any, n_features: int) -> bytes | None:
    """ONNX, if it converts. Its failure is not the bundle's failure."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        onx = convert_sklearn(
            estimator, initial_types=[("input", FloatTensorType([None, n_features]))]
        )
        return onx.SerializeToString()
    except Exception:  # noqa: BLE001 - optional, and genuinely optional
        return None


# -- the README ---------------------------------------------------------------


_LOAD_SNIPPETS = {
    "sklearn": '''import json, joblib
import numpy as np, pandas as pd
from zipfile import ZipFile

with ZipFile("{bundle}") as z:
    model = joblib.load(z.open("model.joblib"))
    spec = json.loads(z.read("preprocessing.json"))
    meta = json.loads(z.read("metadata.json"))

# Apply the fitted preprocessing. Do not re-derive it: the column set and the
# order are what the Model was trained on.
from dat_distiller.training.preprocess import transform_with_spec
X = transform_with_spec(df, spec)

prediction = model.predict(X)
proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
''',
    "lightgbm": '''import json, joblib
import numpy as np, pandas as pd
from zipfile import ZipFile

with ZipFile("{bundle}") as z:
    model = joblib.load(z.open("model.joblib"))
    spec = json.loads(z.read("preprocessing.json"))
    meta = json.loads(z.read("metadata.json"))

from dat_distiller.training.preprocess import transform_with_spec
X = transform_with_spec(df, spec)

prediction = model.predict(X)
proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
''',
    "torch": '''import json
import numpy as np, pandas as pd
import torch
from zipfile import ZipFile

with ZipFile("{bundle}") as z:
    module = torch.jit.load(z.open("model.pt"))
    spec = json.loads(z.read("preprocessing.json"))
    meta = json.loads(z.read("metadata.json"))

from dat_distiller.training.preprocess import transform_with_spec
X = torch.from_numpy(transform_with_spec(df, spec)).float()

with torch.no_grad():
    prediction = module(X)
''',
    "tensorflow": '''import json, tempfile
from pathlib import Path
import numpy as np, pandas as pd
import tensorflow as tf
from zipfile import ZipFile

# Keras needs a real file on disk; it cannot load from a stream inside the zip.
with tempfile.TemporaryDirectory() as tmp:
    with ZipFile("{bundle}") as z:
        z.extract("model.keras", tmp)
        spec = json.loads(z.read("preprocessing.json"))
        meta = json.loads(z.read("metadata.json"))
    model = tf.keras.models.load_model(Path(tmp) / "model.keras")

from dat_distiller.training.preprocess import transform_with_spec
X = transform_with_spec(df, spec).astype("float32")

prediction = model.predict(X, verbose=0)
''',
}


def _readme(metadata: Mapping[str, Any], family: str, bundle_name: str) -> str:
    snippet = _LOAD_SNIPPETS.get(
        family,
        "The estimator is stored with joblib. Load it and call `.predict(X)`.\n",
    ).format(bundle=bundle_name)
    features = metadata.get("features", [])
    n_features = metadata.get("n_source_features")
    primary = metadata.get("primary_metric")
    value = _plain(metadata.get("primary_value"))
    onnx_file = metadata.get("onnx_file")

    lines: list[str] = [
        f"# {metadata.get('name')} — Model Bundle",
        "",
        f"- **Model**: `{metadata.get('model')}` ({family})",
        f"- **Target**: `{metadata.get('target')}`",
        f"- **Task Type**: {metadata.get('task_type')}",
        f"- **Seed**: {metadata.get('seed')}",
        f"- **Ranked by**: {primary} = {value}",
        f"- **Trained on Dataset Version**: `{metadata.get('dataset_version_id')}`",
        "",
        "## What is in here",
        "",
        "| File | What it is |",
        "|---|---|",
        "| `model.*` | the fitted estimator |",
        "| `preprocessing.json` | the fitted preprocessing pipeline, as plain JSON |",
        "| `metadata.json` | Target, Task Type, features, class mapping, seeds, versions |",
        "| `input_schema.json` | which columns to give it, in which order |",
        "| `model_card.md` | the Model Card, with its limitations |",
        "",
        "## Features, in order",
        "",
        *(f"{i + 1}. `{name}`" for i, name in enumerate(features)),
        "",
        (
            f"Give it these {n_features} column(s) in this order; "
            "a missing column is an error, not a guess."
        ),
        "",
        "## Loading it",
        "",
        "```python",
        snippet.rstrip(),
        "```",
        "",
        "## Reproducibility",
        "",
        (
            "Rows written by a Provider cannot be reproduced exactly. The recorded seed reproduces "
            "the sampling and the split, not the text. Read `model_card.md` before you rely on "
            "this."
        ),
        "",
        "## ONNX",
        "",
    ]
    if onnx_file:
        lines.append(
            f"`{onnx_file}` is included for runtimes that need it. It is optional and was produced "
            "on a best-effort basis; the estimator above is the source of truth."
        )
    else:
        lines.append(
            "ONNX conversion did not succeed for this Model, so no `model.onnx` is included. "
            "That is not an error — the estimator above is the source of truth."
        )
    lines.append("")
    return "\n".join(lines)


# -- the bundle ---------------------------------------------------------------


def build_bundle(
    run: Mapping[str, Any],
    model_name: str,
    frame: pd.DataFrame,
    *,
    model_card: str | None = None,
    with_onnx: bool = True,
) -> bytes:
    """Produce the bundle as bytes.

    The Model is refitted on its recorded training split first, because a
    Training Run records *what it fitted*, not the estimator — the same
    deterministic recipe as `load_model`, so the exported Model is the one the
    leaderboard scored.
    """
    spec, _ = load_model(run, model_name)
    fitted = refit(run, model_name, frame)
    sample = _frame(run, frame)

    n_features = int(run.get("n_features") or len(run.get("feature_names") or []))
    estimator_buffer = io.BytesIO()
    estimator_name = _serialise_estimator(model_name, fitted, estimator_buffer, n_features)

    onnx_bytes = _try_onnx(fitted.estimator, n_features) if with_onnx else None

    metadata = build_metadata(
        run,
        model_name,
        estimator_name,
        onnx=ONNX_NAME if onnx_bytes else None,
    )
    schema = build_input_schema(metadata, sample)
    pipeline = run.get("preprocessing")
    if not isinstance(pipeline, Mapping) or not pipeline:
        raise BundleError("this Training Run has no stored preprocessing to bundle")

    family = str(spec.extra or spec.library)
    bundle_name = f"{model_name}-bundle.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(estimator_name, estimator_buffer.getvalue())
        if estimator_name == TORCHSCRIPT_NAME:
            state_dict = io.BytesIO()
            _state_dict(fitted.estimator, state_dict)
            zf.writestr(STATE_DICT_NAME, state_dict.getvalue())
        zf.writestr(PREPROCESSING_NAME, json.dumps(pipeline, indent=2, default=str))
        zf.writestr(METADATA_NAME, json.dumps(metadata, indent=2, default=str))
        zf.writestr(INPUT_SCHEMA_NAME, json.dumps(schema, indent=2, default=str))
        zf.writestr(README_NAME, _readme(metadata, family, bundle_name))
        if model_card:
            zf.writestr(MODEL_CARD_NAME, model_card)
        if onnx_bytes:
            zf.writestr(ONNX_NAME, onnx_bytes)
    return buffer.getvalue()


__all__ = [
    "ESTIMATOR_NAME",
    "INPUT_SCHEMA_NAME",
    "KERAS_NAME",
    "METADATA_NAME",
    "MODEL_CARD_NAME",
    "ONNX_NAME",
    "PREPROCESSING_NAME",
    "README_NAME",
    "STATE_DICT_NAME",
    "TORCHSCRIPT_NAME",
    "BundleError",
    "build_bundle",
    "build_input_schema",
    "build_metadata",
]
