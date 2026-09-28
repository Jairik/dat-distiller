"""The Model registry, the metrics, the ranking — and the split guarantee.

The split tests are the important ones. A TrainingMatrix can only be built with
``TrainingMatrix.from_split``, which reads the training split and nothing else,
and the tuning search accepts nothing but a TrainingMatrix. These tests pin
both: they check that a Training Matrix built from a held-out split contains
*only* the training rows, that the constructor refuses to be called directly,
and that tuning reports exactly which rows it searched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dat_distiller.training import trainers
from dat_distiller.training.preprocess import fit_preprocessor
from dat_distiller.training.setup import TrainingSetupRequest, build_setup
from dat_distiller.training.trainers import (
    METRICS,
    MODEL_SPECS,
    ModelError,
    ModelUnavailableError,
    TrainingMatrix,
    UnsupportedModelError,
    assert_held_out,
    available_models,
    default_primary_metric,
    evaluate_classification,
    evaluate_regression,
    library_versions,
    metric_value,
    metrics_for,
    rank_leaderboard,
    resolve_hyperparameters,
    resolve_models,
    resolve_primary_metric,
)

pytest.importorskip("sklearn")


# -- fixtures ---------------------------------------------------------------


def classification_frame(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, rows),
            "spend": rng.normal(0, 1, rows).round(3),
            "region": rng.choice(["north", "south", "east"], rows),
        }
    )


def training_matrix(frame: pd.DataFrame, target: str = "region", **request_kwargs):
    request = TrainingSetupRequest(version_id="v1", target=target, **request_kwargs)
    resolved = build_setup(request, frame)
    setup = resolved.setup
    pipeline = fit_preprocessor(
        resolved.train_frame(), features=setup.feature_columns
    )
    return setup, pipeline, TrainingMatrix.from_split(
        frame=resolved.frame, spec=pipeline.to_dict(), setup=setup
    )


# -- the registry ------------------------------------------------------------

#: Every Model the issue asks for, and the Task Types it must declare.
EXPECTED = {
    "logistic_regression": ("classification",),
    "linear_regression": ("regression",),
    "svm": ("classification", "regression"),
    "random_forest": ("classification", "regression"),
    "gradient_boosting": ("classification", "regression"),
    "knn": ("classification", "regression"),
    "naive_bayes": ("classification",),
    "lightgbm": ("classification", "regression"),
    "torch_mlp": ("classification", "regression"),
    "tensorflow_mlp": ("classification", "regression"),
}

#: The Models behind a heavy extra: the registry always describes them, but
#: whether *this* machine can build one is exactly what `is_available` answers.
MAYBE_MISSING = ("torch_mlp", "tensorflow_mlp")


def test_every_required_model_is_registered_with_its_task_types() -> None:
    assert set(MODEL_SPECS) == set(EXPECTED)
    for name, task_types in EXPECTED.items():
        assert MODEL_SPECS[name].task_types == task_types, name


def test_the_registry_says_which_optional_extra_each_model_needs() -> None:
    assert {spec.extra for spec in MODEL_SPECS.values()} == {
        "sklearn",
        "lightgbm",
        "torch",
        "tensorflow",
    }
    assert MODEL_SPECS["lightgbm"].library == "lightgbm"
    assert MODEL_SPECS["torch_mlp"].library == "torch"
    assert MODEL_SPECS["tensorflow_mlp"].library == "tensorflow"
    assert all(
        spec.library == "scikit-learn"
        for name, spec in MODEL_SPECS.items()
        if name not in ("lightgbm", *MAYBE_MISSING)
    )


def test_the_registry_reports_which_models_this_install_can_run() -> None:
    for spec in MODEL_SPECS.values():
        if spec.name in MAYBE_MISSING:
            continue
        assert trainers.is_available(spec), f"{spec.name} should be importable here"
        assert spec.name in available_models(spec.task_types[0])
    for name in MAYBE_MISSING:
        spec = MODEL_SPECS[name]
        # Both answers come from the same probe, whether the extra is here or not.
        assert (name in available_models(spec.task_types[0])) is trainers.is_available(spec), name
    assert "lightgbm" in available_models("regression")
    # Naive Bayes is classification only, so it is never offered for regression.
    assert "naive_bayes" not in available_models("regression")
    assert "naive_bayes" in available_models("classification")


def test_an_unknown_model_is_refused_with_the_available_names() -> None:
    with pytest.raises(ModelError, match="unknown Model 'catboost'"):
        trainers.get_spec("catboost")


def test_resolving_models_defaults_to_every_supported_installed_model() -> None:
    chosen = resolve_models(None, "classification")
    # Derived from the registry rather than written out, so the expectation does
    # not depend on which optional extras happen to be installed: `models: None`
    # means "everything this install can run", and that set moves.
    assert [spec.name for spec in chosen] == [
        name
        for name, spec in MODEL_SPECS.items()
        if name != "linear_regression" and trainers.is_available(spec)
    ]
    assert all(spec.supports("classification") for spec in chosen)
    assert all(trainers.is_available(spec) for spec in chosen)


def test_a_model_that_cannot_do_the_task_type_is_refused_readably() -> None:
    with pytest.raises(UnsupportedModelError) as excinfo:
        resolve_models(["naive_bayes"], "regression")
    message = str(excinfo.value)
    assert "naive_bayes" in message
    assert "classification" in message and "regression" in message
    assert "linear_regression" in message, "the error should say what is allowed"


def test_a_model_whose_extra_is_missing_names_the_extra() -> None:
    spec = MODEL_SPECS["lightgbm"]
    original = trainers.is_installed
    try:
        trainers.is_installed = lambda module: module != "lightgbm"  # type: ignore[assignment]
        with pytest.raises(ModelUnavailableError, match="lightgbm"):
            resolve_models(["lightgbm"], "classification")
    finally:
        trainers.is_installed = original  # type: ignore[assignment]
    assert spec.extra == "lightgbm"


def test_selecting_no_models_at_all_is_refused() -> None:
    with pytest.raises(ModelError, match="at least one Model"):
        resolve_models([], "classification")


# -- defaults, seeds, building ---------------------------------------------


def test_defaults_and_seeds_are_per_task_type() -> None:
    svm = MODEL_SPECS["svm"]
    # SVC takes a random_state (it needs one to calibrate), SVR does not.
    assert svm.seeds_for("classification") == ("random_state",)
    assert svm.seeds_for("regression") == ()
    assert "probability" not in svm.defaults_for("classification")
    assert svm.defaults_for("regression") == {"C": 1.0, "kernel": "rbf"}


def test_the_seed_reaches_the_hyperparameters_and_an_override_wins() -> None:
    spec = MODEL_SPECS["random_forest"]
    resolved = resolve_hyperparameters(spec, "classification", seed=42)
    assert resolved["random_state"] == 42
    assert resolved["n_estimators"] == 100
    overridden = resolve_hyperparameters(
        spec, "classification", seed=42, overrides={"n_estimators": 7}
    )
    assert overridden == {"n_estimators": 7, "random_state": 42}


def test_every_registered_model_can_actually_be_built_for_each_task_type() -> None:
    """Genuinely construct every estimator, so a bad default cannot ship."""
    for spec in MODEL_SPECS.values():
        if not trainers.is_available(spec):
            continue  # a heavy extra this machine does not have; tested where it is
        for task_type in spec.task_types:
            params = resolve_hyperparameters(spec, task_type, seed=1)
            estimator = trainers.build_estimator(spec, task_type, hyperparameters=params)
            assert type(estimator).__name__ == spec.path(task_type).split(":")[1]


def test_a_neural_net_is_built_with_the_task_type_it_was_asked_for() -> None:
    """A softmax head and a single linear unit are different networks."""
    for name in ("torch_mlp", "tensorflow_mlp"):
        if not trainers.is_available(MODEL_SPECS[name]):
            continue
        for task_type in ("classification", "regression"):
            spec = MODEL_SPECS[name]
            estimator = trainers.build_estimator(
                spec, task_type, hyperparameters=resolve_hyperparameters(spec, task_type, seed=1)
            )
            assert estimator.task_type == task_type
            assert estimator.wants_task_type


def test_the_task_type_is_not_a_hyperparameter_the_user_can_set() -> None:
    """It is injected by `build_estimator`, so it cannot be pinned or searched."""
    for name in ("torch_mlp", "tensorflow_mlp"):
        spec = MODEL_SPECS[name]
        assert "task_type" not in spec.defaults_for("classification")
        assert "task_type" not in spec.search_space
        assert "task_type" not in resolve_hyperparameters(spec, "classification", seed=1)


def test_library_versions_reports_what_was_used() -> None:
    versions = library_versions(["scikit-learn", "lightgbm"])
    assert set(versions) == {"python", "scikit-learn", "lightgbm"}
    assert versions["scikit-learn"] and versions["scikit-learn"][0].isdigit()
    assert library_versions()["scikit-learn"] is not None
    # A library that ships under two names resolves through either of them.
    assert set(library_versions()) >= {"torch", "tensorflow"}


# -- the split guarantee ----------------------------------------------------


def test_a_training_matrix_cannot_be_built_by_hand() -> None:
    with pytest.raises(TypeError, match="from_split"):
        TrainingMatrix(
            np.zeros((2, 2)),
            np.zeros(2, dtype="int64"),
            feature_names=["a", "b"],
            task_type="classification",
            rows=[0, 1],
            classes=["int:0"],
        )


def test_from_split_contains_the_training_rows_and_nothing_else() -> None:
    """A held-out split marks its test rows; the matrix must not carry them."""
    frame = classification_frame(30)
    setup, pipeline, matrix = training_matrix(frame)
    # Anything the held-out rows carry is far outside the training rows' range.
    marker = 1e6
    frame.loc[list(setup.split.test), "age"] = marker

    rebuilt = TrainingMatrix.from_split(
        frame=frame, spec=pipeline.to_dict(), setup=setup
    )
    assert rebuilt.rows == [int(i) for i in setup.split.train]
    assert rebuilt.n_samples == len(setup.split.train)
    assert set(rebuilt.rows).isdisjoint(set(int(i) for i in setup.split.test))
    age_column = list(pipeline.spec.roles).index("age")
    assert rebuilt.X[:, age_column].max() < marker / 10


def test_from_split_accepts_a_fitted_preprocessor_as_well_as_its_dict() -> None:
    frame = classification_frame(30)
    setup, pipeline, _ = training_matrix(frame)
    from_object = TrainingMatrix.from_split(
        frame=build_setup(TrainingSetupRequest(version_id="v1", target="region"), frame).frame,
        spec=pipeline,
        setup=setup,
    )
    assert from_object.feature_names == pipeline.feature_names


def test_assert_held_out_refuses_an_overlap() -> None:
    assert_held_out([0, 1, 2], [3, 4])  # disjoint is fine
    with pytest.raises(AssertionError, match="held-out test rows"):
        assert_held_out([0, 1, 9], [3, 9])


def test_tuning_only_accepts_a_training_matrix() -> None:
    import inspect

    signature = inspect.signature(trainers.tune_model)
    assert signature.parameters["data"].annotation in ("TrainingMatrix", TrainingMatrix)
    resolved = inspect.signature(trainers.tune_model, eval_str=True)
    assert resolved.parameters["data"].annotation is TrainingMatrix
    spec = MODEL_SPECS["knn"]
    _, _, matrix = training_matrix(classification_frame(30))
    # A raw array cannot be passed where a TrainingMatrix is required.
    with pytest.raises(TypeError):
        trainers.tune_model(spec, matrix.X, metric="f1_macro")  # type: ignore[arg-type]


def test_tuning_searches_exactly_the_training_rows() -> None:
    frame = classification_frame(40)
    setup, _, matrix = training_matrix(frame)
    result = trainers.tune_model(
        MODEL_SPECS["knn"], matrix, metric="f1_macro", strategy="grid", cv=2
    )
    assert result["skipped"] is None
    assert result["trained_on_rows"] == [int(i) for i in setup.split.train]
    assert set(result["trained_on_rows"]).isdisjoint(set(int(i) for i in setup.split.test))
    assert result["n_samples"] == len(setup.split.train)
    assert result["best_params"] and result["best_score"] is not None


def test_tuning_refuses_a_metric_from_the_other_task_type() -> None:
    frame = classification_frame(30)
    _, _, matrix = training_matrix(frame)
    with pytest.raises(ModelError, match="regression metric"):
        trainers.tune_model(MODEL_SPECS["knn"], matrix, metric="mae")  # type: ignore[arg-type]


def test_tuning_on_roc_auc_skips_a_model_with_no_probabilities() -> None:
    frame = classification_frame(40)
    _, _, matrix = training_matrix(frame)
    result = trainers.tune_model(MODEL_SPECS["svm"], matrix, metric="roc_auc", cv=2)
    assert "skipped" in result and "class probabilities" in result["skipped"]
    assert result["trained_on_rows"] == matrix.rows
    # A Model that does produce probabilities is tuned, not skipped.
    tuned = trainers.tune_model(MODEL_SPECS["knn"], matrix, metric="roc_auc", cv=2)
    assert tuned.get("skipped") is None
    assert tuned["best_score"] is not None


def test_tuning_skips_when_there_are_too_few_rows_to_cross_validate() -> None:
    frame = classification_frame(6)
    _, _, matrix = training_matrix(frame)
    result = trainers.tune_model(MODEL_SPECS["knn"], matrix, metric="f1_macro", cv=3)
    assert "too few" in result["skipped"]
    assert result["trained_on_rows"] == matrix.rows


def test_a_pinned_hyperparameter_is_not_searched() -> None:
    frame = classification_frame(40)
    _, _, matrix = training_matrix(frame)
    result = trainers.tune_model(
        MODEL_SPECS["random_forest"],
        matrix,
        metric="f1_macro",
        strategy="grid",
        cv=2,
        overrides={"n_estimators": 7},
    )
    assert "n_estimators" not in result["search_space"], "a pinned value is not searched"
    assert result["search_space"] == {"max_depth": [None, 5, 10]}
    # The winner reports the whole mapping it was built with, pinned value included.
    assert result["best_hyperparameters"]["n_estimators"] == 7
    assert result["best_hyperparameters"]["max_depth"] == result["best_params"]["max_depth"]


def test_a_model_with_nothing_left_to_search_says_so() -> None:
    frame = classification_frame(40)
    _, _, matrix = training_matrix(frame)
    with pytest.raises(ModelError, match="no hyperparameter to search"):
        trainers.tune_model(
            MODEL_SPECS["knn"],
            matrix,
            metric="f1_macro",
            overrides={"n_neighbors": 3},
        )


def test_a_user_search_space_replaces_the_built_in_one() -> None:
    frame = classification_frame(40)
    _, _, matrix = training_matrix(frame)
    result = trainers.tune_model(
        MODEL_SPECS["knn"],
        matrix,
        metric="f1_macro",
        strategy="grid",
        cv=2,
        search_space={"n_neighbors": [1, 2]},
    )
    assert result["search_space"] == {"n_neighbors": [1, 2]}
    assert result["candidates"] == 2
    with pytest.raises(ModelError, match="is empty"):
        trainers.tune_model(
            MODEL_SPECS["knn"], matrix, metric="f1_macro", search_space={"n_neighbors": []}
        )


# -- metrics ----------------------------------------------------------------


def test_classification_metrics_on_the_held_out_split() -> None:
    truth = [0, 0, 1, 1, 1]
    predicted = [0, 1, 1, 1, 0]
    proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.1, 0.9], [0.7, 0.3]])
    metrics = evaluate_classification(truth, predicted, proba=proba, classes=["a", "b"])
    assert metric_value(metrics, "accuracy") == pytest.approx(0.6)
    assert metrics["f1_macro"]["value"] is not None
    assert 0.0 <= metrics["roc_auc"]["value"] <= 1.0
    assert metrics["roc_auc"]["reason"] is None
    assert metrics["confusion_matrix"] == [[1, 1], [1, 2]]
    assert metrics["n_test"] == 5
    assert metrics["classes"] == ["a", "b"]
    assert set(metrics["support"]) == {"a", "b"}


def test_roc_auc_is_null_with_a_reason_when_there_are_no_probabilities() -> None:
    metrics = evaluate_classification([0, 1, 1], [0, 1, 0], classes=["a", "b"])
    assert metrics["roc_auc"] == {
        "value": None,
        "reason": "this Model does not output class probabilities",
    }
    assert metrics["accuracy"]["value"] is not None, "the other metrics still come through"


def test_a_single_class_test_split_nulls_the_class_dependent_metrics() -> None:
    metrics = evaluate_classification([1, 1, 1], [1, 1, 1], classes=["a", "b"])
    for name in ("roc_auc", "f1_macro", "balanced_accuracy", "f1_minority"):
        assert metrics[name]["value"] is None, name
        assert "single class" in metrics[name]["reason"], name
    assert metrics["accuracy"]["value"] == 1.0
    assert "b" in metrics["roc_auc"]["reason"]


def test_multiclass_roc_auc_is_macro_one_versus_rest() -> None:
    good = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.2, 0.1, 0.7],
        ]
    )
    truth = [0, 1, 2, 0, 1, 2]
    ranked = evaluate_classification(truth, truth, proba=good, classes=["a", "b", "c"])
    assert ranked["roc_auc"]["value"] == pytest.approx(1.0)
    assert ranked["roc_auc"]["reason"] is None
    # Point one class's probabilities at the wrong class: the macro one-vs-rest
    # average must drop off a perfect score, which a per-class number would hide.
    inverted = good.copy()
    inverted[[2, 5]] = inverted[[2, 5]][:, [0, 2, 1]]
    wrong = evaluate_classification(truth, truth, proba=inverted, classes=["a", "b", "c"])
    assert 0.5 < wrong["roc_auc"]["value"] < 1.0
    assert wrong["roc_auc"]["value"] < ranked["roc_auc"]["value"]
    assert wrong["f1_minority"]["value"] == pytest.approx(1.0)


def test_tuning_on_roc_auc_works_for_a_multiclass_target() -> None:
    """scikit-learn's own roc_auc scorer refuses a multiclass Target; ours must not."""
    frame = classification_frame(40)  # three regions
    _, _, matrix = training_matrix(frame, target="region")
    result = trainers.tune_model(
        MODEL_SPECS["knn"], matrix, metric="roc_auc", strategy="grid", cv=2
    )
    assert result["skipped"] is None
    assert result["best_score"] is not None and result["best_score"] > 0.0


def test_f1_minority_scores_the_least_frequent_class() -> None:
    truth = [0, 0, 0, 0, 1]
    metrics = evaluate_classification(truth, [0, 0, 0, 0, 0], classes=["a", "b"])
    assert metrics["f1_minority"]["value"] == 0.0
    assert metrics["f1_macro"]["value"] is not None


def test_regression_metrics_mae_rmse_and_r_squared() -> None:
    truth = [1.0, 2.0, 3.0, 4.0]
    predicted = [1.0, 2.0, 2.0, 5.0]
    metrics = evaluate_regression(truth, predicted)
    assert metrics["mae"]["value"] == pytest.approx(0.5)
    assert metrics["rmse"]["value"] == pytest.approx(np.sqrt(0.5))
    assert metrics["r2"]["value"] == pytest.approx(0.6)
    assert metrics["max_error"]["value"] == pytest.approx(1.0)
    assert metrics["n_test"] == 4
    assert metrics["residual_mean"]["value"] == pytest.approx(0.0)


def test_r_squared_needs_at_least_two_held_out_rows() -> None:
    metrics = evaluate_regression([1.0], [1.2])
    assert metrics["r2"] == {"value": None, "reason": "R² needs at least 2 held-out rows"}
    assert metrics["mae"]["value"] == pytest.approx(0.2)


# -- metrics catalogue and the primary metric ------------------------------


def test_the_default_primary_metric_per_task_type() -> None:
    assert default_primary_metric("classification") == "f1_macro"
    assert default_primary_metric("regression") == "r2"
    assert metrics_for("classification")[0] == "f1_macro"
    assert metrics_for("regression")[0] == "r2"


def test_the_leaderboard_can_be_ranked_on_any_metric_of_the_task_type() -> None:
    assert "accuracy" in metrics_for("classification")
    assert "roc_auc" in metrics_for("classification")
    assert "rmse" in metrics_for("regression")
    assert "accuracy" not in metrics_for("regression")
    for name in metrics_for("classification"):
        assert METRICS[name].task_type == "classification"


def test_the_primary_metric_is_validated_against_the_task_type() -> None:
    assert resolve_primary_metric(None, "classification") == "f1_macro"
    assert resolve_primary_metric("roc_auc", "classification") == "roc_auc"
    with pytest.raises(ModelError, match="regression metric"):
        resolve_primary_metric("mae", "classification")
    with pytest.raises(ModelError, match="unknown metric"):
        resolve_primary_metric("logloss", "classification")
    with pytest.raises(ModelError, match="task type must be one of"):
        resolve_primary_metric(None, "clustering")


# -- ranking ----------------------------------------------------------------


def entry(model: str, value, name: str = "f1_macro", status: str = "ok") -> dict:
    return {
        "model": model,
        "status": status,
        "metrics": {name: {"value": value, "reason": None if value is not None else "no"}},
    }


def test_ranking_puts_the_best_score_first_and_shares_ties() -> None:
    ranked = rank_leaderboard(
        [
            entry("a", 0.5, "f1_macro"),
            entry("b", 0.9, "f1_macro"),
            entry("c", 0.9, "f1_macro"),
        ],
        "f1_macro",
    )
    assert [row["model"] for row in ranked] == ["b", "c", "a"]
    # Competition ranking: two Models tied for first means the next is third.
    assert [row["rank"] for row in ranked] == [1, 1, 3]


def test_ranking_knows_that_lower_is_better() -> None:
    ranked = rank_leaderboard(
        [entry("a", 0.4, "mae"), entry("b", 0.1, "mae")], "mae"
    )
    assert [row["model"] for row in ranked] == ["b", "a"]
    assert [row["rank"] for row in ranked] == [1, 2]


def test_an_unmeasurable_model_ranks_last_without_a_rank() -> None:
    ranked = rank_leaderboard(
        [
            entry("a", None, "roc_auc"),
            entry("b", 0.7, "roc_auc"),
        ],
        "roc_auc",
    )
    assert [row["model"] for row in ranked] == ["b", "a"]
    assert ranked[1]["rank"] is None
    assert ranked[1]["primary"]["reason"] == "no"


def test_a_model_that_failed_to_fit_ranks_last_whatever_it_scored() -> None:
    ranked = rank_leaderboard(
        [
            {"model": "broken", "status": "failed", "error": "boom", "metrics": {"f1_macro": {"value": 0.99}}},
            entry("ok", 0.5, "f1_macro"),
        ],
        "f1_macro",
    )
    assert [row["model"] for row in ranked] == ["ok", "broken"]
    assert ranked[1]["rank"] is None
    assert ranked[1]["error"] == "boom"


def test_the_leaderboard_warnings_say_what_could_not_be_ranked() -> None:
    warnings = trainers.leaderboard_warnings(
        [entry("a", None, name="roc_auc")], "roc_auc"
    )
    assert "no Model could be scored" in warnings[0]
    mixed = trainers.leaderboard_warnings(
        [entry("a", 0.4, "f1_macro"), {"model": "b", "status": "failed", "metrics": {}}],
        "f1_macro",
    )
    assert any("failed to fit" in w for w in mixed)
