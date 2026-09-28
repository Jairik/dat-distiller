"""Training setup: Task Type, preprocessing, leakage guard, split, Checks.

numpy + pandas only — the preprocessing pipeline is hand-written so it is a
plain serializable object a later Model Bundle can reuse.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from dat_distiller.checks import CheckStore
from dat_distiller.store.paths import AppPaths
from dat_distiller.store.provenance import PROVENANCE_COLUMN
from dat_distiller.store.store import DatasetStore
from dat_distiller.training.checks import (
    IMBALANCE_MIN_FRACTION,
    LEAKAGE_MIN_SCORE,
    class_balance,
    detect_target_leakage,
    duplicate_rows,
    evaluate_training_checks,
    raise_training_checks,
)
from dat_distiller.training.preprocess import Preprocessor, fit_preprocessor
from dat_distiller.training.setup import (
    TrainingSetupRequest,
    build_setup,
    estimate_setup,
    run_setup,
)
from dat_distiller.training.split import DEFAULT_TEST_SIZE, split_indices
from dat_distiller.training.task import (
    TASK_TYPES,
    infer_task_type,
    label_family,
    sibling_columns,
)

REVIEW_THRESHOLD = 0.8


# -- fixtures ---------------------------------------------------------------


@pytest.fixture
def store(isolated_data_dir) -> DatasetStore:
    return DatasetStore(AppPaths(root=isolated_data_dir))


def labeled_frame(rows: int = 20, *, seed: int = 0) -> pd.DataFrame:
    """A Dataset Version frame with a Noul, a Choice and a Score Label Column."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "age": rng.integers(18, 80, rows),
            "spend": rng.normal(0, 1, rows).round(3),
            "region": rng.choice(["north", "south", "east"], rows),
            "bio": [f"row {i} alpha beta" for i in range(rows)],
        }
    )
    frame["is_spam"] = pd.array([bool(i % 2) for i in range(rows)], dtype="boolean")
    frame["is_spam__confidence"] = pd.array(
        [0.4 + 0.1 * (i % 4) for i in range(rows)], dtype="Float64"
    )
    frame["tone"] = pd.array(["pos" if i % 2 else "neg" for i in range(rows)], dtype="object")
    frame["tone__confidence"] = pd.array([0.9] * rows, dtype="Float64")
    frame["tone__p_pos"] = pd.array([0.9 if i % 2 else 0.1 for i in range(rows)], dtype="Float64")
    frame["tone__p_neg"] = pd.array([0.1 if i % 2 else 0.9 for i in range(rows)], dtype="Float64")
    frame["quality"] = pd.array([float(i % 3) for i in range(rows)], dtype="Float64")
    frame["quality__confidence"] = pd.array([0.9] * rows, dtype="Float64")
    frame["quality__probabilities"] = [
        json.dumps({"bad": 0.2, "ok": 0.5, "great": 0.3}) for _ in range(rows)
    ]
    frame[PROVENANCE_COLUMN] = [
        json.dumps(
            {
                "row_origin": "uploaded",
                "label_origins": {
                    "is_spam": {"origin": "jev", "confidence": 0.4 + 0.1 * (i % 4)},
                    "tone": {"origin": "jev", "confidence": 0.9},
                    "quality": {"origin": "jev", "confidence": 0.9},
                },
            }
        )
        for i in range(rows)
    ]
    return frame


@pytest.fixture
def version(store: DatasetStore) -> str:
    project = store.create_project("train lab")
    return store.create_version(project.id, labeled_frame(20), origin="uploaded").id


def seed_version(store: DatasetStore, frame: pd.DataFrame, name: str = "train lab") -> str:
    project = store.create_project(name)
    return store.create_version(project.id, frame, origin="uploaded").id


class FakeCtx:
    """JobContext double: keeps a checkpoint and reports progress."""

    def __init__(self, params: dict, job_id: str = "run-1") -> None:
        self.params = params
        self.job_id = job_id
        self.project_id = "p"
        self.checkpoint: dict | None = None
        self.progress_log: list[tuple[int, int]] = []
        self.cancel = False

    def progress(self, done: int, total: int, **extra) -> None:
        self.progress_log.append((done, total))

    @property
    def cancelled(self) -> bool:
        return self.cancel

    def save_checkpoint(self, data: dict) -> None:
        self.checkpoint = json.loads(json.dumps(data, default=str))

    def load_checkpoint(self) -> dict | None:
        return self.checkpoint


# -- Target and Task Type ---------------------------------------------------


def test_task_types_are_the_documented_pair() -> None:
    assert TASK_TYPES == ("classification", "regression")


@pytest.mark.parametrize(
    "column, expected",
    [
        ("is_spam", "classification"),  # Noul, boolean
        ("tone", "classification"),  # Choice, few categories
        ("quality", "classification"),  # Score, whole-number levels
        ("is_spam__confidence", "regression"),  # Noul confidence
        ("tone__p_pos", "regression"),  # Choice probability
        ("region", "classification"),  # plain categorical
        ("age", "regression"),  # many distinct integers
        ("spend", "regression"),  # continuous
    ],
)
def test_infer_task_type_per_column(column: str, expected: str) -> None:
    assert infer_task_type(labeled_frame(20)[column]) == expected


def test_task_type_can_be_overridden() -> None:
    frame = labeled_frame(20)
    request = TrainingSetupRequest(version_id="v", target="quality", task_type="regression")
    assert build_setup(request, frame).setup.task_type == "regression"
    request = TrainingSetupRequest(version_id="v", target="quality", task_type="classification")
    assert build_setup(request, frame).setup.task_type == "classification"


def test_task_type_must_be_one_of_the_pair() -> None:
    with pytest.raises(ValueError):
        TrainingSetupRequest(version_id="v", target="age", task_type="clustering")


def test_binary_numeric_target_reads_as_classification() -> None:
    series = pd.Series([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    assert infer_task_type(series) == "classification"


# -- leakage guard: Label Column siblings -----------------------------------


def test_label_family_splits_on_the_double_underscore() -> None:
    assert label_family("is_spam") == "is_spam"
    assert label_family("is_spam__confidence") == "is_spam"
    assert label_family("tone__p_pos") == "tone"
    assert label_family("quality__probabilities") == "quality"


def test_sibling_exclusion_noul_family() -> None:
    columns = list(labeled_frame().columns)
    assert sibling_columns("is_spam", columns) == ["is_spam__confidence"]


def test_sibling_exclusion_choice_family() -> None:
    columns = list(labeled_frame().columns)
    assert sibling_columns("tone", columns) == [
        "tone__confidence",
        "tone__p_pos",
        "tone__p_neg",
    ]


def test_sibling_exclusion_score_family() -> None:
    columns = list(labeled_frame().columns)
    assert sibling_columns("quality", columns) == [
        "quality__confidence",
        "quality__probabilities",
    ]


def test_sibling_exclusion_from_a_sibling_column_itself() -> None:
    """A confidence or probability column is a valid Target — its whole family
    still has to be excluded."""
    columns = list(labeled_frame().columns)
    assert sibling_columns("is_spam__confidence", columns) == ["is_spam"]
    assert sibling_columns("tone__p_pos", columns) == [
        "tone",
        "tone__confidence",
        "tone__p_neg",
    ]


def test_sibling_exclusion_is_prefix_scoped() -> None:
    columns = ["spam", "spam__confidence", "spammy__confidence", "spam_x__p_a"]
    assert sibling_columns("spam", columns) == ["spam__confidence"]


def test_setup_excludes_siblings_from_the_features() -> None:
    frame = labeled_frame(20)
    resolved = build_setup(TrainingSetupRequest(version_id="v", target="tone"), frame)
    setup = resolved.setup
    assert setup.excluded_columns["tone__confidence"] == "target_sibling"
    assert setup.excluded_columns["tone__p_pos"] == "target_sibling"
    assert setup.excluded_columns["tone__p_neg"] == "target_sibling"
    assert setup.excluded_columns["tone"] == "target"
    assert setup.excluded_columns[PROVENANCE_COLUMN] == "provenance"
    assert not [c for c in setup.feature_columns if c.startswith("tone")]


def test_setup_excludes_whole_family_when_a_sibling_is_the_target() -> None:
    frame = labeled_frame(20)
    setup = build_setup(
        TrainingSetupRequest(version_id="v", target="is_spam__confidence"), frame
    ).setup
    assert setup.excluded_columns["is_spam"] == "target_sibling"
    assert "is_spam" not in setup.feature_columns


def test_an_explicit_feature_list_cannot_smuggle_a_sibling_back_in() -> None:
    """The leakage guard is not opt-out: naming a sibling as a feature still drops it."""
    frame = labeled_frame(20)
    setup = build_setup(
        TrainingSetupRequest(
            version_id="v", target="is_spam", features=["age", "is_spam__confidence"]
        ),
        frame,
    ).setup
    assert setup.feature_columns == ["age"]
    assert setup.excluded_columns["is_spam__confidence"] == "target_sibling"


# -- Sensitive Attribute ----------------------------------------------------


def test_sensitive_attribute_is_excluded_by_default() -> None:
    frame = labeled_frame(20)
    request = TrainingSetupRequest(version_id="v", target="is_spam", sensitive_attribute="region")
    setup = build_setup(request, frame).setup
    assert setup.sensitive_attribute == "region"
    assert setup.excluded_columns["region"] == "sensitive_attribute"
    assert "region" not in setup.feature_columns


def test_sensitive_attribute_can_be_opted_in() -> None:
    frame = labeled_frame(20)
    request = TrainingSetupRequest(
        version_id="v", target="is_spam", sensitive_attribute="region",
        include_sensitive_attribute=True,
    )
    setup = build_setup(request, frame).setup
    assert "region" in setup.feature_columns
    assert "region" not in setup.excluded_columns


def test_sensitive_attribute_must_be_known_and_not_the_target() -> None:
    frame = labeled_frame(20)
    with pytest.raises(ValueError, match="unknown Sensitive Attribute"):
        build_setup(
            TrainingSetupRequest(version_id="v", target="is_spam", sensitive_attribute="nope"), frame
        )
    with pytest.raises(ValueError, match="cannot also be the Target"):
        build_setup(
            TrainingSetupRequest(version_id="v", target="region", sensitive_attribute="region"), frame
        )


# -- feature selection ------------------------------------------------------


def test_features_default_to_every_other_column() -> None:
    frame = labeled_frame(20)
    setup = build_setup(TrainingSetupRequest(version_id="v", target="is_spam"), frame).setup
    assert setup.feature_columns == [
        "age", "spend", "region", "bio", "tone", "tone__confidence",
        "tone__p_pos", "tone__p_neg", "quality", "quality__confidence",
        "quality__probabilities",
    ]


def test_explicit_feature_list_wins() -> None:
    frame = labeled_frame(20)
    request = TrainingSetupRequest(
        version_id="v", target="is_spam", features=["age", "spend"]
    )
    assert build_setup(request, frame).setup.feature_columns == ["age", "spend"]


def test_exclude_features_removes_from_the_inferred_set() -> None:
    frame = labeled_frame(20)
    request = TrainingSetupRequest(
        version_id="v", target="is_spam", exclude_features=["bio", "quality"]
    )
    setup = build_setup(request, frame).setup
    assert "bio" not in setup.feature_columns
    assert setup.excluded_columns["bio"] == "user_excluded"
    assert setup.excluded_columns["quality"] == "user_excluded"
    assert "quality__confidence" in setup.feature_columns  # only the exact column


def test_unknown_columns_are_rejected() -> None:
    frame = labeled_frame(20)
    with pytest.raises(ValueError, match="unknown Target"):
        build_setup(TrainingSetupRequest(version_id="v", target="nope"), frame)
    with pytest.raises(ValueError, match="unknown feature columns"):
        build_setup(
            TrainingSetupRequest(version_id="v", target="is_spam", features=["nope"]), frame
        )
    with pytest.raises(ValueError, match="unknown excluded columns"):
        build_setup(
            TrainingSetupRequest(version_id="v", target="is_spam", exclude_features=["nope"]), frame
        )
    with pytest.raises(ValueError, match="no feature columns"):
        build_setup(
            TrainingSetupRequest(version_id="v", target="is_spam", exclude_features=[c for c in frame.columns if c != "is_spam"]),
            frame,
        )


def test_setup_needs_two_usable_target_values() -> None:
    frame = labeled_frame(20)
    frame["constant"] = "same"
    with pytest.raises(ValueError, match="fewer than 2 distinct"):
        build_setup(TrainingSetupRequest(version_id="v", target="constant"), frame)
    with pytest.raises(ValueError, match="at least 2 rows"):
        build_setup(
            TrainingSetupRequest(version_id="v", target="is_spam", features=["age"]),
            frame.head(1),
        )


def test_rows_with_a_missing_target_are_dropped() -> None:
    frame = labeled_frame(20)
    frame["is_spam"] = pd.array(
        [None if i < 4 else bool(i % 2) for i in range(20)], dtype="boolean"
    )
    setup = build_setup(TrainingSetupRequest(version_id="v", target="is_spam"), frame).setup
    assert setup.dropped_null_target == 4
    assert setup.kept_rows == 16


# -- unreviewed rows --------------------------------------------------------


def unreviewed_frame(rows: int = 20) -> pd.DataFrame:
    """A quarter of the Jev labels sit below the review threshold and are unreviewed."""
    frame = labeled_frame(rows)
    provenance = []
    confidence = []
    for index in range(rows):
        low = index % 4 == 0  # unreviewed + low confidence (is_spam is False here)
        reviewed = index % 4 == 1  # low confidence, but a human looked at it
        value = 0.4 if low or reviewed else 0.95
        origin = "human_reviewed" if reviewed else "jev"
        confidence.append(value)
        provenance.append(
            json.dumps(
                {
                    "row_origin": "uploaded",
                    "label_origins": {"is_spam": {"origin": origin, "confidence": value}},
                }
            )
        )
    frame["is_spam__confidence"] = pd.array(confidence, dtype="Float64")
    frame[PROVENANCE_COLUMN] = provenance
    return frame


def test_unreviewed_low_confidence_rows_are_counted() -> None:
    resolved = build_setup(
        TrainingSetupRequest(version_id="v", target="is_spam"),
        unreviewed_frame(),
        review_threshold=REVIEW_THRESHOLD,
    )
    assert resolved.setup.unreviewed_rows == 5
    assert resolved.setup.confidence_column == "is_spam__confidence"
    assert resolved.setup.reviewed_rows == 15
    assert resolved.setup.kept_rows == 20


def test_exclude_unreviewed_drops_them_before_the_split() -> None:
    resolved = build_setup(
        TrainingSetupRequest(version_id="v", target="is_spam", exclude_unreviewed=True),
        unreviewed_frame(),
        review_threshold=REVIEW_THRESHOLD,
    )
    setup = resolved.setup
    assert setup.dropped_unreviewed == 5
    assert setup.unreviewed_rows == 0
    assert setup.kept_rows == 15
    assert len(resolved.frame) == 15
    assert len(setup.split.train) + len(setup.split.test) == 15


def test_human_reviewed_low_confidence_rows_are_not_unreviewed() -> None:
    frame = unreviewed_frame()
    frame[PROVENANCE_COLUMN] = [
        json.dumps(
            {
                "row_origin": "uploaded",
                "label_origins": {"is_spam": {"origin": "human_reviewed", "confidence": 0.4}},
            }
        )
        for _ in range(len(frame))
    ]
    setup = build_setup(
        TrainingSetupRequest(version_id="v", target="is_spam", exclude_unreviewed=True),
        frame,
        review_threshold=REVIEW_THRESHOLD,
    ).setup
    # every low-confidence label was looked at, so nothing is dropped
    assert setup.unreviewed_rows == 0
    assert setup.dropped_unreviewed == 0
    assert setup.kept_rows == 20


# -- split ------------------------------------------------------------------


def test_split_defaults_to_twenty_percent_held_out() -> None:
    split = split_indices(100, seed=0)
    assert DEFAULT_TEST_SIZE == 0.2
    assert len(split.test) == 20
    assert len(split.train) == 80
    assert not set(split.train) & set(split.test)
    assert sorted(np.concatenate([split.train, split.test])) == list(range(100))


def test_split_is_reproducible_from_the_recorded_seed() -> None:
    first = split_indices(50, seed=7)
    second = split_indices(50, seed=7)
    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.test, second.test)
    assert not np.array_equal(split_indices(50, seed=8).test, first.test)


def test_classification_split_is_stratified() -> None:
    labels = np.array(["a"] * 18 + ["b"] * 2)
    split = split_indices(20, seed=0, stratify=labels)
    assert split.stratified is True
    # the rare class still gets a test row, and keeps one for training
    assert set(labels[split.test]) == {"a", "b"}
    assert set(labels[split.train]) == {"a", "b"}
    assert len(split.test) == 5  # 20% of 'a' plus the guaranteed rare-class row
    assert len(split.train) + len(split.test) == 20


def test_tiny_frames_never_produce_an_empty_train_split() -> None:
    split = split_indices(2, seed=0)
    assert len(split.train) == 1 and len(split.test) == 1
    single = split_indices(1, seed=0)
    assert len(single.train) == 1 and len(single.test) == 0


def test_setup_records_the_seed_and_the_split() -> None:
    frame = labeled_frame(20)
    request = TrainingSetupRequest(version_id="v", target="is_spam", test_size=0.25)
    setup = build_setup(request, frame).setup
    assert setup.test_size == 0.25
    assert setup.split.stratified is True
    assert len(setup.split.test) == 4  # 25% of each of the two classes
    assert len(setup.split.train) == 16
    repeat = build_setup(request, frame).setup
    assert repeat.seed == setup.seed
    assert np.array_equal(repeat.split.test, setup.split.test)


def test_explicit_seed_wins_over_the_derived_one() -> None:
    frame = labeled_frame(20)
    base = build_setup(TrainingSetupRequest(version_id="v", target="is_spam"), frame).setup
    explicit = build_setup(
        TrainingSetupRequest(version_id="v", target="is_spam", seed=42), frame
    ).setup
    assert explicit.seed == 42
    assert base.seed != 42
    assert np.array_equal(explicit.split.test, split_indices(20, seed=42, stratify=frame["is_spam"].to_numpy()).test)


def test_test_size_must_be_a_fraction() -> None:
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            TrainingSetupRequest(version_id="v", target="age", test_size=bad)


# -- preprocessing ----------------------------------------------------------


def test_preprocessing_is_a_serializable_object() -> None:
    frame = labeled_frame(20)
    pipeline = fit_preprocessor(frame, features=["age", "spend", "region", "bio"])
    payload = pipeline.to_dict()
    assert json.loads(json.dumps(payload)) == payload  # plain JSON, no numpy
    restored = Preprocessor.from_dict(payload)
    assert restored.feature_names == pipeline.feature_names
    np.testing.assert_allclose(restored.transform(frame), pipeline.transform(frame))


def test_preprocessing_from_json_round_trip() -> None:
    frame = labeled_frame(20)
    pipeline = fit_preprocessor(frame, features=["age", "region", "bio"])
    restored = Preprocessor.from_json(pipeline.to_json())
    np.testing.assert_allclose(restored.transform(frame), pipeline.transform(frame))


def test_numeric_columns_are_imputed_and_scaled() -> None:
    frame = labeled_frame(20)
    frame["spend"] = pd.array([None if i < 3 else float(i) for i in range(20)], dtype="Float64")
    pipeline = fit_preprocessor(frame, features=["spend"])
    matrix = pipeline.transform(frame)
    assert matrix.shape == (20, 1)
    assert np.isfinite(matrix).all()  # missing values filled
    assert abs(float(np.mean(matrix))) < 1e-9  # z-scored on the training rows
    assert abs(float(np.std(matrix)) - 1.0) < 1e-9
    assert pipeline.spec.imputer.values["spend"] == 11.0  # median of [3..19]


def test_categorical_columns_are_one_hot_encoded() -> None:
    frame = labeled_frame(20)
    frame["region"] = [None if i < 2 else value for i, value in enumerate(frame["region"])]
    pipeline = fit_preprocessor(frame, features=["region"])
    assert pipeline.feature_names == [
        "region=east", "region=north", "region=south", "region=__missing__",
    ]
    matrix = pipeline.transform(frame)
    assert matrix.shape == (20, 4)
    assert np.allclose(matrix.sum(axis=1), 1.0)  # exactly one category per row
    assert matrix[0].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_boolean_columns_pass_through_as_zero_one() -> None:
    frame = labeled_frame(20)
    pipeline = fit_preprocessor(frame, features=["is_spam"])
    matrix = pipeline.transform(frame)
    assert pipeline.spec.roles["is_spam"] == "bool"
    assert set(np.unique(matrix)) <= {0.0, 1.0}
    assert matrix[0].tolist() == [0.0] and matrix[1].tolist() == [1.0]


def test_text_columns_become_tfidf_features() -> None:
    # 24 rows so the shared kind heuristic reads "bio" as free text, not categories
    frame = labeled_frame(24)
    pipeline = fit_preprocessor(frame, features=["bio"])
    matrix = pipeline.transform(frame)
    assert pipeline.spec.roles["bio"] == "text"
    assert "tfidf:bio=alpha" in pipeline.feature_names
    assert "tfidf:bio=beta" in pipeline.feature_names
    assert matrix.shape[0] == 24
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0)  # l2-normalized
    assert "alpha" in pipeline.spec.tfidf.vocabulary["bio"]
    assert len(pipeline.spec.tfidf.idf["bio"]) == len(pipeline.spec.tfidf.vocabulary["bio"])


def text_frame(rows: int = 24) -> pd.DataFrame:
    """A free-text column, long enough that the shared heuristic calls it text."""
    return labeled_frame(rows)


def test_text_vocabulary_respects_min_df_and_max_features() -> None:
    frame = text_frame()
    pipeline = fit_preprocessor(frame, features=["bio"], min_df=2)
    # "row", "alpha" and "beta" appear in every row; the row numbers once each
    assert set(pipeline.spec.tfidf.vocabulary["bio"]) == {"row", "alpha", "beta"}
    capped = fit_preprocessor(frame, features=["bio"], max_features=1)
    assert len(capped.spec.tfidf.vocabulary["bio"]) == 1


def test_big_rams_are_supported() -> None:
    frame = pd.DataFrame(
        {"bio": [f"good deal number {i}" for i in range(24)] + [f"bad deal number {i}" for i in range(4)]}
    )
    pipeline = fit_preprocessor(frame, features=["bio"], ngram_range=(1, 2))
    assert "good deal" in pipeline.spec.tfidf.vocabulary["bio"]
    assert "deal number" in pipeline.spec.tfidf.vocabulary["bio"]
    assert "good deal number" not in pipeline.spec.tfidf.vocabulary["bio"]  # a trigram
    pipeline3 = fit_preprocessor(frame, features=["bio"], ngram_range=(1, 3))
    assert "good deal number" in pipeline3.spec.tfidf.vocabulary["bio"]


def test_transform_frame_is_labelled_with_the_feature_names() -> None:
    frame = labeled_frame(20)
    pipeline = fit_preprocessor(frame, features=["region", "age"])
    transformed = pipeline.transform_frame(frame)
    assert list(transformed.columns) == pipeline.feature_names
    assert len(transformed) == len(frame)


def test_transform_rejects_a_missing_feature_column() -> None:
    frame = labeled_frame(20)
    pipeline = fit_preprocessor(frame, features=["age", "region"])
    with pytest.raises(ValueError, match="missing from the data"):
        pipeline.transform(frame.drop(columns=["region"]))


def test_fitting_rejects_an_unknown_or_empty_feature_list() -> None:
    frame = labeled_frame(20)
    with pytest.raises(ValueError, match="missing from the data"):
        fit_preprocessor(frame, features=["nope"])
    with pytest.raises(ValueError, match="no feature columns"):
        fit_preprocessor(frame, features=[])


def test_preprocessing_fits_only_on_the_training_split() -> None:
    frame = labeled_frame(20)
    train = frame.iloc[:15]
    pipeline = fit_preprocessor(train, features=["region"])
    assert "region=south" not in pipeline.spec.one_hot.categories["region"]  # unseen in train


def test_output_sources_map_derived_columns_back_to_their_column() -> None:
    frame = labeled_frame(24)
    pipeline = fit_preprocessor(frame, features=["region", "bio", "age"])
    sources = pipeline.spec.output_sources
    assert sources["region=north"] == "region"
    assert sources["tfidf:bio=alpha"] == "bio"
    assert sources["age"] == "age"


# -- checks: class imbalance ------------------------------------------------


def imbalanced_frame() -> pd.DataFrame:
    frame = labeled_frame(30)
    frame["region"] = ["north"] * 28 + ["south", "east"]
    return frame


def test_class_balance_reports_counts_and_fractions() -> None:
    balance = class_balance(["a", "a", "a", "b"])
    assert balance["counts"] == {"str:a": 3, "str:b": 1}
    assert balance["min_fraction"] == pytest.approx(0.25)
    assert balance["imbalance_ratio"] == pytest.approx(3.0)


def test_class_balance_measures_nothing_when_the_target_has_no_values() -> None:
    """No values means no minority class and no imbalance — not "balanced".

    The empty case used to report `min_fraction: 0.0` and `imbalance_ratio: 1.0`.
    A ratio of exactly 1.0 is the signature of a perfectly even Target, so the
    Check read as a clean result and would have passed a gate, when the honest
    answer is that there is nothing to measure. The rest of the codebase already
    has a convention for this — `fairness._measured` returns `value: None` with
    the reason spelled out — and this now follows it.
    """
    balance = class_balance([])
    assert balance["total"] == 0
    assert balance["classes"] == 0
    assert balance["min_fraction"] is None
    assert balance["imbalance_ratio"] is None
    assert "no values" in balance["reason"]
    # And it is still JSON-safe, since this payload is persisted in a Check.
    assert json.loads(json.dumps(balance))["min_fraction"] is None


def test_a_single_class_target_still_reports_a_real_ratio() -> None:
    """One class is perfectly balanced, and that is a measurement, not a gap."""
    balance = class_balance(["a", "a", "a"])
    assert balance["min_fraction"] == pytest.approx(1.0)
    assert balance["imbalance_ratio"] == pytest.approx(1.0)


def test_class_imbalance_check_fires_on_a_skewed_target() -> None:
    frame = imbalanced_frame()
    request = TrainingSetupRequest(version_id="v", target="region", features=["age"])
    resolved = build_setup(request, frame)
    report = evaluate_training_checks(
        setup=resolved.setup, train_frame=resolved.train_frame(), test_frame=resolved.test_frame()
    )
    check = report["class_imbalance"]
    assert check["raised"] is True
    assert check["severity"] == "warning"
    assert check["details"]["min_fraction"] < IMBALANCE_MIN_FRACTION
    assert check["details"]["imbalance_ratio"] > 5


def test_class_imbalance_check_is_quiet_on_a_balanced_target() -> None:
    frame = labeled_frame(20)
    resolved = build_setup(
        TrainingSetupRequest(version_id="v", target="is_spam", features=["age"]), frame
    )
    report = evaluate_training_checks(
        setup=resolved.setup, train_frame=resolved.train_frame(), test_frame=resolved.test_frame()
    )
    assert report["class_imbalance"]["raised"] is False


def test_class_imbalance_does_not_apply_to_regression() -> None:
    frame = labeled_frame(20)
    resolved = build_setup(
        TrainingSetupRequest(version_id="v", target="spend", features=["age"]), frame
    )
    report = evaluate_training_checks(
        setup=resolved.setup, train_frame=resolved.train_frame(), test_frame=resolved.test_frame()
    )
    assert report["class_imbalance"]["raised"] is False
    assert report["class_imbalance"]["details"]["task_type"] == "regression"


# -- checks: train/test duplicates ------------------------------------------


def duplicate_frame() -> pd.DataFrame:
    """Two identical blocks, so the split necessarily straddles each of them."""
    block_a = pd.DataFrame(
        {
            "age": [30] * 8, "spend": [1.0] * 8, "region": ["north"] * 8,
            "bio": ["same text"] * 8, "is_spam": pd.array([False] * 8, dtype="boolean"),
        }
    )
    block_b = block_a.copy()
    block_b["region"] = "south"
    return pd.concat([block_a, block_b], ignore_index=True)


def test_duplicate_rows_counts_the_overlap() -> None:
    train = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})
    test = pd.DataFrame({"a": [1, 1, 3], "b": ["x", "y", "w"]})
    found = duplicate_rows(train, test, ["a", "b"])
    assert found["count"] == 2
    assert found["fraction"] == pytest.approx(2 / 3, abs=1e-4)
    assert found["examples"], "the report names the duplicated rows"


def test_duplicate_rows_handles_missing_values() -> None:
    train = pd.DataFrame({"a": pd.array([None, 1.0], dtype="Float64")})
    test = pd.DataFrame({"a": pd.array([None, 2.0], dtype="Float64")})
    assert duplicate_rows(train, test, ["a"])["count"] == 1


def test_train_test_duplicates_check_fires() -> None:
    frame = duplicate_frame()
    resolved = build_setup(
        TrainingSetupRequest(version_id="v", target="region", features=["age", "spend", "bio"]),
        frame,
    )
    report = evaluate_training_checks(
        setup=resolved.setup, train_frame=resolved.train_frame(), test_frame=resolved.test_frame()
    )
    check = report["train_test_duplicates"]
    assert check["raised"] is True
    assert check["severity"] == "warning"
    assert check["details"]["count"] > 0


def test_train_test_duplicates_check_is_quiet_on_unique_rows() -> None:
    frame = labeled_frame(20)
    resolved = build_setup(
        TrainingSetupRequest(version_id="v", target="is_spam", features=["age", "spend", "region"]),
        frame,
    )
    report = evaluate_training_checks(
        setup=resolved.setup, train_frame=resolved.train_frame(), test_frame=resolved.test_frame()
    )
    assert report["train_test_duplicates"]["raised"] is False


# -- checks: target leakage -------------------------------------------------


def test_single_feature_scores_flag_a_perfect_predictor() -> None:
    values = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    matrix = np.column_stack([values.astype(float), [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5]])
    found = detect_target_leakage(
        matrix, values, ["leak", "noise"], task_type="classification"
    )
    assert found["suspects"][0]["feature"] == "leak"
    assert found["suspects"][0]["score"] == pytest.approx(1.0)
    assert found["suspects"][0]["metric"] == "stump_accuracy"
    assert found["count"] == 1


def test_single_feature_scores_on_regression_use_r_squared() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    matrix = np.column_stack([values, [3.0, 1.0, 4.0, 1.0]])
    found = detect_target_leakage(matrix, values, ["answer", "noise"], task_type="regression")
    assert found["suspects"][0]["metric"] == "r_squared"
    assert found["suspects"][0]["score"] == pytest.approx(1.0)
    assert found["count"] == 1


def test_leakage_screen_is_skipped_for_a_hopeless_target() -> None:
    values = np.array([f"class-{i}" for i in range(10)])
    matrix = np.column_stack([np.arange(10.0), np.arange(10.0)])
    found = detect_target_leakage(
        matrix, values, ["a", "b"], task_type="classification", max_classes=5
    )
    assert found["count"] == 0
    assert "skipped" in found


def test_target_leakage_check_fires_on_a_leaky_feature() -> None:
    frame = labeled_frame(20)
    frame["verdict"] = pd.array(list(frame["is_spam"]), dtype="boolean")
    resolved = build_setup(
        TrainingSetupRequest(version_id="v", target="is_spam", features=["age", "verdict"]),
        frame,
    )
    pipeline = fit_preprocessor(resolved.train_frame(), features=resolved.setup.feature_columns)
    report = evaluate_training_checks(
        setup=resolved.setup,
        train_frame=resolved.train_frame(),
        test_frame=resolved.test_frame(),
        matrix=pipeline.transform(resolved.train_frame()),
        values=resolved.train_frame()["is_spam"].to_numpy(),
        sources=pipeline.spec.output_sources,
        feature_names=pipeline.feature_names,
    )
    check = report["target_leakage"]
    assert check["raised"] is True
    assert check["severity"] == "warning"
    assert check["details"]["count"] == 1
    assert check["details"]["suspects"][0]["column"] == "verdict"
    assert check["details"]["threshold"] == LEAKAGE_MIN_SCORE


def test_target_leakage_check_is_quiet_without_a_leak() -> None:
    frame = labeled_frame(20)
    resolved = build_setup(
        TrainingSetupRequest(version_id="v", target="is_spam", features=["age", "spend"]), frame
    )
    pipeline = fit_preprocessor(resolved.train_frame(), features=resolved.setup.feature_columns)
    report = evaluate_training_checks(
        setup=resolved.setup,
        train_frame=resolved.train_frame(),
        test_frame=resolved.test_frame(),
        matrix=pipeline.transform(resolved.train_frame()),
        values=resolved.train_frame()["is_spam"].to_numpy(),
        sources=pipeline.spec.output_sources,
        feature_names=pipeline.feature_names,
    )
    assert report["target_leakage"]["raised"] is False


def test_target_leakage_is_not_evaluated_without_a_matrix() -> None:
    frame = labeled_frame(20)
    resolved = build_setup(
        TrainingSetupRequest(version_id="v", target="is_spam", features=["age"]), frame
    )
    report = evaluate_training_checks(
        setup=resolved.setup, train_frame=resolved.train_frame(), test_frame=resolved.test_frame()
    )
    assert report["target_leakage"]["raised"] is False
    assert report["target_leakage"]["details"].get("skipped")


# -- checks: unreviewed labels ---------------------------------------------


def test_unreviewed_labels_check_warns_when_low_confidence_rows_remain() -> None:
    resolved = build_setup(
        TrainingSetupRequest(version_id="v", target="is_spam", features=["age"]),
        unreviewed_frame(),
        review_threshold=REVIEW_THRESHOLD,
    )
    report = evaluate_training_checks(
        setup=resolved.setup, train_frame=resolved.train_frame(), test_frame=resolved.test_frame()
    )
    check = report["unreviewed_labels"]
    assert check["raised"] is True
    assert check["severity"] == "warning"
    assert check["details"]["unreviewed_rows"] == 5
    assert check["details"]["confidence_column"] == "is_spam__confidence"


def test_unreviewed_labels_check_is_info_once_they_are_excluded() -> None:
    resolved = build_setup(
        TrainingSetupRequest(
            version_id="v", target="is_spam", features=["age"], exclude_unreviewed=True
        ),
        unreviewed_frame(),
        review_threshold=REVIEW_THRESHOLD,
    )
    report = evaluate_training_checks(
        setup=resolved.setup, train_frame=resolved.train_frame(), test_frame=resolved.test_frame()
    )
    check = report["unreviewed_labels"]
    assert check["raised"] is True
    assert check["severity"] == "info"
    assert check["details"]["excluded_rows"] == 5
    assert check["details"]["unreviewed_rows"] == 0


def test_unreviewed_labels_check_is_quiet_when_everything_was_reviewed() -> None:
    frame = unreviewed_frame()
    frame[PROVENANCE_COLUMN] = [
        json.dumps(
            {"row_origin": "uploaded", "label_origins": {"is_spam": {"origin": "human_reviewed"}}}
        )
        for _ in range(len(frame))
    ]
    resolved = build_setup(
        TrainingSetupRequest(version_id="v", target="is_spam", features=["age"]), frame
    )
    report = evaluate_training_checks(
        setup=resolved.setup, train_frame=resolved.train_frame(), test_frame=resolved.test_frame()
    )
    assert report["unreviewed_labels"]["raised"] is False


# -- checks: sibling exclusion (info) ---------------------------------------


def test_sibling_exclusion_check_is_info_and_lists_the_columns() -> None:
    frame = labeled_frame(20)
    resolved = build_setup(
        TrainingSetupRequest(version_id="v", target="tone", features=["age"]), frame
    )
    report = evaluate_training_checks(
        setup=resolved.setup, train_frame=resolved.train_frame(), test_frame=resolved.test_frame()
    )
    check = report["label_sibling_exclusion"]
    assert check["raised"] is True
    assert check["severity"] == "info"
    assert check["details"]["excluded"] == ["tone__confidence", "tone__p_neg", "tone__p_pos"]
    assert check["details"]["family"] == "tone"


def test_sibling_exclusion_check_is_quiet_for_a_plain_target() -> None:
    frame = labeled_frame(20)
    resolved = build_setup(
        TrainingSetupRequest(version_id="v", target="spend", features=["age"]), frame
    )
    report = evaluate_training_checks(
        setup=resolved.setup, train_frame=resolved.train_frame(), test_frame=resolved.test_frame()
    )
    assert report["label_sibling_exclusion"]["raised"] is False


# -- registration of the checks --------------------------------------------


def test_raise_training_checks_registers_against_the_training_run(tmp_path) -> None:
    from dat_distiller.store.db import Database

    report = {
        "class_imbalance": {
            "raised": True, "severity": "warning", "message": "skewed",
            "details": {"min_fraction": 0.02},
        },
        "train_test_duplicates": {"raised": False, "severity": None, "message": "", "details": {}},
    }
    store = CheckStore(Database(tmp_path / "checks.db"))
    raised = raise_training_checks(store, "run-9", report)
    assert raised == ["class_imbalance"]
    checks = store.list_for_subject("training_run", "run-9")
    assert checks[0].subject_type == "training_run"
    assert checks[0].subject_id == "run-9"
    assert checks[0].severity == "warning"
    assert store.has_check("class_imbalance", "training_run", "run-9")
    # a re-run does not duplicate the Check
    assert raise_training_checks(store, "run-9", report) == []
    assert len(store.list_for_subject("training_run", "run-9")) == 1


# -- estimate ---------------------------------------------------------------


def test_estimate_reports_the_plan_without_fitting(store: DatasetStore, version: str) -> None:
    request = TrainingSetupRequest(
        version_id=version, target="is_spam", sensitive_attribute="region"
    )
    estimated = estimate_setup(request, store.load_dataframe(version, include_provenance=True))
    assert estimated["target"] == "is_spam"
    assert estimated["task_type"] == "classification"
    assert estimated["rows"] == 20
    assert estimated["train_rows"] == 16
    assert estimated["test_rows"] == 4
    assert estimated["test_size"] == 0.2
    assert estimated["stratified"] is True
    assert isinstance(estimated["seed"], int)
    assert estimated["class_counts"] == {"bool:False": 10, "bool:True": 10}
    assert estimated["n_source_features"] == len(estimated["feature_columns"])
    assert estimated["sensitive_attribute"] == "region"
    assert estimated["excluded_columns"]["region"] == "sensitive_attribute"


# -- the run ----------------------------------------------------------------


def test_run_setup_returns_a_json_serializable_plan(store: DatasetStore, version: str) -> None:
    checks = CheckStore(store.db)
    ctx = FakeCtx({"version_id": version, "target": "is_spam", "sensitive_attribute": "region"})
    result = run_setup(ctx, store=store, checks=checks, review_threshold=REVIEW_THRESHOLD)
    assert json.loads(json.dumps(result)) == result  # the job row is JSON
    assert result["training_run_id"] == "run-1"
    assert result["target"] == "is_spam"
    assert result["task_type"] == "classification"
    assert result["train_rows"] == 16 and result["test_rows"] == 4
    assert result["sensitive_attribute"] == "region"
    assert "region" not in result["feature_columns"]
    assert result["n_features"] == len(result["feature_names"])
    assert result["preprocessing"]["features"] == result["feature_columns"]
    assert "is_spam__confidence" not in result["preprocessing"]["features"]
    assert ctx.progress_log, "the job reports progress"


def test_run_setup_checkpoints_the_pipeline_for_the_model_bundle(
    store: DatasetStore, version: str
) -> None:
    checks = CheckStore(store.db)
    ctx = FakeCtx({"version_id": version, "target": "is_spam"})
    result = run_setup(ctx, store=store, checks=checks, review_threshold=REVIEW_THRESHOLD)
    checkpoint = ctx.checkpoint
    assert checkpoint["training_run_id"] == "run-1"
    pipeline = Preprocessor.from_dict(checkpoint["preprocessing"])
    frame = store.load_dataframe(version)
    assert pipeline.transform(frame).shape == (20, result["n_features"])
    assert checkpoint["setup"]["target"] == "is_spam"
    assert len(checkpoint["train_indices"]) == 16
    assert len(checkpoint["test_indices"]) == 4
    assert not set(checkpoint["train_indices"]) & set(checkpoint["test_indices"])


def test_run_setup_registers_checks_on_the_training_run(store: DatasetStore) -> None:
    frame = imbalanced_frame()
    leaky = frame.copy()
    leaky["verdict"] = pd.array(list(leaky["region"].map({"north": False, "south": True, "east": True})), dtype="boolean")
    leaky_version = seed_version(store, leaky)
    checks = CheckStore(store.db)
    ctx = FakeCtx(
        {
            "version_id": leaky_version,
            "target": "region",
            "features": ["age", "verdict"],
        },
        job_id="run-checks",
    )
    result = run_setup(ctx, store=store, checks=checks, review_threshold=REVIEW_THRESHOLD)
    registered = {c.kind for c in checks.list_for_subject("training_run", "run-checks")}
    assert "class_imbalance" in registered
    assert "target_leakage" in registered
    assert set(result["checks_raised"]) == registered


def test_run_setup_cancels_between_stages(store: DatasetStore, version: str) -> None:
    checks = CheckStore(store.db)
    ctx = FakeCtx({"version_id": version, "target": "is_spam"})
    ctx.cancel = True
    with pytest.raises(RuntimeError, match="cancelled"):
        run_setup(ctx, store=store, checks=checks, review_threshold=REVIEW_THRESHOLD)


def test_run_setup_is_reproducible(store: DatasetStore, version: str) -> None:
    checks = CheckStore(store.db)
    params = {"version_id": version, "target": "is_spam", "seed": 11}
    first = run_setup(FakeCtx(params, "a"), store=store, checks=checks, review_threshold=0.8)
    second = run_setup(FakeCtx(params, "b"), store=store, checks=checks, review_threshold=0.8)
    assert first["feature_names"] == second["feature_names"]
    assert first["preprocessing"] == second["preprocessing"]
    assert first["train_rows"] == second["train_rows"]


def test_estimator_can_train_on_the_produced_matrix(store: DatasetStore, version: str) -> None:
    """The end goal: a plain numpy matrix plus labels, ready for a Model."""
    checks = CheckStore(store.db)
    ctx = FakeCtx({"version_id": version, "target": "is_spam"})
    result = run_setup(ctx, store=store, checks=checks, review_threshold=REVIEW_THRESHOLD)
    pipeline = Preprocessor.from_dict(result["preprocessing"])
    frame = store.load_dataframe(version)
    train = frame.iloc[ctx.checkpoint["train_indices"]]
    matrix = pipeline.transform(train)
    assert matrix.ndim == 2
    assert matrix.shape == (result["train_rows"], result["n_features"])
    assert np.isfinite(matrix).all()
    assert train[result["target"]].notna().all()


def test_setup_round_trips_through_its_dict() -> None:
    from dat_distiller.training.setup import TrainingSetup

    frame = labeled_frame(20)
    setup = build_setup(
        TrainingSetupRequest(version_id="v", target="tone", sensitive_attribute="region"), frame
    ).setup
    restored = TrainingSetup.from_dict(setup.to_dict())
    assert restored.to_dict() == setup.to_dict()
    assert restored.feature_columns == setup.feature_columns
    assert math.isclose(restored.seed, setup.seed)


# -- a missing timestamp stays missing ---------------------------------------


def dated_frame(rows: int = 40, missing: tuple[int, ...] = (5, 17, 29)) -> pd.DataFrame:
    """A datetime column with real spread and a few gaps in it."""
    rng = np.random.default_rng(4)
    base = pd.Timestamp("2021-01-01")
    days = rng.integers(0, 900, rows)
    when = [base + pd.Timedelta(days=int(day)) for day in days]
    for position in missing:
        when[position] = pd.NaT
    return pd.DataFrame({"when": pd.Series(when, dtype="datetime64[ns]")}), days


def test_a_missing_timestamp_does_not_enter_the_fitted_statistics() -> None:
    """`NaT` is a gap, not the number -9.2e18.

    Casting a datetime64 column to int64 maps `NaT` to the int64 minimum, and
    that number is finite — so it survives every isfinite guard and lands in the
    median, the mean and the scale. One missing timestamp then inflates the
    scale far enough to squash all the real dates into a sliver, and the Model
    sees a feature with almost no signal in it.

    The pipeline deliberately fits its statistics on the median-filled column —
    the values the Model will actually be handed — so that is what the mean and
    the scale are compared against here. What matters is that the sentinel is
    not in it.
    """
    frame, _days = dated_frame()
    pipeline = Preprocessor.fit(frame, features=["when"])

    real = frame["when"].dropna().astype("int64").to_numpy(dtype="float64") / 1e9
    fill = float(np.median(real))
    filled = np.where(frame["when"].isna().to_numpy(), fill, real_all(frame))

    assert pipeline.spec.imputer.values["when"] == pytest.approx(fill)
    assert pipeline.spec.scaler.mean["when"] == pytest.approx(float(filled.mean()))
    assert pipeline.spec.scaler.scale["when"] == pytest.approx(float(filled.std()))

    # Which is the point: the real dates still use the range they have, spread
    # over the scale that was fitted. With the sentinel in the column the scale
    # came out ~45x too large and this span collapsed to a fraction of a unit.
    transformed = pipeline.transform(frame)[:, 0]
    seen = np.delete(transformed, [5, 17, 29])
    span_in_units = (real.max() - real.min()) / filled.std()
    assert seen.max() - seen.min() == pytest.approx(span_in_units, rel=1e-6)
    assert span_in_units > 1.0, "a real date range must not collapse into a sliver"
    # A gap lands on the imputed value, not off at the far end of the scale.
    expected_gap = (fill - filled.mean()) / filled.std()
    for position in (5, 17, 29):
        assert transformed[position] == pytest.approx(expected_gap)


def real_all(frame: pd.DataFrame) -> np.ndarray:
    """Epoch seconds for every row, with the gaps left as they are."""
    converted = pd.to_datetime(frame["when"], errors="coerce", utc=True)
    out = converted.astype("int64").to_numpy(dtype="float64") / 1e9
    out[converted.isna().to_numpy()] = np.nan
    return out


def test_a_datetime_column_that_is_entirely_missing_is_still_handled() -> None:
    """All gaps is a degenerate column, and it must not produce a sentinel.

    There is no median to take here, so the pipeline falls back to 0 — which for
    a datetime column means the epoch. That is a real instant, and it keeps the
    matrix finite. The sentinel would not: it is a negative number that no
    timestamp can ever be, and it would be persisted as this column's imputed
    value.
    """
    frame = pd.DataFrame({"when": pd.Series([pd.NaT] * 6, dtype="datetime64[ns]")})
    pipeline = Preprocessor.fit(frame, features=["when"])
    assert np.isfinite(pipeline.transform(frame)).all()
    assert pipeline.spec.scaler.scale["when"] > 0
    # Epoch seconds are positive; the int64 minimum is not.
    assert pipeline.spec.imputer.values["when"] >= 0


def test_the_persisted_pipeline_on_a_run_is_the_corrected_one(client) -> None:
    """What the run records — and the bundle ships — is fitted from real values.

    The corrupted mean and scale were not confined to one call: they were
    persisted on the Training Run and written into the Model Bundle, so a
    downstream consumer inherited them. This pins the persisted spec.
    """
    import time as _time

    frame, _days = dated_frame()
    frame["kind"] = ["a", "b"] * (len(frame) // 2)
    frame[PROVENANCE_COLUMN] = [json.dumps({"row_origin": "uploaded"}) for _ in range(len(frame))]

    project = client.post("/api/projects", json={"name": "dates"}).json()["id"]
    store: DatasetStore = client.app.state.store
    version = store.create_version(project, frame, origin="uploaded").id
    start = client.post(
        "/api/train/run",
        json={"version_id": version, "target": "kind", "models": ["logistic_regression"], "seed": 2},
    )
    assert start.status_code == 202, start.text
    deadline = _time.time() + 240
    job: dict = {}
    while _time.time() < deadline:
        job = client.get(f"/api/jobs/{start.json()['id']}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            break
        _time.sleep(0.02)
    assert job["status"] == "completed", job.get("error")

    spec = job["result"]["preprocessing"]
    # The run fits on its own training split, so this compares against a bound
    # rather than re-deriving the split. The sentinel put the scale ~45x out and
    # the mean a decade out; neither can hide inside these bounds.
    real = frame["when"].dropna().astype("int64").to_numpy(dtype="float64") / 1e9
    assert spec["scaler"]["scale"]["when"] == pytest.approx(float(real.std()), rel=0.5)
    assert spec["scaler"]["mean"]["when"] == pytest.approx(float(real.mean()), rel=0.01)
    assert spec["imputer"]["values"]["when"] == pytest.approx(float(np.median(real)), rel=0.01)
