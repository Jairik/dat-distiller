"""The Fairness Report: per-group metrics, the gaps, and the Checks they earn.

Most of this file is a **hand-built fixture**: the group column, the held-out
truth and the predictions are written out by hand, so the expected accuracy, TPR,
FPR, selection rate and MAE are arithmetic anyone can check without a Model, and
nothing here needs an optional extra. The arithmetic is deliberately unfriendly —
skewed selection rates, a group with no positive case, a group with no negative
case, a group that never reached the test split — because those are the cases
where a fairness report quietly invents a number.

The Tests that refit a real Model live in ``test_fairness_run.py`` behind
``importorskip("sklearn")``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from dat_distiller.training.checks import value_key
from dat_distiller.training.fairness import (
    CLASSIFICATION_GAPS,
    DEFAULT_GAP_THRESHOLD,
    REGRESSION_GAPS,
    FairnessError,
    ReportSource,
    build_report,
    classification_groups,
    comparable_gaps,
    exceeding_gaps,
    group_keys,
    group_sizes,
    measured_gaps,
    metric_gap,
    positive_class_code,
    regression_groups,
    resolve_gap_threshold,
)

CLASSES = ["bool:False", "bool:True"]


def source(**overrides) -> ReportSource:
    base = {
        "training_run_id": "run-1",
        "version_id": "v1",
        "target": "churned",
        "task_type": "classification",
        "model": "logistic_regression",
        "model_label": "Logistic regression",
        "sensitive_attribute": "region",
        "threshold": 0.1,
        "threshold_source": "default",
        "train_rows": 8,
        "test_rows": 8,
    }
    return ReportSource(**{**base, **overrides})


def group(name: str) -> str:
    return value_key(name)


# -- the hand-built classification fixture ------------------------------------
#
# Two groups, eight held-out rows, four each. Group 'north' has 3 positives and 1
# negative; the Model catches 2 of the 3 positives (TPR 2/3) and flags 0 of the 1
# negative (FPR 0/1), for an accuracy of 3/4. Group 'south' is the mirror: 1
# positive and 3 negatives, catching 0 of its positives (TPR 0) and flagging 1 of
# its 3 negatives (FPR 1/3), for an accuracy of 2/4.
#
#     group   truth  pred
#     north   1      1        hit
#     north   1      1        hit
#     north   1      0        miss
#     north   0      0        correct
#     south   1      0        miss
#     south   0      1        false positive
#     south   0      0        correct
#     south   0      0        correct
NORTH = [group("north")] * 4
SOUTH = [group("south")] * 4
TRUTH = [1, 1, 1, 0, 1, 0, 0, 0]
PRED = [1, 1, 0, 0, 0, 1, 0, 0]
KEPT = {group("north"): 6, group("south"): 6}


def test_per_group_classification_metrics_are_the_arithmetic_by_hand() -> None:
    groups = classification_groups(
        group_of_row=NORTH + SOUTH,
        y_true=TRUTH,
        y_pred=PRED,
        positive=1,
        kept_counts=KEPT,
        version_counts=KEPT,
        classes=CLASSES,
    )
    assert [entry["group"] for entry in groups] == ["str:north", "str:south"]
    north, south = groups

    assert north["n_test"] == 4 and north["n_total"] == 6 and north["n_train"] == 2
    assert north["n_actual_positive"] == 3 and north["n_actual_negative"] == 1
    assert north["n_predicted_positive"] == 2
    assert north["measured"] is True and north["reason"] is None
    assert north["metrics"]["accuracy"]["value"] == 0.75
    assert north["metrics"]["tpr"]["value"] == pytest.approx(2 / 3)
    assert north["metrics"]["fpr"]["value"] == 0.0
    # Selection rate is "of the group's rows", not "of the positives": 2 of 4.
    assert north["metrics"]["selection_rate"]["value"] == 0.5
    assert north["class_counts"] == {"bool:True": 3, "bool:False": 1}

    assert south["n_actual_positive"] == 1 and south["n_actual_negative"] == 3
    assert south["n_predicted_positive"] == 1
    assert south["metrics"]["accuracy"]["value"] == 0.5
    assert south["metrics"]["tpr"]["value"] == 0.0
    assert south["metrics"]["fpr"]["value"] == pytest.approx(1 / 3)
    assert south["metrics"]["selection_rate"]["value"] == 0.25


def test_per_group_regression_metrics_are_the_arithmetic_by_hand() -> None:
    # north: truth 1, 3, 5, 7 against predictions 2, 2, 8, 8 -> residuals
    # -1, 1, -3, -1, so MAE 1.5 and a mean error of -1. south: truth 1, 1, 1, 1
    # against 1, 1, 1, 5 -> residual 0, 0, 0, -4, so MAE 1.0, mean error -1.
    groups = regression_groups(
        group_of_row=NORTH + SOUTH,
        y_true=[1, 3, 5, 7, 1, 1, 1, 1],
        y_pred=[2, 2, 8, 8, 1, 1, 1, 5],
        kept_counts=KEPT,
        version_counts=KEPT,
    )
    north, south = groups
    assert north["metrics"]["mae"]["value"] == 1.5
    assert north["metrics"]["mean_error"]["value"] == -1.0
    assert north["metrics"]["rmse"]["value"] == pytest.approx(math.sqrt((1 + 1 + 9 + 1) / 4))
    assert south["metrics"]["mae"]["value"] == 1.0
    assert south["metrics"]["mean_error"]["value"] == -1.0


def test_the_signed_error_separates_opposite_biases_with_the_same_mae() -> None:
    """Both groups miss by 1.0 on average, but one is always over-predicted and
    the other always under-predicted. MAE cannot tell them apart; the mean error
    can, so the report carries both."""
    groups = regression_groups(
        group_of_row=[group("north")] * 3 + [group("south")] * 3,
        y_true=[1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        y_pred=[2.0, 3.0, 4.0, 0.0, 1.0, 2.0],
        kept_counts={group("north"): 3, group("south"): 3},
        version_counts={group("north"): 3, group("south"): 3},
    )
    north, south = groups
    assert north["metrics"]["mae"]["value"] == south["metrics"]["mae"]["value"] == 1.0
    assert north["metrics"]["mean_error"]["value"] == -1.0  # always over-predicts
    assert south["metrics"]["mean_error"]["value"] == 1.0  # always under-predicts


# -- a group that cannot be measured must not produce a number ----------------


def test_a_group_with_no_held_out_rows_is_null_with_a_reason() -> None:
    """The classic fairness-report bug: a group that exists but is not in the
    test split silently getting 0.0, or being dropped, or 50/50-ing the rest."""
    groups = classification_groups(
        group_of_row=NORTH,  # 'east' never reached the test split
        y_true=TRUTH[:4],
        y_pred=PRED[:4],
        positive=1,
        kept_counts={**KEPT, group("east"): 5},
        version_counts={**KEPT, group("east"): 5},
        classes=CLASSES,
    )
    east = next(entry for entry in groups if entry["group"] == group("east"))
    assert east["n_test"] == 0
    assert east["n_total"] == 5, "the group is reported as existing in the data"
    assert east["n_train"] == 5
    assert east["measured"] is False
    assert "no rows of group" in east["reason"]
    assert "held-out test split" in east["reason"]
    for name in ("accuracy", "tpr", "fpr", "selection_rate"):
        assert east["metrics"][name]["value"] is None
        assert east["metrics"][name]["reason"] == east["reason"]


def test_a_group_dropped_before_the_split_is_still_listed() -> None:
    """'east' exists in the Dataset Version but every row was dropped before the
    split, so it has no rows at all in the Training Run's data."""
    groups = classification_groups(
        group_of_row=NORTH + SOUTH,
        y_true=TRUTH,
        y_pred=PRED,
        positive=1,
        kept_counts=KEPT,  # no 'east' here
        version_counts={**KEPT, group("east"): 9},
        classes=CLASSES,
    )
    east = next(entry for entry in groups if entry["group"] == group("east"))
    assert east["n_version"] == 9 and east["n_total"] == 0
    assert east["metrics"]["accuracy"]["value"] is None
    assert "dropped before the split" in east["reason"]


def test_tpr_is_null_without_a_positive_case_and_fpr_without_a_negative() -> None:
    groups = classification_groups(
        group_of_row=[group("a")] * 3 + [group("b")] * 3,
        y_true=[1, 1, 1, 0, 0, 0],
        y_pred=[1, 1, 0, 1, 0, 0],
        positive=1,
        kept_counts={group("a"): 3, group("b"): 3},
        version_counts={group("a"): 3, group("b"): 3},
        classes=CLASSES,
    )
    a, b = groups
    assert a["metrics"]["tpr"]["value"] == pytest.approx(2 / 3)
    assert a["metrics"]["fpr"]["value"] is None
    assert "no negative case" in a["metrics"]["fpr"]["reason"]
    assert a["measured"] is False
    assert a["reason"] and a["reason"].startswith("fpr could not be measured")
    # accuracy and selection rate are still real numbers for a one-sided group.
    assert a["metrics"]["accuracy"]["value"] == pytest.approx(2 / 3)
    assert a["metrics"]["selection_rate"]["value"] == pytest.approx(2 / 3)

    assert b["metrics"]["fpr"]["value"] == pytest.approx(1 / 3)
    assert b["metrics"]["tpr"]["value"] is None
    assert "no positive case" in b["metrics"]["tpr"]["reason"]


def test_a_group_with_no_metric_value_is_left_out_of_the_gap_and_says_so() -> None:
    """A TPR gap can only use the groups that actually hold a positive case."""
    groups = classification_groups(
        # 'a' and 'b' hold positives; 'c' holds none, so its TPR is null.
        group_of_row=[group("a")] * 3 + [group("b")] * 3 + [group("c")] * 2,
        y_true=[1, 1, 1, 1, 0, 0, 0, 0],
        y_pred=[1, 0, 0, 1, 1, 0, 1, 0],
        positive=1,
        kept_counts={group("a"): 3, group("b"): 3, group("c"): 2},
        version_counts={group("a"): 3, group("b"): 3, group("c"): 2},
        classes=CLASSES,
    )
    tpr_gap = metric_gap(CLASSIFICATION_GAPS[1], groups, threshold=0.1)
    assert tpr_gap["metric"] == "tpr"
    assert tpr_gap["groups_total"] == 3
    assert tpr_gap["groups_compared"] == 2
    assert tpr_gap["groups_unmeasured"] == [group("c")]
    # a catches 1 of its 3 positives, b catches its only 1 -> 1 - 1/3.
    assert tpr_gap["value"] == pytest.approx(1 - 1 / 3)
    assert tpr_gap["highest_group"] == group("b")
    assert tpr_gap["lowest_group"] == group("a")
    # The selection-rate gap is computable for all three, so it compares them all.
    parity_gap = metric_gap(CLASSIFICATION_GAPS[0], groups, threshold=0.1)
    assert parity_gap["groups_compared"] == 3
    assert parity_gap["groups_unmeasured"] == []


def test_a_gap_with_a_single_usable_group_is_null_with_a_reason() -> None:
    groups = classification_groups(
        group_of_row=[group("a")] * 3 + [group("b")] * 3,
        y_true=[1, 1, 1, 0, 0, 0],
        y_pred=[1, 1, 0, 1, 0, 0],
        positive=1,
        kept_counts={group("a"): 3, group("b"): 3},
        version_counts={group("a"): 3, group("b"): 3},
        classes=CLASSES,
    )
    gap = metric_gap(CLASSIFICATION_GAPS[1], groups, threshold=0.1)
    assert gap["groups_compared"] == 1
    assert gap["groups_unmeasured"] == [group("b")]
    assert gap["value"] is None
    assert "single group" in gap["reason"]


def test_a_single_group_dataset_version_reports_a_null_gap_not_a_zero() -> None:
    groups = classification_groups(
        group_of_row=NORTH,
        y_true=TRUTH[:4],
        y_pred=PRED[:4],
        positive=1,
        kept_counts={group("north"): 4},
        version_counts={group("north"): 4},
        classes=CLASSES,
    )
    for spec in CLASSIFICATION_GAPS:
        gap = metric_gap(spec, groups, threshold=0.1)
        assert gap["value"] is None, spec.name
        assert "single group" in gap["reason"], spec.name
        assert gap["exceeds"] is None, spec.name


def test_missing_sensitive_attribute_values_are_a_reported_group() -> None:
    groups = classification_groups(
        group_of_row=[value_key(None), value_key(None), group("a")],
        y_true=[1, 0, 1],
        y_pred=[1, 0, 1],
        positive=1,
        kept_counts={"missing": 2, group("a"): 1},
        version_counts={"missing": 2, group("a"): 1},
        classes=CLASSES,
    )
    # Keys are sorted, and 'missing' sorts before a 'str:' key.
    assert [entry["group"] for entry in groups] == ["missing", group("a")]
    assert next(e for e in groups if e["group"] == "missing")["n_test"] == 2


def test_mismatched_lengths_are_refused_rather_than_zip_silently() -> None:
    with pytest.raises(FairnessError, match="one entry per held-out row"):
        classification_groups(
            group_of_row=NORTH,
            y_true=TRUTH,
            y_pred=PRED,
            positive=1,
            kept_counts=KEPT,
            classes=CLASSES,
        )
    with pytest.raises(FairnessError, match="one entry per held-out row"):
        regression_groups(
            group_of_row=NORTH, y_true=[1.0, 2.0], y_pred=[1.0], kept_counts=KEPT
        )


# -- the gaps ----------------------------------------------------------------


def build_fixture_report(**overrides) -> dict:
    return build_report(
        source(**overrides),
        group_of_row=NORTH + SOUTH,
        y_true=TRUTH,
        y_pred=PRED,
        kept_counts=KEPT,
        version_counts=KEPT,
        classes=CLASSES,
    )


def test_the_demographic_parity_difference_is_the_selection_rate_gap() -> None:
    report = build_fixture_report()
    parity = next(g for g in report["gaps"] if g["name"] == "demographic_parity")
    # north selects 0.5, south selects 0.25 -> 0.25.
    assert parity["value"] == 0.25
    assert parity["highest_group"] == group("north")
    assert parity["lowest_group"] == group("south")
    assert parity["highest_value"] == 0.5 and parity["lowest_value"] == 0.25
    assert parity["unit"] == "rate"
    assert parity["comparable"] is True
    assert parity["exceeds"] is True, "0.25 is over the 0.1 default"
    assert parity["groups_compared"] == 2


def test_the_tpr_and_fpr_gaps_are_computed_from_the_same_groups() -> None:
    report = build_fixture_report()
    by_name = {gap["name"]: gap for gap in report["gaps"]}
    assert by_name["tpr"]["value"] == pytest.approx(2 / 3)
    assert by_name["tpr"]["highest_group"] == group("north")
    assert by_name["tpr"]["lowest_group"] == group("south")
    assert by_name["fpr"]["value"] == pytest.approx(1 / 3)
    assert by_name["fpr"]["highest_group"] == group("south")
    assert by_name["fpr"]["lowest_group"] == group("north")
    assert by_name["accuracy"]["value"] == 0.25
    assert {gap["name"] for gap in report["gaps"]} == {
        "demographic_parity",
        "tpr",
        "fpr",
        "accuracy",
    }


def test_a_gap_exactly_at_the_threshold_does_not_exceed_it() -> None:
    report = build_fixture_report(threshold=0.25)
    parity = next(g for g in report["gaps"] if g["name"] == "demographic_parity")
    assert parity["value"] == 0.25
    assert parity["exceeds"] is False, "the threshold is exclusive: 'above' means >"
    assert parity["threshold"] == 0.25


def test_the_report_names_the_widest_gap_and_orders_the_exceeding_ones() -> None:
    report = build_fixture_report()
    # Widest first: tpr 0.667, fpr 0.333, then accuracy and demographic parity
    # tied at 0.25 (ties broken by name so the order never wobbles).
    assert report["gaps_exceeding_threshold"] == [
        "tpr",
        "fpr",
        "accuracy",
        "demographic_parity",
    ]
    assert report["largest_gap"]["name"] == "tpr"
    assert report["largest_gap"]["value"] == pytest.approx(2 / 3)
    assert report["largest_comparable_gap"]["name"] == "tpr"
    assert [g["name"] for g in comparable_gaps(report["gaps"])] == report["gaps_exceeding_threshold"]
    assert exceeding_gaps(report["gaps"])[0]["name"] == "tpr"


def test_a_perfect_model_with_unequal_base_rates_still_has_a_parity_gap() -> None:
    """Worth stating outright: a Model that is right every time is still unequal
    when the groups' base rates differ, and the report must not hide that behind
    a comfortable accuracy."""
    report = build_report(
        source(),
        group_of_row=NORTH + SOUTH,
        y_true=TRUTH,
        y_pred=TRUTH,  # a perfect Model
        kept_counts=KEPT,
        version_counts=KEPT,
        classes=CLASSES,
    )
    assert report["overall"]["accuracy"]["value"] == 1.0
    assert report["gaps_exceeding_threshold"] == ["demographic_parity"]
    parity = next(g for g in report["gaps"] if g["name"] == "demographic_parity")
    assert parity["value"] == 0.5  # north selects 3 of its 4 rows, south 1 of its 4
    for name in ("tpr", "fpr", "accuracy"):
        gap = next(g for g in report["gaps"] if g["name"] == name)
        assert gap["value"] == 0.0, name
        assert gap["exceeds"] is False, name


def test_a_model_that_treats_every_group_the_same_produces_no_exceeding_gap() -> None:
    """A genuinely fair fixture: two groups with the same base rate and the same
    predictions on both. Every rate is identical, so every gap is 0."""
    groups = [group("north")] * 4 + [group("south")] * 4
    report = build_report(
        source(),
        group_of_row=groups,
        y_true=[1, 1, 0, 0, 1, 1, 0, 0],
        y_pred=[1, 0, 1, 0, 1, 0, 1, 0],
        kept_counts={group("north"): 4, group("south"): 4},
        version_counts={group("north"): 4, group("south"): 4},
        classes=CLASSES,
    )
    assert report["gaps_exceeding_threshold"] == []
    assert all(gap["exceeds"] is False for gap in report["gaps"])
    assert report["largest_gap"]["value"] == 0.0
    assert set(report["gaps_within_threshold"]) == {
        "demographic_parity",
        "tpr",
        "fpr",
        "accuracy",
    }


# -- regression: the gap in the Target's units is not a rate ------------------


def test_the_absolute_mae_gap_is_reported_but_not_compared() -> None:
    report = build_report(
        source(task_type="regression", target="spend"),
        group_of_row=NORTH + SOUTH,
        y_true=[1, 3, 5, 7, 1, 1, 1, 1],
        y_pred=[2, 2, 8, 8, 1, 1, 1, 5],
        kept_counts=KEPT,
        version_counts=KEPT,
    )
    by_name = {gap["name"]: gap for gap in report["gaps"]}
    absolute = by_name["mae"]
    # 1.5 against 1.0, in the Target's own units.
    assert absolute["value"] == 0.5
    assert absolute["unit"] == "target_units"
    assert absolute["comparable"] is False
    assert absolute["exceeds"] is None, "a 0.5-unit gap is not a 10% rate gap"
    assert "own units" in absolute["reason"]
    assert "mae_relative" in absolute["reason"]

    relative = by_name["mae_relative"]
    # overall MAE is (1+1+3+1+0+0+0+4)/8 = 1.25, so 0.5/1.25 = 0.4.
    assert relative["overall_mae"] == 1.25
    assert relative["value"] == pytest.approx(0.4)
    assert relative["comparable"] is True
    assert relative["exceeds"] is True
    assert report["gaps_exceeding_threshold"] == ["mae_relative"]
    assert report["largest_gap"]["name"] == "mae"
    assert report["largest_comparable_gap"]["name"] == "mae_relative"


def test_a_perfect_regression_model_leaves_the_relative_gap_undefined() -> None:
    """A zero overall MAE is the one case where a relative gap cannot exist: the
    0-over-0 is reported as null with a reason, and the absolute gap (which is a
    real, measured 0.0) is still there to read."""
    report = build_report(
        source(task_type="regression", target="spend"),
        group_of_row=NORTH + SOUTH,
        y_true=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        y_pred=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        kept_counts=KEPT,
        version_counts=KEPT,
    )
    assert report["gaps_exceeding_threshold"] == []
    assert report["largest_gap"]["name"] == "mae"
    assert report["largest_gap"]["value"] == 0.0
    assert report["largest_comparable_gap"] is None, "no rate gap could be computed"
    relative = next(g for g in report["gaps"] if g["name"] == "mae_relative")
    assert relative["value"] is None
    assert relative["exceeds"] is None
    assert "overall MAE is 0" in relative["reason"]
    assert report["gaps_not_compared"] == ["mae"]


def test_a_regression_gap_in_the_targets_own_units_is_named_as_such() -> None:
    """The MAE gap is 0.5 in the Target's units, and 0.4 as a share of the overall
    MAE. Both are in the report; only the share is judged."""
    report = build_report(
        source(task_type="regression", target="spend"),
        group_of_row=NORTH + SOUTH,
        y_true=[1.0, 3.0, 5.0, 7.0, 1.0, 1.0, 1.0, 1.0],
        y_pred=[2.0, 2.0, 8.0, 8.0, 1.0, 1.0, 1.0, 5.0],
        kept_counts=KEPT,
        version_counts=KEPT,
    )
    assert report["largest_gap"]["name"] == "mae", "the widest gap of any kind"
    assert report["largest_gap"]["unit"] == "target_units"
    assert report["largest_comparable_gap"]["name"] == "mae_relative"
    assert report["gaps_not_compared"] == ["mae"]


# -- the positive class ------------------------------------------------------


def test_the_positive_class_defaults_to_the_last_sorted_class() -> None:
    assert positive_class_code(CLASSES) == (1, "bool:True")
    # Reversed order: the default follows the sorted list, not the argument order.
    assert positive_class_code(["bool:True", "bool:False"]) == (1, "bool:False")


@pytest.mark.parametrize(
    "label,expected",
    [
        (True, "bool:True"),
        (False, "bool:False"),
        ("bool:True", "bool:True"),
        ("True", "bool:True"),
        ("true", "bool:True"),
    ],
)
def test_a_named_positive_label_resolves_through_the_class_keys(label, expected) -> None:
    code, key = positive_class_code(CLASSES, label)
    assert key == expected
    assert CLASSES[code] == expected


def test_a_positive_label_that_is_not_a_class_is_refused() -> None:
    with pytest.raises(FairnessError, match="is not a class of this Target"):
        positive_class_code(CLASSES, "maybe")


def test_the_named_positive_class_flips_which_rate_is_called_positive() -> None:
    report = build_fixture_report(positive_label="False")
    assert report["positive_label"] == "False"
    assert report["positive_class"] == "bool:False"
    north = next(g for g in report["groups"] if g["group"] == group("north"))
    # north now has 1 'positive' (the one 0) and 3 negatives.
    assert north["n_actual_positive"] == 1
    assert north["metrics"]["tpr"]["value"] == 1.0
    assert north["metrics"]["fpr"]["value"] == pytest.approx(1 / 3)


# -- the threshold -----------------------------------------------------------


def test_the_default_threshold_is_the_configured_one() -> None:
    assert DEFAULT_GAP_THRESHOLD == 0.1
    assert resolve_gap_threshold(None) == 0.1
    assert resolve_gap_threshold(None, 0.25) == 0.25
    assert resolve_gap_threshold(0.0) == 0.0
    assert resolve_gap_threshold(1.0) == 1.0


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), float("inf")])
def test_an_impossible_threshold_is_refused_rather_than_clamped(bad) -> None:
    with pytest.raises(FairnessError):
        resolve_gap_threshold(bad)


def test_a_per_request_threshold_overrides_the_configured_one() -> None:
    report = build_fixture_report(threshold=0.5, threshold_source="request")
    parity = next(g for g in report["gaps"] if g["name"] == "demographic_parity")
    assert parity["threshold"] == 0.5
    assert parity["exceeds"] is False
    assert report["threshold"] == 0.5
    assert report["threshold_source"] == "request"


# -- the report's own bookkeeping --------------------------------------------


def test_the_report_records_the_run_the_model_and_the_positive_class() -> None:
    report = build_fixture_report(
        project_id="p1", seed=17, hyperparameters={"C": 1.0, "max_iter": 1000}
    )
    assert report["training_run_id"] == "run-1"
    assert report["project_id"] == "p1"
    assert report["version_id"] == "v1"
    assert report["target"] == "churned"
    assert report["model"] == "logistic_regression"
    assert report["model_label"] == "Logistic regression"
    assert report["seed"] == 17
    assert report["hyperparameters"] == {"C": 1.0, "max_iter": 1000}
    assert report["positive_class"] == "bool:True"
    assert report["classes"] == CLASSES
    assert report["refit"] is True
    assert report["split"] == {
        "train_rows": 8,
        "test_rows": 8,
        "reused_stored_split": True,
    }
    assert report["overall"]["n_test"] == 8
    assert report["overall"]["accuracy"]["value"] == pytest.approx(5 / 8)
    assert report["n_groups"] == 2 and report["n_groups_measured"] == 2


def test_the_report_says_when_the_sensitive_attribute_was_a_feature() -> None:
    """A fairness report on a column the Model was allowed to use is a different
    finding, and the report must not bury that."""
    excluded = build_fixture_report(excluded_from_features=True)
    assert excluded["sensitive_attribute_excluded_from_features"] is True
    assert not any("was a feature" in note for note in excluded["notes"])

    used = build_fixture_report(excluded_from_features=False)
    assert used["sensitive_attribute_excluded_from_features"] is False
    assert any("was a feature of this Model" in note for note in used["notes"])


def test_the_report_notes_a_different_attribute_than_the_run_declared() -> None:
    report = build_fixture_report(declared_sensitive_attribute="age")
    assert any("declared 'age'" in note for note in report["notes"])


def test_the_report_notes_the_missing_group_and_the_unmeasured_groups() -> None:
    """`missing` holds only positives, so its FPR is undefined; both facts belong
    in the notes rather than being left for the reader to infer."""
    report = build_report(
        source(),
        group_of_row=[value_key(None), value_key(None)] * 3 + [group("north")] * 3,
        y_true=[1, 1, 1, 1, 1, 1, 1, 0, 0],
        y_pred=[1, 0, 0, 0, 0, 0, 1, 0, 0],
        kept_counts={"missing": 6, group("north"): 3},
        version_counts={"missing": 6, group("north"): 3},
        classes=CLASSES,
    )
    assert any("'missing'" in note for note in report["notes"])
    assert any("no complete metric" in note for note in report["notes"])
    unmeasured = report["unmeasured_groups"]
    assert [entry["group"] for entry in unmeasured] == ["missing"]
    assert "no negative case" in unmeasured[0]["reason"]


def test_a_report_with_too_few_measured_groups_says_it_cannot_compare() -> None:
    report = build_report(
        source(),
        group_of_row=NORTH,
        y_true=TRUTH[:4],
        y_pred=PRED[:4],
        kept_counts={group("north"): 4},
        version_counts={group("north"): 4},
        classes=CLASSES,
    )
    assert any("fewer than 2 groups" in note for note in report["notes"])


def test_an_unsupported_task_type_is_refused() -> None:
    with pytest.raises(FairnessError, match="classification or regression"):
        build_report(
            source(task_type="ranking"),
            group_of_row=NORTH,
            y_true=TRUTH[:4],
            y_pred=PRED[:4],
            kept_counts=KEPT,
            classes=CLASSES,
        )


def test_a_report_never_leaks_nan_into_json() -> None:
    """A NaN in a report is a null in JSON but an error for a strict parser."""
    import json

    report = build_fixture_report()
    text = json.dumps(report, allow_nan=False)
    assert "NaN" not in text
    assert json.loads(text)["model"] == "logistic_regression"


def test_the_gap_specs_name_the_metric_they_are_computed_from() -> None:
    for spec in (*CLASSIFICATION_GAPS, *REGRESSION_GAPS):
        assert spec.metric in ("selection_rate", "tpr", "fpr", "accuracy", "mae"), spec.name
        assert spec.label and spec.description
    assert next(spec.name for spec in CLASSIFICATION_GAPS) == "demographic_parity"
    assert [spec.name for spec in REGRESSION_GAPS] == ["mae", "mae_relative"]


def test_measured_gaps_keeps_the_ones_a_threshold_cannot_judge() -> None:
    """The absolute MAE gap is in the Target's units, so no threshold can call it
    a finding — but it is still the widest gap there is, and dropping it from
    `measured_gaps` would hide the fact."""
    report = build_report(
        source(task_type="regression", target="spend"),
        group_of_row=NORTH + SOUTH,
        y_true=[1.0, 3.0, 5.0, 7.0, 1.0, 1.0, 1.0, 1.0],
        y_pred=[2.0, 2.0, 8.0, 8.0, 1.0, 1.0, 1.0, 5.0],
        kept_counts=KEPT,
        version_counts=KEPT,
    )
    assert [gap["name"] for gap in measured_gaps(report["gaps"])] == ["mae", "mae_relative"]
    assert [gap["name"] for gap in comparable_gaps(report["gaps"])] == ["mae_relative"]
    # A gap with no value is in neither list.
    empty = build_fixture_report(threshold=0.9)
    assert measured_gaps(empty["gaps"]) != []
    assert all(gap["value"] is not None for gap in measured_gaps(empty["gaps"]))


def test_group_helpers_are_stable_and_sorted() -> None:
    values = ["south", "north", "south", None]
    assert group_sizes(values) == {"missing": 1, "str:north": 1, "str:south": 2}
    assert list(group_sizes(values)) == sorted(group_sizes(values))
    assert group_keys(values) == ["str:south", "str:north", "str:south", "missing"]


def test_a_report_over_a_frame_agrees_with_the_underlying_functions() -> None:
    """The report is a thin layer: the numbers in it are the numbers the pure
    group functions produce, not a second, divergent implementation."""
    frame = pd.DataFrame(
        {
            "region": NORTH + SOUTH + [group("east")] * 2,
            "truth": TRUTH + [1, 0],
            "pred": PRED + [1, 0],
        }
    )
    counts = {**KEPT, group("east"): 2}
    report = build_report(
        source(),
        group_of_row=list(frame["region"]),
        y_true=list(frame["truth"]),
        y_pred=list(frame["pred"]),
        kept_counts=counts,
        version_counts=counts,
        classes=CLASSES,
    )
    # 'east' contributes one positive and one negative, so all three groups are
    # fully measured and the parity gap compares all three.
    assert report["groups"] == classification_groups(
        group_of_row=list(frame["region"]),
        y_true=list(frame["truth"]),
        y_pred=list(frame["pred"]),
        positive=1,
        kept_counts=counts,
        version_counts=counts,
        classes=CLASSES,
    )
    assert report["n_groups"] == 3
    assert report["n_groups_measured"] == 3
    assert report["unmeasured_groups"] == []
    parity = next(g for g in report["gaps"] if g["name"] == "demographic_parity")
    assert parity["groups_compared"] == 3 and parity["groups_total"] == 3


def test_values_from_a_real_dataframe_go_through_the_same_encoding() -> None:
    """`value_key` is the join between the group column and the Target, so a
    nullable boolean Target and a string attribute land on the same footing."""
    frame = pd.DataFrame(
        {
            "region": pd.array(["north", "south", "north", None], dtype="string"),
            "churned": pd.array([True, None, False, True], dtype="boolean"),
        }
    )
    report = build_report(
        source(),
        group_of_row=[value_key(v) for v in frame["region"]],
        y_true=[1, 1, 0, 1],
        y_pred=[1, 0, 0, 1],
        kept_counts=group_sizes(frame["region"]),
        version_counts=group_sizes(frame["region"]),
        classes=CLASSES,
    )
    names = {g["group"] for g in report["groups"]}
    assert names == {"missing", "str:north", "str:south"}
    north = next(g for g in report["groups"] if g["group"] == "str:north")
    assert north["n_test"] == 2
    assert north["class_counts"] == {"bool:True": 1, "bool:False": 1}


def test_the_group_matrix_is_not_handed_out_when_it_does_not_line_up() -> None:
    """Sanity: a genuinely consistent fixture really does produce a report, so
    the refusals above are about the mismatches and not about everything."""
    assert np.asarray(TRUTH).shape == np.asarray(PRED).shape == (8,)
    assert len(NORTH) + len(SOUTH) == 8
