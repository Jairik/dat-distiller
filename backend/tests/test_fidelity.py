"""Fidelity Report: statistics, near-copies, checks, JSON-serializability."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from dat_distiller.checks import CheckStore
from dat_distiller.generate.copula import GaussianCopula
from dat_distiller.generate.fidelity import (
    build_fidelity_report,
    raise_fidelity_checks,
)
from dat_distiller.store.db import Database
from dat_distiller.store.paths import AppPaths
from dat_distiller.store.store import DatasetStore


@pytest.fixture
def sample() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x = rng.normal(50, 10, 800)
    return pd.DataFrame(
        {
            "x": x,
            "y": 0.7 * x + rng.normal(0, 6, 800),
            "plan": rng.choice(["basic", "pro", "team"], 800, p=[0.5, 0.3, 0.2]),
            "churned": rng.random(800) < 0.3,
        }
    )


def good_generated(sample: pd.DataFrame, n: int = 600, seed: int = 11) -> pd.DataFrame:
    return GaussianCopula.fit(sample).sample(n, seed=seed)


def test_faithful_generation_reports_no_warnings(sample) -> None:
    report = build_fidelity_report(sample, good_generated(sample))
    assert report["warnings"] == []
    assert report["near_copies"]["count"] == 0
    assert report["exact_duplicates"]["count"] == 0
    assert report["provider_derived_profile"] is False
    assert report["correlation_drift"] < 0.2
    json.dumps(report)  # AC: JSON-serializable


def test_near_copies_detected_and_threshold_is_configurable(sample) -> None:
    generated = good_generated(sample, 300, seed=4)
    # inject three exact copies of sample rows at known positions
    copied = sample.iloc[[10, 20, 30]].reset_index(drop=True)
    generated.iloc[5] = copied.iloc[0]
    generated.iloc[77] = copied.iloc[1]
    generated.iloc[150] = copied.iloc[2]
    report = build_fidelity_report(sample, generated, near_copy_threshold=1e-6)
    assert report["near_copies"]["count"] == 3
    assert set(report["near_copies"]["examples"]) == {5, 77, 150}
    assert "near_copies" in report["warnings"]
    # a generous threshold sweeps in ordinary rows too: configurable
    loose = build_fidelity_report(sample, generated, near_copy_threshold=0.2)
    assert loose["near_copies"]["count"] > 3


def test_exact_duplicates_counted(sample) -> None:
    generated = good_generated(sample, 100, seed=6)
    generated = pd.concat([generated, generated.head(10)], ignore_index=True)
    report = build_fidelity_report(sample, generated)
    assert report["exact_duplicates"]["count"] >= 10


def test_high_drop_ratio_and_provider_profile_raise_codes(sample) -> None:
    report = build_fidelity_report(
        sample, good_generated(sample, 200, seed=8), dropped_rows=50, requested_count=200
    )
    assert report["dropped_ratio"] == 0.25
    assert "dropped_rows_high" in report["warnings"]
    derived = build_fidelity_report(sample, good_generated(sample, 50, seed=9), profile_derived=True)
    assert "provider_derived_profile" in derived["warnings"]


def test_distribution_shift_raises_drift_warning(sample) -> None:
    generated = good_generated(sample, 600, seed=12)
    generated["x"] = generated["x"] + 25  # shifted well outside the sample bulk
    report = build_fidelity_report(sample, generated)
    assert "fidelity_drift" in report["warnings"]


def test_balance_actual_versus_requested(sample) -> None:
    generated = GaussianCopula.fit(sample).sample(
        500, seed=13, balance={"churned": {"True": 0.5, "False": 0.5}}
    )
    report = build_fidelity_report(sample, generated, balance={"churned": {"True": 0.5, "False": 0.5}})
    actual = report["balance"]["churned"]["True"]
    assert actual["requested"] == 0.5
    assert abs(actual["actual"] - 0.5) < 0.06


def test_checks_raised_once_per_version(sample, isolated_data_dir) -> None:
    paths = AppPaths(root=isolated_data_dir)
    store = DatasetStore(paths)
    checks = CheckStore(store.db)
    project = store.create_project("fidelity")
    report = build_fidelity_report(
        sample, good_generated(sample, 200, seed=14), dropped_rows=80, profile_derived=True
    )
    version = store.create_version(
        project.id,
        _rows(sample, 5),
        origin="generate",
        meta={"fidelity": json.loads(json.dumps(report))},
    )
    raised = raise_fidelity_checks(checks, version.id, report)
    assert {"dropped_rows_high", "provider_derived_profile"} <= set(raised)
    # deduped per Version
    assert raise_fidelity_checks(checks, version.id, report) == []
    warnings = checks.unacknowledged_warnings("dataset_version", version.id)
    kinds = {check.kind for check in warnings}
    assert {"dropped_rows_high", "provider_derived_profile"} <= kinds
    # report survived the round-trip through the Dataset Version meta
    reloaded = store.get_version(version.id)
    assert reloaded.meta["fidelity"]["near_copies"]["threshold"] == pytest.approx(
        report["near_copies"]["threshold"]
    )


def _rows(sample: pd.DataFrame, n: int) -> pd.DataFrame:
    return good_generated(sample, 20, seed=n)


def test_empty_and_text_only_frames_do_not_crash() -> None:
    sample = pd.DataFrame({"a": [1, 2, 3], "review": ["long text one", "long text two", "text three"]})
    generated = pd.DataFrame({"a": [5, 6, 7], "review": ["fresh", "words", "here"]})
    report = build_fidelity_report(sample, generated)
    assert math.isfinite(report["correlation_drift"])
    assert report["near_copies"]["count"] == 0  # text ignored
