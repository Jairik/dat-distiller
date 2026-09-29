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


# -- a missing timestamp is a gap, not a number ------------------------------


def dated_pair(rows: int = 200, seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A sample and a generated set sharing a datetime column, with a gap in one."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2021-01-01")

    def frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "when": pd.Series(
                    [base + pd.Timedelta(days=int(d)) for d in rng.integers(0, 900, rows)],
                    dtype="datetime64[ns]",
                ),
                "amount": rng.normal(10, 2, rows),
            }
        )

    sample, generated = frame(), frame()
    # One gap in the sample only — a date that failed to parse, a blank cell.
    sample.loc[3, "when"] = pd.NaT
    return sample, generated


def test_the_frame_the_fidelity_distances_are_measured_in_has_no_sentinel() -> None:
    """The numbers the distances are computed from must not contain the sentinel.

    `NaT` cast to int64 is the int64 minimum, which is finite — so it survived the
    `.dropna()` in `_ks_tests` and entered the comparison as a real observation
    about 5e10 seconds from every other date. The KS statistic barely moves,
    because one extreme point leaves both ECDFs at zero below it and the curves
    still track; the corruption is in the *input*, not visibly in that one
    number. So this asserts on the input, which is the thing that was wrong.
    """
    from dat_distiller.generate.fidelity import _numeric_frame

    sample, _generated = dated_pair()
    numbers = _numeric_frame(sample)
    assert "when" in numbers.columns
    column = numbers["when"].to_numpy()
    assert np.isnan(column[3]), "the gap should be a gap"
    real = column[~np.isnan(column)]
    # Every real observation is a real date; the sentinel is before 1970.
    assert real.min() > 1_000_000_000
    assert (real > 0).all()


def test_the_copulas_datetime_marginal_ignores_a_missing_timestamp() -> None:
    """One gap must not become a date the copula can generate.

    The copula interpolates between the values it was fitted on, so a sentinel
    inside `sorted_values` is not just a bad input — it is a value it can draw,
    and quantile interpolation will happily return a date near 1677.
    """
    from dat_distiller.generate.copula import marginal_values

    series = pd.Series(pd.to_datetime(["2021-01-01", None, "2021-06-01"], utc=True))
    values = marginal_values("datetime", series)
    assert np.isnan(values[1])
    assert np.nanmin(values) > 1_000_000_000
    assert np.nanmedian(values) == pytest.approx(
        float(pd.Timestamp("2021-03-17", tz="UTC").value / 1e9), rel=0.01
    )


def test_generated_datetimes_land_in_the_same_era_as_the_sample() -> None:
    """The conversion and its inverse have to agree on the unit.

    Worth its own test because the two can drift apart without either raising:
    a marginal fitted in seconds and read back as nanoseconds generates 1970 for
    every row, which is a valid datetime and a completely wrong dataset.
    """
    rng = np.random.default_rng(3)
    base = pd.Timestamp("2021-01-01")
    when = [base + pd.Timedelta(days=int(day)) for day in rng.integers(0, 900, 300)]
    when[7] = pd.NaT
    frame = pd.DataFrame(
        {
            "when": pd.Series(when, dtype="datetime64[ns]"),
            "amount": rng.normal(10, 2, 300),
        }
    )
    generated = GaussianCopula.fit(frame).sample(200, seed=0)
    dates = pd.to_datetime(generated["when"], errors="coerce")
    assert dates.notna().all()
    assert dates.min() >= pd.Timestamp("2020-01-01")
    assert dates.max() <= pd.Timestamp("2025-01-01")
    # Resolution-agnostic: a column carried at [s] or [us] must still be read as
    # the dates it is, not as a different era entirely.
    assert dates.dt.year.between(2021, 2023).all()


def test_the_shared_conversion_is_the_one_every_caller_uses() -> None:
    """`epoch_seconds` is the definition, and NaT is a gap in it.

    The same expression was written out five times across the fitter, the copula
    and the fidelity report, which is how it came to be fixed in one place and
    missed in the others. This pins the helper itself rather than each caller.
    """
    from dat_distiller.store.columns import epoch_seconds

    series = pd.Series(pd.to_datetime(["2020-01-01", None, "2020-01-03"], utc=True))
    values = epoch_seconds(series)
    assert np.isnan(values[1])
    assert values[0] < values[2]
    # And the sentinel is nowhere in sight.
    assert np.isfinite(values[[0, 2]]).all()
