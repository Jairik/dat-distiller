"""Profile building from samples and Column Specs."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from dat_distiller.profile import (
    ColumnSpec,
    Profile,
    build_profile,
    profile_from_specs,
    validate_specs,
)


def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "age": rng.normal(40, 5, 200).round(1),
            "score": rng.integers(0, 100, 200),
            "churned": rng.random(200) < 0.3,
            "plan": rng.choice(["basic", "pro", "team"], 200),
            "notes": [f"a review of item {i} mentioning quality {i % 7}" for i in range(200)],
        }
    )


def test_profile_from_sample_captures_marginals_frequencies_missingness() -> None:
    df = sample_df()
    df.loc[0:19, "score"] = None  # 10% missing
    profile = build_profile(df)

    age = profile.column("age")
    assert age.kind == "number"
    assert age.mean is not None and abs(age.mean - 40) < 1.5
    assert len(age.quantiles) == 7 and age.quantiles[0] <= age.quantiles[-1]

    plan = profile.column("plan")
    assert plan.kind == "categorical"
    assert sum(plan.frequencies.values()) == 200
    assert plan.categories == ["basic", "pro", "team"]

    churned = profile.column("churned")
    assert churned.kind == "bool"
    assert set(churned.frequencies) == {"True", "False"}

    score = profile.column("score")
    assert score.missing_rate == pytest.approx(0.1)
    # int columns with nulls become float-backed -> "number" (pandas semantics)
    assert score.kind == "number"
    assert build_profile(pd.DataFrame({"score": [1, 2, 3]})).column("score").kind == "integer"

    notes = profile.column("notes")
    assert notes.kind == "text"
    assert notes.mean_length > 20 and notes.cardinality == 200


def test_correlations_recorded() -> None:
    df = pd.DataFrame({"x": np.arange(50, dtype=float), "y": np.arange(50, dtype=float) * 2})
    profile = build_profile(df)
    assert profile.correlations["x"]["y"] == pytest.approx(1.0)


def test_profile_is_json_round_trippable() -> None:
    profile = build_profile(sample_df())
    payload = json.loads(json.dumps(profile.to_json_dict()))
    back = Profile.from_json_dict(payload)
    assert back.to_json_dict() == profile.to_json_dict()
    assert json.dumps(profile.to_json_dict())  # no NaN -> invalid JSON


def test_profile_ignores_provenance_column() -> None:
    df = sample_df()
    df["__provenance__"] = '{"row_origin": "uploaded"}'
    profile = build_profile(df)
    assert profile.column("__provenance__") is None
    assert [c.name for c in profile.columns] == ["age", "score", "churned", "plan", "notes"]


def test_profile_from_specs_alone() -> None:
    specs = [
        ColumnSpec(name="price", type="number", min=0, max=100, description="unit price"),
        ColumnSpec(name="plan", type="categorical", categories=["basic", "pro"]),
        ColumnSpec(name="review", type="text", description="free text"),
    ]
    profile = profile_from_specs(specs)
    assert profile.source == "specs"
    price = profile.column("price")
    assert (price.min, price.max) == (0.0, 100.0)
    assert price.declared_only and price.description == "unit price"
    assert profile.column("plan").categories == ["basic", "pro"]
    assert profile.correlations == {}


def test_spec_validation_errors() -> None:
    bad = [
        ColumnSpec(name="a", type="number", min=5, max=1),
        ColumnSpec(name="a", type="categorical"),
        ColumnSpec(name="b", type="widget"),
        ColumnSpec(name="", type="text"),
    ]
    errors = validate_specs(bad)
    joined = " | ".join(errors)
    assert "min must be <= max" in joined
    assert "duplicate name" in joined
    assert "at least one category" in joined
    assert "type must be one of" in joined
    assert "name is required" in joined
    with pytest.raises(ValueError):
        profile_from_specs(bad)
