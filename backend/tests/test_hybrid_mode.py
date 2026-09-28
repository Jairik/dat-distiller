"""Hybrid mode and Provider seed sets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dat_distiller.generate.modes import fill_text_columns, generate_with_mode
from dat_distiller.profile import ColumnSpec, build_profile, profile_from_specs
from dat_distiller.providers import FakeProvider


def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "age": rng.integers(20, 70, 300),
            "score": rng.uniform(0, 1, 300),
            "review": [f"placeholder review {i}" for i in range(300)],
        }
    )


def text_provider():
    """Deterministic provider that writes reviews mentioning the row's age."""

    def responder(prompt, schema):
        if "For EACH" in prompt:
            n = int(prompt.split("For EACH of the ")[1].split(" ")[0])
            return {"rows": [{"review": f"My age is {20 + i} and it is great"} for i in range(n)]}
        return {"rows": [{"review": "generic"}]}

    return FakeProvider(response_for=responder)


def test_hybrid_copula_structured_provider_text() -> None:
    provider = text_provider()
    profile = build_profile(sample_df())
    result = generate_with_mode(provider, "hybrid", profile, "people", 40, sample_df=sample_df(), seed=11)
    assert list(result.rows.columns) == ["age", "score", "review"]
    assert len(result.rows) == 40
    # structured columns came from the copula: within source ranges
    assert result.rows["age"].between(20, 69).all()
    assert result.rows["score"].between(0, 1).all()
    # text column came from the Provider, conditioned on each structured row
    assert result.rows["review"].str.startswith("My age is ").all()
    assert '"age"' in provider.calls[0]["prompt"]  # structured values were in the prompt
    assert result.provider_text_rows == 40


def test_hybrid_without_text_is_pure_copula() -> None:
    df = sample_df().drop(columns=["review"])
    provider = FakeProvider()
    result = generate_with_mode(provider, "hybrid", build_profile(df), "people", 25, sample_df=df, seed=2)
    assert list(result.rows.columns) == ["age", "score"]
    assert provider.calls == []  # nothing needed the Provider


def test_no_sample_statistical_generates_seed_set_and_flags_profile() -> None:
    specs = [
        ColumnSpec(name="price", type="number", min=1, max=50),
        ColumnSpec(name="tier", type="categorical", categories=["a", "b"]),
    ]
    provider = FakeProvider()  # synthesizes schema-conforming rows
    spec_profile = profile_from_specs(specs)
    result = generate_with_mode(provider, "statistical", spec_profile, "products", 100, seed=3)
    assert result.profile.provider_derived is True  # Fidelity will warn (#21)
    assert provider.calls  # the seed set came from the Provider
    assert len(result.rows) == 100
    assert result.rows["price"].between(1, 50).all()
    assert set(result.rows["tier"].unique()) <= {"a", "b"}


def test_no_sample_hybrid_also_flags_profile_and_fills_text() -> None:
    def responder(prompt, schema):
        if "For EACH" in prompt:
            n = int(prompt.split("For EACH of the ")[1].split(" ")[0])
            return {"rows": [{"blurb": f"nice item number {i}"} for i in range(n)]}
        return {
            "rows": [
                {"price": float(i % 40) + 2, "tier": ["a", "b"][i % 2], "blurb": f"seed {i} text long enough"}
                for i in range(50)
            ]
        }

    provider = FakeProvider(response_for=responder)
    specs = [
        ColumnSpec(name="price", type="number", min=1, max=50),
        ColumnSpec(name="tier", type="categorical", categories=["a", "b"]),
        ColumnSpec(name="blurb", type="text"),
    ]
    result = generate_with_mode(provider, "hybrid", profile_from_specs(specs), "items", 60, seed=5)
    assert result.profile.provider_derived is True
    assert len(result.rows) == 60
    assert result.rows["blurb"].notna().all()


def test_statistical_mode_leaves_text_null() -> None:
    profile = build_profile(sample_df())
    result = generate_with_mode(
        FakeProvider(), "statistical", profile, "people", 10, sample_df=sample_df(), seed=1
    )
    assert result.rows["review"].isna().all()
    assert result.dropped == 10


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError, match="unknown Generation Mode"):
        generate_with_mode(FakeProvider(), "vibes", profile_from_specs([ColumnSpec(name="a", type="text")]), "x", 5)


def test_text_fill_keeps_going_when_provider_returns_wrong_row_count() -> None:
    calls = {"n": 0}

    def responder(prompt, schema):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"rows": [{"review": "too few rows"}]}  # wrong count -> retry
        n = int(prompt.split("For EACH of the ")[1].split(" ")[0])
        return {"rows": [{"review": f"review number {i}"} for i in range(n)]}

    provider = FakeProvider(response_for=responder)
    frame = pd.DataFrame({"age": range(4), "score": [0.5] * 4})
    filled, dropped = fill_text_columns(
        provider, frame.assign(review=None), ["review"], "people", batch_size=4
    )
    assert filled["review"].notna().all()
    assert dropped == 0
