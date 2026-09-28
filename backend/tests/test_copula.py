"""Gaussian copula: marginals & correlations preserved, balance, determinism.

No SDV / `copulas` (ADR-0002) — assert the module never imports them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dat_distiller.generate.copula import GaussianCopula


def source_df(n: int = 1500, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(50, 10, n)
    y = 0.8 * x + rng.normal(0, 6, n)  # strongly correlated with x
    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "count": rng.integers(0, 100, n),
            "churned": rng.random(n) < 0.3,
            "plan": rng.choice(["basic", "pro", "team"], n, p=[0.5, 0.3, 0.2]),
            "signup": pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 1500, n), unit="D"),
            "review": [f"free text {i}" for i in range(n)],
        }
    )


def test_fit_preserves_supported_columns_and_skips_text() -> None:
    copula = GaussianCopula.fit(source_df())
    assert copula.skipped == ["review"]  # free text belongs to the Provider
    assert set(copula.columns) == {"x", "y", "count", "churned", "plan", "signup"}


def test_samples_keep_marginals_and_correlations() -> None:
    src = source_df()
    copula = GaussianCopula.fit(src)
    gen = copula.sample(3000, seed=7)
    assert len(gen) == 3000

    # numeric marginals close
    assert abs(gen["x"].mean() - src["x"].mean()) < 1.5
    assert abs(gen["x"].std() - src["x"].std()) < 2.0
    # correlation x~y roughly preserved
    r_src = np.corrcoef(src["x"], src["y"])[0, 1]
    r_gen = np.corrcoef(gen["x"], gen["y"])[0, 1]
    assert abs(r_src - r_gen) < 0.12

    # integer column stays integral and within range
    assert gen["count"].dropna().mod(1).eq(0).all()
    assert gen["count"].min() >= src["count"].min() and gen["count"].max() <= src["count"].max()

    # categorical frequencies close
    src_share = src["plan"].value_counts(normalize=True)
    gen_share = gen["plan"].value_counts(normalize=True)
    for cat in ["basic", "pro", "team"]:
        assert abs(src_share[cat] - gen_share[cat]) < 0.05

    # bool stays bool, datetime stays datetime
    assert gen["churned"].dropna().isin([True, False]).all()
    assert pd.api.types.is_datetime64_any_dtype(gen["signup"])


def test_balance_targets_met_within_tolerance() -> None:
    copula = GaussianCopula.fit(source_df())
    gen = copula.sample(2000, seed=3, balance={"churned": {"True": 0.5, "False": 0.5}})
    share = gen["churned"].astype(str).value_counts(normalize=True)
    assert abs(share["True"] - 0.5) < 0.05
    assert abs(share["False"] - 0.5) < 0.05


def test_balance_on_multi_category_column() -> None:
    copula = GaussianCopula.fit(source_df())
    gen = copula.sample(
        1500, seed=5, balance={"plan": {"basic": 0.2, "pro": 0.4, "team": 0.4}}
    )
    share = gen["plan"].value_counts(normalize=True)
    for cat, want in [("basic", 0.2), ("pro", 0.4), ("team", 0.4)]:
        assert abs(share.get(cat, 0) - want) < 0.05


def test_same_seed_identical_output_different_seed_differs() -> None:
    copula = GaussianCopula.fit(source_df())
    a = copula.sample(500, seed=42)
    b = copula.sample(500, seed=42)
    c = copula.sample(500, seed=43)
    pd.testing.assert_frame_equal(a, b)
    assert not a["x"].equals(c["x"])


def test_missingness_is_reinjected() -> None:
    src = source_df()
    src.loc[::5, "x"] = None  # 20% missing
    copula = GaussianCopula.fit(src)
    gen = copula.sample(1500, seed=9)
    assert abs(gen["x"].isna().mean() - 0.2) < 0.05


def test_no_bsl_libraries_imported() -> None:
    import dat_distiller.generate.copula as module
    import sys

    src = open(module.__file__).read()
    for banned in ("import sdv", "from sdv", "import copulas", "from copulas"):
        assert banned not in src
    assert not any(name.split(".")[0] in {"sdv", "copulas"} for name in sys.modules)
