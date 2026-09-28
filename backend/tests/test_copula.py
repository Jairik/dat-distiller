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


# -- a column with no values at all ------------------------------------------


def test_a_column_that_is_entirely_null_samples_as_missing() -> None:
    """An all-null column has no distribution, so missing is the honest sample.

    A column added in advance, a parse that failed, a spreadsheet with a trailing
    header — all ordinary. Previously the numeric path divided by zero fitting
    the marginal and then indexed an empty array sampling from it, so the whole
    Generation Run died with `IndexError: index -1 is out of bounds for axis 0
    with size 0`. The all-null *boolean* column already worked and sampled as
    `None`; this makes the numeric and categorical paths agree with it.
    """
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            # float64/NaN, which is what an empty column in an uploaded CSV
            # actually parses to. A column of Python `None` would be dtype
            # object, and `infer_kind` reads *that* as free text — see the test
            # below, which pins the difference rather than papering over it.
            "numeric": [float("nan")] * 3,
            "flag": pd.array([None, None, None], dtype="boolean"),
            "words": pd.array(["x", "y", "z"], dtype="object"),
        }
    )
    copula = GaussianCopula.fit(frame)
    # The column is still there: the user put it there.
    assert "numeric" in copula.columns
    assert "flag" in copula.columns

    sampled = copula.sample(4, seed=0)
    assert len(sampled) == 4
    for column in ("numeric", "flag"):
        assert sampled[column].isna().all(), f"{column} should sample as missing"
    # And the columns that had values are untouched: every value drawn is one
    # that was actually observed, which is the copula's whole contract.
    assert sampled["a"].notna().all()
    assert set(sampled["words"]) <= {"x", "y", "z"}


def test_an_all_null_object_column_is_still_treated_as_free_text() -> None:
    """A dtype-based kind, so an all-null object column has no evidence of type.

    `infer_kind` reads float, integer, bool and datetime off the dtype, so those
    survive an empty column and are handled above. An object column with no
    values is indistinguishable from free text, and free text is the Provider's
    job — so it is skipped for the copula exactly as it always was. This is
    pinned so the change above is not later read as "empty columns are never
    skipped".
    """
    frame = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "maybe_text": pd.array([None, None, None], dtype="object")}
    )
    copula = GaussianCopula.fit(frame)
    assert "maybe_text" in copula.skipped
    assert "maybe_text" not in copula.columns


def test_a_fully_null_column_emits_no_runtime_warning(recwarn) -> None:
    """Fitting one must not warn about an invalid division on the way past."""
    frame = pd.DataFrame({"a": [1.0, 2.0], "empty": [float("nan")] * 2})
    GaussianCopula.fit(frame).sample(2, seed=0)
    assert [w for w in recwarn if "invalid value" in str(w.message)] == []


def test_a_mostly_null_column_still_learns_from_what_there_is() -> None:
    """Unobserved is not the same as rare: one value is enough to fit."""
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "sparse": [7.0] + [float("nan")] * 3})
    sampled = GaussianCopula.fit(frame).sample(6, seed=0)
    observed = sampled["sparse"].dropna()
    # Every value it can produce is the one it saw.
    assert set(observed.unique()) == {7.0}
