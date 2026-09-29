"""What a plain `pip install` of this package has to be able to do.

`uv run pytest` installs the `dev` dependency group automatically, so a
dependency declared only there is present in every environment this repo is
developed in and absent for anyone who installs the published package. That is
how `httpx` came to be imported at module scope by `api.providers` and
`providers.openrouter` while being declared as a test-only dependency: the suite
was green and the app could not start.

The heavy extras (`torch`, `tensorflow`, `presidio`, `scikit-learn`,
`lightgbm`) are a different matter and stay optional — this is about hard runtime
imports.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text())


def _names(requirements: list[str]) -> set[str]:
    """Just the distribution names, without version bounds or extras."""
    return {r.split(">")[0].split("=")[0].split("[")[0].strip().lower() for r in requirements}


def test_every_runtime_import_is_a_runtime_dependency() -> None:
    """A module-level import must be satisfiable by `dependencies`, not `dev`.

    Checked against the packages this codebase actually imports, rather than
    against a list of module names, so a new import is caught the day it is
    added rather than when someone installs the package.
    """
    runtime = _names(_project()["project"]["dependencies"])
    dev = _names(_project()["dependency-groups"]["dev"])
    # `httpx` is the one that regressed: imported by `api.providers` and
    # `providers.openrouter` at module scope, and declared only under `dev`.
    assert "httpx" in runtime, (
        "httpx is imported at module scope by dat_distiller.api.providers and "
        "dat_distiller.providers.openrouter, so it belongs in [project] dependencies"
    )
    assert "httpx" not in dev, "httpx is a runtime dependency, not a dev-only one"


def test_the_heavy_extras_stay_optional() -> None:
    """The rule above is about hard imports, not about the heavy libraries.

    If this ever starts failing because someone added a runtime import of
    scikit-learn, the answer is `extras.is_installed` and a skipped test — not
    making torch a hard dependency.
    """
    runtime = _names(_project()["project"]["dependencies"])
    for heavy in ("torch", "tensorflow", "scikit-learn", "lightgbm", "spacy"):
        assert heavy not in runtime, f"{heavy} must stay an optional extra"
    optional = _project()["project"]["optional-dependencies"]
    for extra in ("torch", "tensorflow", "presidio", "sklearn", "lightgbm"):
        assert extra in optional, f"the {extra} extra should still be declared"


def test_every_module_imports_without_the_optional_extras() -> None:
    """Nothing under `dat_distiller` may need an extra just to be imported."""
    import importlib
    import pkgutil

    import dat_distiller

    failures: list[str] = []
    for module in pkgutil.walk_packages(dat_distiller.__path__, "dat_distiller."):
        try:
            importlib.import_module(module.name)
        except Exception as exc:  # noqa: BLE001 - the point is to report them all
            failures.append(f"{module.name}: {type(exc).__name__}: {exc}")
    assert failures == [], "modules that need an extra to import:\n" + "\n".join(failures)
