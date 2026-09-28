"""Optional extras, and whether this install can actually use them.

torch, TensorFlow, Presidio, scikit-learn and LightGBM are declared as optional
extras in `pyproject.toml` and are deliberately absent from the dev environment.
Anything that needs one asks here first and degrades with a "not installed"
message.
"""

from __future__ import annotations

from importlib.util import find_spec

#: Extras name -> module that must import for the extra to be usable.
EXTRA_MODULES: dict[str, str] = {
    "torch": "torch",
    "tensorflow": "tensorflow",
    "presidio": "presidio_analyzer",
    "sklearn": "sklearn",
    "lightgbm": "lightgbm",
}


def is_installed(module: str) -> bool:
    """Whether `module` resolves in this environment.

    Probes with `find_spec` rather than importing, so answering stays cheap and
    a half-installed package cannot raise its way into a request.
    """
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def installed_extras() -> dict[str, bool]:
    """Install status of every optional extra, keyed by extras name."""
    return {name: is_installed(module) for name, module in EXTRA_MODULES.items()}
