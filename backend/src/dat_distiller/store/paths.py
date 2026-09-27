"""Where the app keeps its data on local disk."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DATA_DIR_ENV = "DAT_DISTILLER_DATA_DIR"


def data_dir() -> Path:
    """The data directory, created on demand.

    ``DAT_DISTILLER_DATA_DIR`` overrides the default so tests (and users with
    unusual disks) can redirect all storage. The default sits in the current
    working directory (gitignored), which is the repo when run via
    ``uv run dat-distiller serve``.
    """
    raw = os.environ.get(DATA_DIR_ENV)
    root = Path(raw) if raw else Path.cwd() / "dat_distiller_data"
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass(frozen=True)
class AppPaths:
    """Resolved paths inside a data directory."""

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "AppPaths":
        return cls(root=data_dir())

    @property
    def db_path(self) -> Path:
        return self.root / "app.db"

    @property
    def versions_dir(self) -> Path:
        out = self.root / "versions"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def version_file(self, version_id: str) -> Path:
        return self.versions_dir / f"{version_id}.parquet"
