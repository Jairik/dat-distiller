"""The Project and Dataset Version store: SQLite metadata plus Parquet data.

Storage layout (``AppPaths``): everything lives under a data directory taken
from the ``DAT_DISTILLER_DATA_DIR`` environment variable, defaulting to
``dat_distiller_data/`` in the repo root.

    <data>/app.db                     SQLite metadata (projects, dataset versions)
    <data>/versions/<version_id>.parquet   one immutable data file per Dataset Version

Every row of every Dataset Version carries its Provenance in a reserved
``__provenance__`` column (see ``provenance.py``). Dataset Versions are
immutable: the Parquet file is written once at creation and never modified;
every change creates a child version pointing at its ``parent_id``, which is
what forms the per-Project version tree.
"""

from .columns import ColumnInfo, infer_column_info
from .paths import AppPaths, data_dir
from .provenance import (
    PROVENANCE_COLUMN,
    Provenance,
    provenance_breakdown,
)
from .store import DatasetStore, DatasetVersion, Project

__all__ = [
    "AppPaths",
    "ColumnInfo",
    "DatasetStore",
    "DatasetVersion",
    "PROVENANCE_COLUMN",
    "Project",
    "Provenance",
    "data_dir",
    "infer_column_info",
    "provenance_breakdown",
]
