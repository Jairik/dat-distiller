"""The domain store: Projects, immutable Dataset Versions, Provenance.

FastAPI-free on purpose — the API layer in ``dat_distiller.api`` is thin."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .columns import ColumnInfo, infer_column_info
from .db import Database
from .paths import AppPaths
from .provenance import PROVENANCE_COLUMN, Provenance, provenance_breakdown


class DuplicateProjectError(ValueError):
    pass


class ProjectNotFoundError(KeyError):
    pass


class DatasetVersionNotFoundError(KeyError):
    pass


class ParentNotFoundError(KeyError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    created_at: str


@dataclass(frozen=True)
class DatasetVersion:
    id: str
    project_id: str
    parent_id: str | None
    number: int
    origin: str
    row_count: int
    columns: list[ColumnInfo]
    provenance_summary: dict[str, int]
    seed: int | None
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "parent_id": self.parent_id,
            "number": self.number,
            "origin": self.origin,
            "row_count": self.row_count,
            "columns": [c.to_dict() for c in self.columns],
            "provenance_summary": self.provenance_summary,
            "seed": self.seed,
            "meta": self.meta,
            "created_at": self.created_at,
        }


class DatasetStore:
    """Projects + immutable Dataset Version tree backed by SQLite + Parquet."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.db = Database(paths.db_path)

    # -- Projects ---------------------------------------------------------

    def create_project(self, name: str) -> Project:
        name = name.strip()
        if not name:
            raise ValueError("project name must not be empty")
        project = Project(id=uuid.uuid4().hex, name=name, created_at=_now())
        with self.db.connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO projects (id, name, created_at) VALUES (?, ?, ?)",
                    (project.id, project.name, project.created_at),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateProjectError(name) from exc
        return project

    def list_projects(self) -> list[Project]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT id, name, created_at FROM projects ORDER BY created_at, name"
            ).fetchall()
        return [Project(**dict(row)) for row in rows]

    def get_project(self, project_id: str) -> Project:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT id, name, created_at FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
        if row is None:
            raise ProjectNotFoundError(project_id)
        return Project(**dict(row))

    def delete_project(self, project_id: str) -> None:
        self.get_project(project_id)
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT id FROM dataset_versions WHERE project_id = ?", (project_id,)
            ).fetchall()
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        for row in rows:
            path = self.paths.version_file(row["id"])
            path.unlink(missing_ok=True)

    # -- Dataset Versions ---------------------------------------------------

    def create_version(
        self,
        project_id: str,
        df: pd.DataFrame,
        *,
        parent_id: str | None = None,
        origin: str = "uploaded",
        provenance: list[dict[str, Any]] | None = None,
        seed: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> DatasetVersion:
        """Create a new immutable Dataset Version and its Parquet file.

        ``df`` may or may not already contain the ``__provenance__`` column
        (e.g. when derived from ``load_dataframe(..., include_provenance=True)``
        — existing Provenance survives unless overridden). Pass an explicit
        per-row ``provenance`` list to set/extend it; otherwise rows default to
        ``origin``.
        """
        self.get_project(project_id)  # 404 at the API layer via exception
        if parent_id is not None and parent_id not in {
            v.id for v in self.list_versions(project_id)
        }:
            raise ParentNotFoundError(parent_id)

        df = df.copy()
        if PROVENANCE_COLUMN in df.columns:
            existing = [Provenance.parse(v) for v in df[PROVENANCE_COLUMN]]
            df = df.drop(columns=[PROVENANCE_COLUMN])
        else:
            existing = None

        if existing is not None and provenance is None:
            rows_provenance = existing
        elif provenance is not None:
            if len(provenance) != len(df):
                raise ValueError("provenance list length must match row count")
            if existing is not None:
                # derived rows keep the parent's record unless the new one says
                # otherwise; label_origins merge rather than replace.
                rows_provenance = []
                for old, new in zip(existing, provenance):
                    merged = {**old, **new}
                    if "label_origins" in old or "label_origins" in new:
                        merged["label_origins"] = {
                            **old.get("label_origins", {}),
                            **new.get("label_origins", {}),
                        }
                    rows_provenance.append(merged)
            else:
                rows_provenance = provenance
        else:
            rows_provenance = [
                Provenance.row(
                    origin, provider=(meta or {}).get("provider"),
                    model=(meta or {}).get("model"), mode=(meta or {}).get("mode"),
                    seed=seed,
                )
                for _ in range(len(df))
            ]

        df[PROVENANCE_COLUMN] = [json.dumps(p) for p in rows_provenance]

        version_id = uuid.uuid4().hex
        file_path = self.paths.version_file(version_id)
        df.to_parquet(file_path, engine="pyarrow", index=False)

        columns = infer_column_info(df.drop(columns=[PROVENANCE_COLUMN]))
        summary = provenance_breakdown(df[PROVENANCE_COLUMN].tolist())
        with self.db.connect() as conn:
            (number,) = conn.execute(
                "SELECT COALESCE(MAX(number), 0) + 1 FROM dataset_versions "
                "WHERE project_id = ?",
                (project_id,),
            ).fetchone()
            conn.execute(
                "INSERT INTO dataset_versions (id, project_id, parent_id, number, "
                "origin, row_count, columns_json, provenance_summary_json, "
                "seed_json, meta_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    version_id,
                    project_id,
                    parent_id,
                    number,
                    origin,
                    len(df),
                    json.dumps([c.to_dict() for c in columns]),
                    json.dumps(summary),
                    json.dumps(seed),
                    json.dumps(meta or {}),
                    _now(),
                ),
            )
        return DatasetVersion(
            id=version_id,
            project_id=project_id,
            parent_id=parent_id,
            number=int(number),
            origin=origin,
            row_count=len(df),
            columns=columns,
            provenance_summary=summary,
            seed=seed,
            meta=meta or {},
            created_at=_now(),
        )

    def list_versions(self, project_id: str) -> list[DatasetVersion]:
        self.get_project(project_id)
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM dataset_versions WHERE project_id = ? "
                "ORDER BY created_at, number",
                (project_id,),
            ).fetchall()
        return [self._version_from_row(row) for row in rows]

    def get_version(self, version_id: str) -> DatasetVersion:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM dataset_versions WHERE id = ?", (version_id,)
            ).fetchone()
        if row is None:
            raise DatasetVersionNotFoundError(version_id)
        return self._version_from_row(row)

    def load_dataframe(
        self, version_id: str, *, include_provenance: bool = False
    ) -> pd.DataFrame:
        version = self.get_version(version_id)  # validates id
        df = pd.read_parquet(self.paths.version_file(version.id), engine="pyarrow")
        if not include_provenance:
            df = df.drop(columns=[PROVENANCE_COLUMN], errors="ignore")
        return df

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _version_from_row(row: Any) -> DatasetVersion:
        return DatasetVersion(
            id=row["id"],
            project_id=row["project_id"],
            parent_id=row["parent_id"],
            number=int(row["number"]),
            origin=row["origin"],
            row_count=int(row["row_count"]),
            columns=[ColumnInfo(**c) for c in json.loads(row["columns_json"])],
            provenance_summary=json.loads(row["provenance_summary_json"]),
            seed=json.loads(row["seed_json"]) if row["seed_json"] else None,
            meta=json.loads(row["meta_json"] or "{}"),
            created_at=row["created_at"],
        )
