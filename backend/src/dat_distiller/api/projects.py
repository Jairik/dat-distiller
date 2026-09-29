"""REST endpoints for Projects and Dataset Versions.

Dataset Versions are returned as a FLAT list ordered by creation; each item
carries ``parent_id`` (null for a root version), so the frontend renders the
version tree from that. There are no endpoints that change a version: every
change creates a child.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..store import DatasetStore, DatasetVersion
from ..store.store import (
    DatasetVersionNotFoundError,
    DuplicateProjectError,
    ParentNotFoundError,
    ProjectNotFoundError,
)

router = APIRouter(tags=["projects"])


def _store(request: Request) -> DatasetStore:
    return request.app.state.store


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class ProjectOut(BaseModel):
    id: str
    name: str
    created_at: str
    version_count: int
    latest_version_id: str | None


class DatasetVersionIn(BaseModel):
    """Minimal JSON rows creation; rich upload formats arrive in #5."""

    columns: list[str] = Field(min_length=1)
    rows: list[list[Any]] = []
    parent_id: str | None = None


class PreviewOut(BaseModel):
    version_id: str
    columns: list[dict[str, Any]]
    page: int
    page_size: int
    total_rows: int
    rows: list[list[Any]]


def _project_out(store: DatasetStore, project_id: str) -> ProjectOut:
    project = store.get_project(project_id)
    versions = store.list_versions(project_id)
    return ProjectOut(
        id=project.id,
        name=project.name,
        created_at=project.created_at,
        version_count=len(versions),
        latest_version_id=versions[-1].id if versions else None,
    )


@router.post("/projects", response_model=ProjectOut, status_code=201)
def create_project(body: ProjectIn, request: Request) -> ProjectOut:
    try:
        project = _store(request).create_project(body.name)
    except DuplicateProjectError as exc:
        raise HTTPException(409, f"a project named {exc.args[0]!r} already exists") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _project_out(_store(request), project.id)


@router.get("/projects", response_model=list[ProjectOut])
def list_projects(request: Request) -> list[ProjectOut]:
    store = _store(request)
    return [_project_out(store, p.id) for p in store.list_projects()]


@router.get("/projects/{project_id}", response_model=ProjectOut)
def get_project(project_id: str, request: Request) -> ProjectOut:
    _require_project(request, project_id)
    return _project_out(_store(request), project_id)


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: str, request: Request) -> None:
    _require_project(request, project_id)
    _store(request).delete_project(project_id)


@router.get("/projects/{project_id}/dataset_versions")
def list_dataset_versions(project_id: str, request: Request) -> list[dict[str, Any]]:
    _require_project(request, project_id)
    return [v.to_dict() for v in _store(request).list_versions(project_id)]


@router.post(
    "/projects/{project_id}/dataset_versions", status_code=201, response_model=None
)
def create_dataset_version(
    project_id: str, body: DatasetVersionIn, request: Request
) -> dict[str, Any]:
    _require_project(request, project_id)
    store = _store(request)
    try:
        df = pd.DataFrame(body.rows, columns=body.columns)
    except ValueError as exc:
        raise HTTPException(422, f"rows do not match columns: {exc}") from exc
    try:
        version = store.create_version(project_id, df, parent_id=body.parent_id)
    except ParentNotFoundError as exc:
        raise HTTPException(404, f"parent dataset version {exc.args[0]!r} not found") from exc
    return version.to_dict()


@router.get("/dataset-versions/{version_id}")
def get_dataset_version(version_id: str, request: Request) -> dict[str, Any]:
    return _require_version(request, version_id).to_dict()


@router.get("/dataset-versions/{version_id}/preview", response_model=PreviewOut)
def preview_dataset_version(
    version_id: str,
    request: Request,
    page: int = Query(0, ge=0),
    page_size: int = Query(200, ge=1, le=1000),
    row: int | None = Query(
        None,
        ge=0,
        description=(
            "One row by position, instead of a page. The Review Queue needs the "
            "State of the row it is showing, and paging to reach one row means "
            "either fetching up to 1000 rows or guessing the page — both of "
            "which quietly return a different row than the one asked for."
        ),
    ),
) -> PreviewOut:
    version = _require_version(request, version_id)
    df = _store(request).load_dataframe(version.id)  # provenance excluded
    if row is not None:
        if row >= len(df):
            raise HTTPException(
                422, f"row {row} is past the end of this version ({len(df)} rows)"
            )
        start, end = row, row + 1
        effective_page, effective_size = 0, 1
    else:
        start, end = page * page_size, (page + 1) * page_size
        effective_page, effective_size = page, page_size
    rows = [[_jsonify(v) for v in row_values] for row_values in df.iloc[start:end].to_numpy(dtype=object)]
    return PreviewOut(
        version_id=version.id,
        columns=[c.to_dict() for c in version.columns],
        page=effective_page,
        page_size=effective_size,
        total_rows=len(df),
        rows=rows,
    )


def _jsonify(value: Any) -> Any:
    """numpy/pandas scalars -> JSON-native values."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, datetime):
            return value.isoformat()
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _require_project(request: Request, project_id: str) -> None:
    try:
        _store(request).get_project(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(404, "project not found") from exc


def _require_version(request: Request, version_id: str) -> DatasetVersion:
    try:
        return _store(request).get_version(version_id)
    except DatasetVersionNotFoundError as exc:
        raise HTTPException(404, "dataset version not found") from exc
