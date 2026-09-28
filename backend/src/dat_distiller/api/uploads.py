"""Upload Dataset Versions (CSV / Parquet / JSONL) and download them back."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Form, HTTPException, Query, Response, UploadFile
from fastapi.requests import Request

from ..ingest import (
    SUPPORTED_SUFFIXES,
    MalformedFileError,
    UnsupportedFormatError,
    read_dataframe,
    to_csv_bytes,
    to_parquet_bytes,
)
from ..pii import register_pii_checks, scan, summarize_findings
from ..store.provenance import PROVENANCE_COLUMN
from ..store.store import ParentNotFoundError
from .projects import _require_project, _require_version

router = APIRouter(tags=["upload"])


@router.post("/projects/{project_id}/upload", status_code=201)
async def upload_dataset_version(
    project_id: str,
    request: Request,
    file: UploadFile,
    parent_id: str | None = Form(None),
) -> dict[str, Any]:
    """Create a new Dataset Version with `uploaded` Provenance from a file."""
    _require_project(request, project_id)
    store = request.app.state.store
    content = await file.read()
    if not content:
        raise HTTPException(422, "uploaded file is empty")
    try:
        df = read_dataframe(file.filename or "upload.csv", content)
    except UnsupportedFormatError as exc:
        raise HTTPException(415, str(exc)) from exc
    except MalformedFileError as exc:
        raise HTTPException(422, str(exc)) from exc
    if len(df.columns) == 0:
        raise HTTPException(422, "uploaded file has no columns")
    try:
        version = store.create_version(project_id, df, parent_id=parent_id)
    except ParentNotFoundError as exc:
        raise HTTPException(404, "parent dataset version not found") from exc
    # Scan before the user has done anything with the Version, so the Checks are
    # already waiting at the end of the step. Findings only raise Checks; the
    # user still chooses warn / mask / drop (see `api/pii.py`).
    findings = scan(df)
    checks = register_pii_checks(request.app.state.checks, version.id, findings)
    payload = version.to_dict()
    payload["pii"] = summarize_findings(findings)
    payload["pii_checks"] = [check.id for check in checks]
    return payload


@router.get("/dataset-versions/{version_id}/download")
def download_dataset_version(
    version_id: str,
    request: Request,
    format: str = Query("csv", pattern="^(csv|parquet)$"),
    include_provenance: bool = Query(False),
) -> Response:
    """CSV by default, Parquet when requested; Provenance only on request."""
    version = _require_version(request, version_id)
    store = request.app.state.store
    df = store.load_dataframe(version.id, include_provenance=include_provenance)
    if format == "parquet":
        if not include_provenance and PROVENANCE_COLUMN in df.columns:
            df = df.drop(columns=[PROVENANCE_COLUMN])
        body, media, suffix = (
            to_parquet_bytes(df),
            "application/vnd.apache.parquet",
            "parquet",
        )
    else:
        body, media, suffix = to_csv_bytes(df), "text/csv", "csv"
    filename = f"dataset-v{version.number}.{suffix}"
    return Response(
        content=body,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


__all__ = ["SUPPORTED_SUFFIXES", "router"]
