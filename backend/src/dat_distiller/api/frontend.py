"""Serving the built frontend (`frontend/dist`) from the same port as the API."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

#: Set this to serve a build from somewhere else (the tests use a temp build).
FRONTEND_DIST_ENV = "DAT_DISTILLER_FRONTEND_DIST"

PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # backend/src/dat_distiller
REPO_ROOT = PACKAGE_ROOT.parents[2]  # repo root
DEFAULT_FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


class FrontendFiles(StaticFiles):
    """Static files of a frontend build, with SPA fallback.

    A path that has no matching file is a client-side route, so it gets
    `index.html`. Paths under `/api` are the exception: they never matched a
    route, so they stay JSON 404s instead of pretending to be a page.
    """

    def __init__(self, directory: Path) -> None:
        super().__init__(directory=str(directory), html=True)
        self.index_path = directory / "index.html"

    async def get_response(self, path: str, scope: Scope) -> Response:
        if self._is_api_path(path, scope):
            raise HTTPException(status_code=404)
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:  # Starlette's own, which FastAPI subclasses
            if exc.status_code == 404 and scope.get("method", "GET") in ("GET", "HEAD"):
                shell = await self._index_response(scope)
                if shell is not None:
                    return shell
            raise

    async def _index_response(self, scope: Scope) -> Response | None:
        """The SPA shell, or None when the build has no usable index.html."""
        if not self.index_path.is_file():
            return None
        try:
            return await super().get_response("index.html", scope)
        except HTTPException:
            return None

    @staticmethod
    def _is_api_path(path: str, scope: Scope) -> bool:
        # Starlette hands a mounted app the path both ways (as an argument and
        # in the scope), and the leading slash depends on the version.
        for candidate in (path, str(scope.get("path", ""))):
            normalised = "/" + candidate.lstrip("/")
            if normalised == "/api" or normalised.startswith("/api/"):
                return True
        return False


def frontend_dist() -> Path | None:
    """The built frontend to serve, or None when there is no build to serve."""
    override = os.environ.get(FRONTEND_DIST_ENV, "")
    dist = Path(override).expanduser() if override else DEFAULT_FRONTEND_DIST
    return dist if (dist / "index.html").is_file() else None


def mount_frontend(app: FastAPI) -> None:
    """Serve the built frontend from `app`'s root, if the build exists.

    Call this last: the mount catches everything the routes above it did not
    match, which is how unknown `/api` paths end up as JSON 404s.
    """
    dist = frontend_dist()
    if dist is not None:
        app.mount("/", FrontendFiles(dist), name="frontend")
    return True
