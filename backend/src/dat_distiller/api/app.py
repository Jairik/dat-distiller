"""The FastAPI app behind `dat-distiller serve`."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI

from .. import __version__
from ..extras import installed_extras
from ..store import DatasetStore
from ..store.paths import AppPaths
from .frontend import mount_frontend
from .projects import router as projects_router

api_router = APIRouter(prefix="/api")


@api_router.get("/health", tags=["system"])
async def health() -> dict[str, Any]:
    """Whether the app is up, and which optional extras this install can use."""
    return {
        "status": "ok",
        "version": __version__,
        "extras": installed_extras(),
    }


def create_app(store: DatasetStore | None = None) -> FastAPI:
    """Build the API app.

    Later issues include their routers here, always before `mount_frontend`:
    the mount catch-alls everything the routes above it did not match.
    """
    app = FastAPI(title="Dat Distiller", version=__version__)
    app.state.store = store or DatasetStore(AppPaths.from_env())
    app.include_router(api_router)
    app.include_router(projects_router, prefix="/api")
    mount_frontend(app)
    return app
