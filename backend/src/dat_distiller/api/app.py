"""The FastAPI app behind `dat-distiller serve`."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI

from .. import __version__
from ..extras import installed_extras
from .frontend import mount_frontend

api_router = APIRouter(prefix="/api")


@api_router.get("/health", tags=["system"])
async def health() -> dict[str, Any]:
    """Whether the app is up, and which optional extras this install can use."""
    return {
        "status": "ok",
        "version": __version__,
        "extras": installed_extras(),
    }


def create_app() -> FastAPI:
    """Build the API app.

    Later issues include their routers here, always before `mount_frontend`:
    the mount catch-alls everything the routes above it did not match.
    """
    app = FastAPI(title="Dat Distiller", version=__version__)
    app.include_router(api_router)
    mount_frontend(app)
    return app
