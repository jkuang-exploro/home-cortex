"""Health and maintenance routes."""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from ...config import get_settings
from ..dependencies import authenticate_request
from ..errors import APIError, request_id
from ..schemas import ExportRequest


logger = logging.getLogger("uvicorn.error.home_cortex.api")
router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    try:
        surreal_version = await request.app.state.database.version()
        return {"status": "ok", "surrealdb": surreal_version}
    except Exception as error:
        logger.warning(
            "health_check_failed request_id=%s dependency=surrealdb "
            "exception_type=%s",
            request_id(request),
            type(error).__name__,
        )
        raise APIError(
            503,
            "database_unavailable",
            "SurrealDB health check failed",
        ) from error


async def run_ingest(database: Any, data_dir: Path, edge_registry: Any):
    from ...persistence.ingestion import ingest_directory

    return await ingest_directory(database, data_dir, edge_registry)


async def run_export(database: Any, target_dir: Path, edge_registry: Any):
    from ...persistence.export import export_directory

    return await export_directory(database, target_dir, edge_registry)


@router.post("/admin/ingest")
async def ingest(request: Request) -> dict[str, Any]:
    authenticate_request(request)
    settings = get_settings()
    try:
        result = await run_ingest(
            request.app.state.database,
            settings.data_dir,
            getattr(request.app.state, "edge_registry", None),
        )
        return {"status": "ok", **asdict(result)}
    except (FileNotFoundError, ValueError) as error:
        raise APIError(400, "ingestion_failed", str(error)) from error


@router.post("/admin/export")
async def export(body: ExportRequest, request: Request) -> dict[str, Any]:
    authenticate_request(request)
    if not body.target_dir.is_absolute():
        raise APIError(
            400,
            "export_failed",
            "target_dir must be an absolute path on the API server. "
            "From Docker Compose use /app/export (host directory tmp/db-export).",
        )
    try:
        result = await run_export(
            request.app.state.database,
            body.target_dir,
            getattr(request.app.state, "edge_registry", None),
        )
        return {"status": "ok", **asdict(result)}
    except (FileNotFoundError, ValueError, OSError) as error:
        raise APIError(400, "export_failed", str(error)) from error
