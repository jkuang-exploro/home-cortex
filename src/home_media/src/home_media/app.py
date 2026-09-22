"""FastAPI surface for the standalone read-only media service."""

from __future__ import annotations

import mimetypes
import re
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .config import Settings
from .content import UnsafeMediaPath, resolve_media_path
from .contracts import MediaItem, MediaPage, RefreshResult
from .index import MediaIndex
from .scanner import MediaScanner, RefreshInProgress
from .thumbnails import ThumbnailError, ThumbnailService


MEDIA_ID = re.compile(r"^media_[0-9a-f]{24}$")


def create_app(config: Settings | None = None) -> FastAPI:
    settings = config or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        settings.prepare()
        index = MediaIndex(settings.database_path)
        index.initialize()
        application.state.settings = settings
        application.state.index = index
        application.state.scanner = MediaScanner(settings, index)
        application.state.thumbnails = ThumbnailService(settings, index)
        yield

    application = FastAPI(
        title="Home Media API",
        version=__version__,
        description="Deterministic read-only household media browsing.",
        lifespan=lifespan,
    )

    @application.exception_handler(UnsafeMediaPath)
    def unavailable_handler(_request: Request, _error: UnsafeMediaPath) -> JSONResponse:
        return _error_response(404, "media_unavailable", "Media file is unavailable")

    @application.get("/health")
    def health(request: Request) -> dict[str, Any]:
        index = _index(request)
        return {
            "status": "ok",
            "version": __version__,
            "index": index.path.name,
            "refreshing": _scanner(request).refreshing,
        }

    @application.get("/stats")
    def stats(request: Request) -> dict[str, Any]:
        payload = _index(request).stats()
        payload["refreshing"] = _scanner(request).refreshing
        return payload

    @application.post("/refresh", response_model=RefreshResult)
    def refresh(request: Request) -> RefreshResult | JSONResponse:
        try:
            return _scanner(request).refresh()
        except RefreshInProgress:
            return _error_response(409, "refresh_in_progress", "A media refresh is already running")

    @application.get("/items", response_model=MediaPage)
    def list_items(
        request: Request,
        media_type: str = Query("all", alias="type", pattern="^(all|photo|video)$"),
        limit: int | None = Query(None, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> MediaPage:
        page_limit = limit or settings.page_size
        rows, total = _index(request).list_items(
            media_type=media_type,
            limit=page_limit,
            offset=offset,
        )
        next_offset = offset + len(rows) if offset + len(rows) < total else None
        return MediaPage(
            items=[_public_item(row) for row in rows],
            total=total,
            limit=page_limit,
            offset=offset,
            next_offset=next_offset,
        )

    @application.get("/items/{media_id}", response_model=MediaItem)
    def get_item(media_id: str, request: Request) -> MediaItem | JSONResponse:
        item = _find_item(request, media_id)
        if isinstance(item, JSONResponse):
            return item
        return _public_item(item)

    @application.get("/items/{media_id}/thumbnail", response_model=None)
    def thumbnail(media_id: str, request: Request) -> FileResponse | JSONResponse:
        item = _find_item(request, media_id)
        if isinstance(item, JSONResponse):
            return item
        try:
            path = _thumbnails(request).get_or_create(item)
        except ThumbnailError:
            return _error_response(422, "thumbnail_failed", "Thumbnail generation failed")
        return FileResponse(
            path,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=86400"},
        )

    @application.get("/items/{media_id}/content", response_model=None)
    def content(media_id: str, request: Request) -> FileResponse | JSONResponse:
        item = _find_item(request, media_id)
        if isinstance(item, JSONResponse):
            return item
        path = resolve_media_path(item, settings)
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(
            path,
            media_type=media_type,
            filename=path.name,
            content_disposition_type="inline",
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "private, max-age=3600",
            },
        )

    return application


def _index(request: Request) -> MediaIndex:
    return request.app.state.index


def _scanner(request: Request) -> MediaScanner:
    return request.app.state.scanner


def _thumbnails(request: Request) -> ThumbnailService:
    return request.app.state.thumbnails


def _find_item(request: Request, media_id: str) -> dict[str, Any] | JSONResponse:
    if not MEDIA_ID.fullmatch(media_id):
        return _error_response(404, "media_not_found", "Media item was not found")
    item = _index(request).get(media_id)
    if item is None:
        return _error_response(404, "media_not_found", "Media item was not found")
    return item


def _public_item(row: dict[str, Any]) -> MediaItem:
    media_id = row["id"]
    return MediaItem(
        id=media_id,
        type=row["media_type"],
        filename=row["filename"],
        relative_path=row["relative_path"],
        file_size=row["file_size"],
        captured_at=row["captured_at"],
        width=row["width"],
        height=row["height"],
        duration_seconds=row["duration_seconds"],
        orientation=row["orientation"],
        camera_make=row["camera_make"],
        camera_model=row["camera_model"],
        video_codec=row["video_codec"],
        metadata_status=row["metadata_status"],
        thumbnail_status=row["thumbnail_status"],
        thumbnail_url=f"/media-api/items/{media_id}/thumbnail",
        content_url=f"/media-api/items/{media_id}/content",
    )


def _error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


app = create_app()
