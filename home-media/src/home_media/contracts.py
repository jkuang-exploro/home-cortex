"""Public API contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


MediaType = Literal["photo", "video"]


class MediaItem(BaseModel):
    id: str
    type: MediaType
    filename: str
    relative_path: str
    file_size: int
    captured_at: datetime
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    orientation: int | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    video_codec: str | None = None
    metadata_status: str
    thumbnail_status: str
    thumbnail_url: str
    content_url: str


class MediaPage(BaseModel):
    items: list[MediaItem]
    total: int
    limit: int
    offset: int
    next_offset: int | None


class RefreshResult(BaseModel):
    discovered: int
    added: int
    updated: int
    unchanged: int
    removed: int
    unsupported: int
    errors: int
    extension_counts: dict[str, int]
    unsupported_extension_counts: dict[str, int]
    duration_seconds: float

