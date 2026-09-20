"""Environment-backed service configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    photo_root: Path
    video_root: Path
    data_root: Path
    thumbnail_size: int = 500
    page_size: int = 100
    ffprobe_path: str = "ffprobe"
    ffmpeg_path: str = "ffmpeg"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            photo_root=Path(os.environ.get("HOME_MEDIA_PHOTO_ROOT", "/media/photo")),
            video_root=Path(os.environ.get("HOME_MEDIA_VIDEO_ROOT", "/media/video")),
            data_root=Path(os.environ.get("HOME_MEDIA_DATA_ROOT", "/data")),
            thumbnail_size=int(os.environ.get("HOME_MEDIA_THUMBNAIL_SIZE", "500")),
            page_size=int(os.environ.get("HOME_MEDIA_PAGE_SIZE", "100")),
            ffprobe_path=os.environ.get("HOME_MEDIA_FFPROBE_PATH", "ffprobe"),
            ffmpeg_path=os.environ.get("HOME_MEDIA_FFMPEG_PATH", "ffmpeg"),
        )

    @property
    def database_path(self) -> Path:
        return self.data_root / "media.db"

    @property
    def thumbnail_root(self) -> Path:
        return self.data_root / "thumbnails"

    def prepare(self) -> None:
        if not 64 <= self.thumbnail_size <= 2000:
            raise ValueError("HOME_MEDIA_THUMBNAIL_SIZE must be between 64 and 2000")
        if not 1 <= self.page_size <= 500:
            raise ValueError("HOME_MEDIA_PAGE_SIZE must be between 1 and 500")
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.thumbnail_root.mkdir(parents=True, exist_ok=True)

