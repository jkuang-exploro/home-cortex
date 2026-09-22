"""Lazy, concurrency-safe JPEG thumbnail and video poster cache."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .config import Settings
from .content import resolve_media_path
from .index import MediaIndex


class ThumbnailError(RuntimeError):
    pass


class ThumbnailService:
    def __init__(self, settings: Settings, index: MediaIndex) -> None:
        self.settings = settings
        self.index = index
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def get_or_create(self, item: dict[str, Any]) -> Path:
        media_id = item["id"]
        target = self.settings.thumbnail_root / f"{media_id}.jpg"
        if target.is_file():
            return target
        with self._locks_guard:
            lock = self._locks.setdefault(media_id, threading.Lock())
        with lock:
            if target.is_file():
                return target
            temporary = target.with_suffix(".tmp")
            temporary.unlink(missing_ok=True)
            try:
                source = resolve_media_path(item, self.settings)
                if item["media_type"] == "photo":
                    self._photo(source, temporary)
                else:
                    self._video(source, temporary, item.get("duration_seconds"))
                temporary.replace(target)
                self.index.set_thumbnail_status(media_id, "ready")
                return target
            except Exception as error:
                temporary.unlink(missing_ok=True)
                self.index.set_thumbnail_status(media_id, f"error:{type(error).__name__}")
                raise ThumbnailError("Thumbnail generation failed") from error
            finally:
                with self._locks_guard:
                    self._locks.pop(media_id, None)

    def _photo(self, source: Path, target: Path) -> None:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image)
            image.thumbnail((self.settings.thumbnail_size, self.settings.thumbnail_size))
            if image.mode not in {"RGB", "L"}:
                background = Image.new("RGB", image.size, "white")
                if "A" in image.getbands():
                    background.paste(image, mask=image.getchannel("A"))
                else:
                    background.paste(image)
                image = background
            image.convert("RGB").save(target, format="JPEG", quality=84, optimize=True)

    def _video(self, source: Path, target: Path, duration: float | None) -> None:
        seek = min(max((duration or 0) * 0.1, 0), 5)
        completed = subprocess.run(
            [
                self.settings.ffmpeg_path,
                "-v",
                "error",
                "-ss",
                f"{seek:.3f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                f"scale={self.settings.thumbnail_size}:{self.settings.thumbnail_size}:force_original_aspect_ratio=decrease",
                "-f",
                "image2",
                "-c:v",
                "mjpeg",
                str(target),
            ],
            check=False,
            capture_output=True,
            timeout=90,
        )
        if completed.returncode or not target.is_file():
            raise ThumbnailError(completed.stderr.decode(errors="replace")[-500:])

