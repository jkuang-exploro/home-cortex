"""Photo and video metadata extraction using established media libraries."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # Pillow can still handle formats supported by its base build.
    pass


@dataclass(frozen=True, slots=True)
class Metadata:
    captured_at: datetime
    width: int | None
    height: int | None
    duration_seconds: float | None = None
    orientation: int | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    video_codec: str | None = None
    status: str = "ok"


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = value.strip().replace("Z", "+00:00")
    for parser in (
        lambda text: datetime.fromisoformat(text),
        lambda text: datetime.strptime(text, "%Y:%m:%d %H:%M:%S"),
        lambda text: datetime.strptime(text, "%Y-%m-%d %H:%M:%S"),
    ):
        try:
            parsed = parser(cleaned)
            if parsed.tzinfo is None:
                parsed = parsed.astimezone()
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def photo_metadata(path: Path) -> Metadata:
    """Read EXIF original/digitized/general timestamps, then fall back to mtime."""
    with Image.open(path) as image:
        exif = image.getexif()
        captured = None
        for tag in (
            ExifTags.Base.DateTimeOriginal,
            ExifTags.Base.DateTimeDigitized,
            ExifTags.Base.DateTime,
        ):
            captured = _parse_datetime(exif.get(tag))
            if captured is not None:
                break
        orientation = _integer(exif.get(ExifTags.Base.Orientation))
        make = _text(exif.get(ExifTags.Base.Make))
        model = _text(exif.get(ExifTags.Base.Model))
        return Metadata(
            captured_at=captured or _mtime(path),
            width=image.width,
            height=image.height,
            orientation=orientation,
            camera_make=make,
            camera_model=model,
        )


def video_metadata(path: Path, ffprobe_path: str) -> Metadata:
    completed = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    stream = next((value for value in streams if value.get("codec_type") == "video"), None)
    if stream is None:
        raise ValueError("No video stream found")
    format_data = payload.get("format") or {}
    stream_tags = stream.get("tags") or {}
    format_tags = format_data.get("tags") or {}
    captured = _parse_datetime(stream_tags.get("creation_time"))
    captured = captured or _parse_datetime(format_tags.get("creation_time"))
    duration = _float(stream.get("duration")) or _float(format_data.get("duration"))
    rotation = _rotation(stream)
    width = _integer(stream.get("width"))
    height = _integer(stream.get("height"))
    if rotation in {90, 270}:
        width, height = height, width
    return Metadata(
        captured_at=captured or _mtime(path),
        width=width,
        height=height,
        duration_seconds=duration,
        orientation=rotation,
        video_codec=_text(stream.get("codec_name")),
    )


def failed_metadata(path: Path, error: Exception) -> Metadata:
    return Metadata(
        captured_at=_mtime(path),
        width=None,
        height=None,
        status=f"error:{type(error).__name__}",
    )


def probe(path: Path, media_type: str, ffprobe_path: str) -> Metadata:
    try:
        if media_type == "photo":
            return photo_metadata(path)
        return video_metadata(path, ffprobe_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        return failed_metadata(path, error)


def _rotation(stream: dict[str, Any]) -> int | None:
    tags = stream.get("tags") or {}
    value = _integer(tags.get("rotate"))
    if value is None:
        for side_data in stream.get("side_data_list") or []:
            value = _integer(side_data.get("rotation"))
            if value is not None:
                break
    return abs(value) % 360 if value is not None else None


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None

