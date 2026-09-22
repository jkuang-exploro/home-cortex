"""ID-only, containment-checked original media resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Settings


class UnsafeMediaPath(FileNotFoundError):
    pass


def resolve_media_path(item: dict[str, Any], settings: Settings) -> Path:
    roots = {"photo": settings.photo_root, "video": settings.video_root}
    root = roots.get(item.get("source"))
    relative = item.get("relative_path")
    if root is None or not isinstance(relative, str):
        raise UnsafeMediaPath("Unknown media source")
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise UnsafeMediaPath("Invalid media path")
    try:
        canonical_root = root.resolve(strict=True)
        resolved = (root / candidate_relative).resolve(strict=True)
    except OSError as error:
        raise UnsafeMediaPath("Media file is unavailable") from error
    if not resolved.is_file() or not resolved.is_relative_to(canonical_root):
        raise UnsafeMediaPath("Media path escaped its configured root")
    return resolved

