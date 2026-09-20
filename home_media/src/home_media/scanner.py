"""Incremental recursive discovery for immutable media roots."""

from __future__ import annotations

import hashlib
import threading
import time
from collections import Counter
from pathlib import Path

from .config import Settings
from .contracts import RefreshResult
from .index import MediaIndex
from .metadata import probe


PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}


class RefreshInProgress(RuntimeError):
    pass


def stable_media_id(media_type: str, relative_path: str) -> str:
    digest = hashlib.sha256(f"{media_type}\0{relative_path}".encode()).hexdigest()[:24]
    return f"media_{digest}"


class MediaScanner:
    def __init__(self, settings: Settings, index: MediaIndex) -> None:
        self.settings = settings
        self.index = index
        self._lock = threading.Lock()
        self.refreshing = False

    def refresh(self) -> RefreshResult:
        if not self._lock.acquire(blocking=False):
            raise RefreshInProgress("A media refresh is already running")
        self.refreshing = True
        started = time.perf_counter()
        try:
            return self._refresh(started)
        finally:
            self.refreshing = False
            self._lock.release()

    def _refresh(self, started: float) -> RefreshResult:
        existing = self.index.records_by_identity()
        present: set[tuple[str, str]] = set()
        extension_counts: Counter[str] = Counter()
        unsupported_counts: Counter[str] = Counter()
        discovered = added = updated = unchanged = errors = 0

        for source, root, media_type, supported in (
            ("photo", self.settings.photo_root, "photo", PHOTO_EXTENSIONS),
            ("video", self.settings.video_root, "video", VIDEO_EXTENSIONS),
        ):
            if not root.is_dir():
                continue
            canonical_root = root.resolve()
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                extension = path.suffix.casefold() or "<none>"
                extension_counts[extension] += 1
                if extension not in supported:
                    unsupported_counts[extension] += 1
                    continue
                try:
                    resolved = path.resolve(strict=True)
                    if not resolved.is_relative_to(canonical_root):
                        unsupported_counts["<unsafe-symlink>"] += 1
                        continue
                    relative_path = path.relative_to(root).as_posix()
                    stat = resolved.stat()
                except (OSError, ValueError):
                    errors += 1
                    continue
                discovered += 1
                identity = (source, relative_path)
                present.add(identity)
                old = existing.get(identity)
                if (
                    old is not None
                    and old["file_size"] == stat.st_size
                    and old["mtime_ns"] == stat.st_mtime_ns
                ):
                    unchanged += 1
                    continue
                metadata = probe(resolved, media_type, self.settings.ffprobe_path)
                if metadata.status != "ok":
                    errors += 1
                media_id = stable_media_id(media_type, relative_path)
                self.index.upsert(
                    media_id=media_id,
                    media_type=media_type,
                    source=source,
                    relative_path=relative_path,
                    filename=path.name,
                    file_size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    metadata=metadata,
                )
                (self.settings.thumbnail_root / f"{media_id}.jpg").unlink(missing_ok=True)
                if old is None:
                    added += 1
                else:
                    updated += 1

        removed_ids = self.index.remove_missing(present)
        for media_id in removed_ids:
            (self.settings.thumbnail_root / f"{media_id}.jpg").unlink(missing_ok=True)
        return RefreshResult(
            discovered=discovered,
            added=added,
            updated=updated,
            unchanged=unchanged,
            removed=len(removed_ids),
            unsupported=sum(unsupported_counts.values()),
            errors=errors,
            extension_counts=dict(sorted(extension_counts.items())),
            unsupported_extension_counts=dict(sorted(unsupported_counts.items())),
            duration_seconds=round(time.perf_counter() - started, 6),
        )

