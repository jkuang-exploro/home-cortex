"""Rebuildable SQLite index for derived media state."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .metadata import Metadata


SCHEMA = """
CREATE TABLE IF NOT EXISTS media_items (
    id TEXT PRIMARY KEY,
    media_type TEXT NOT NULL CHECK (media_type IN ('photo', 'video')),
    source TEXT NOT NULL CHECK (source IN ('photo', 'video')),
    relative_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    captured_at TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    duration_seconds REAL,
    orientation INTEGER,
    camera_make TEXT,
    camera_model TEXT,
    video_codec TEXT,
    thumbnail_status TEXT NOT NULL DEFAULT 'missing',
    metadata_status TEXT NOT NULL,
    UNIQUE(source, relative_path)
);
CREATE INDEX IF NOT EXISTS idx_media_timeline
ON media_items(captured_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_media_type_timeline
ON media_items(media_type, captured_at DESC, id DESC);
"""


class MediaIndex:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def records_by_identity(self) -> dict[tuple[str, str], dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM media_items").fetchall()
        return {(row["source"], row["relative_path"]): dict(row) for row in rows}

    def upsert(
        self,
        *,
        media_id: str,
        media_type: str,
        source: str,
        relative_path: str,
        filename: str,
        file_size: int,
        mtime_ns: int,
        metadata: Metadata,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO media_items (
                    id, media_type, source, relative_path, filename, file_size,
                    mtime_ns, captured_at, width, height, duration_seconds,
                    orientation, camera_make, camera_model, video_codec,
                    thumbnail_status, metadata_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'missing', ?)
                ON CONFLICT(id) DO UPDATE SET
                    media_type=excluded.media_type,
                    source=excluded.source,
                    relative_path=excluded.relative_path,
                    filename=excluded.filename,
                    file_size=excluded.file_size,
                    mtime_ns=excluded.mtime_ns,
                    captured_at=excluded.captured_at,
                    width=excluded.width,
                    height=excluded.height,
                    duration_seconds=excluded.duration_seconds,
                    orientation=excluded.orientation,
                    camera_make=excluded.camera_make,
                    camera_model=excluded.camera_model,
                    video_codec=excluded.video_codec,
                    thumbnail_status='missing',
                    metadata_status=excluded.metadata_status
                """,
                (
                    media_id,
                    media_type,
                    source,
                    relative_path,
                    filename,
                    file_size,
                    mtime_ns,
                    metadata.captured_at.isoformat(),
                    metadata.width,
                    metadata.height,
                    metadata.duration_seconds,
                    metadata.orientation,
                    metadata.camera_make,
                    metadata.camera_model,
                    metadata.video_codec,
                    metadata.status,
                ),
            )

    def remove_missing(self, present: Iterable[tuple[str, str]]) -> list[str]:
        keep = set(present)
        records = self.records_by_identity()
        removed_ids = [row["id"] for key, row in records.items() if key not in keep]
        if not removed_ids:
            return []
        placeholders = ",".join("?" for _ in removed_ids)
        with self._connect() as connection:
            connection.execute(
                f"DELETE FROM media_items WHERE id IN ({placeholders})",  # noqa: S608
                removed_ids,
            )
        return removed_ids

    def get(self, media_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM media_items WHERE id = ?", (media_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_items(
        self,
        *,
        media_type: str,
        limit: int,
        offset: int,
    ) -> tuple[list[dict[str, Any]], int]:
        where = "" if media_type == "all" else "WHERE media_type = ?"
        parameters: list[Any] = [] if media_type == "all" else [media_type]
        with self._connect() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM media_items {where}",  # noqa: S608
                parameters,
            ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT * FROM media_items {where}
                ORDER BY captured_at DESC, id DESC LIMIT ? OFFSET ?
                """,  # noqa: S608
                [*parameters, limit, offset],
            ).fetchall()
        return [dict(row) for row in rows], total

    def set_thumbnail_status(self, media_id: str, status: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE media_items SET thumbnail_status = ? WHERE id = ?",
                (status, media_id),
            )

    def stats(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(media_type = 'photo') AS photos,
                       SUM(media_type = 'video') AS videos,
                       COALESCE(SUM(file_size), 0) AS total_bytes
                FROM media_items
                """
            ).fetchone()
        return {key: int(row[key] or 0) for key in row.keys()}

