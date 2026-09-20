from __future__ import annotations

import os
from pathlib import Path

from home_media.index import MediaIndex
from home_media.scanner import MediaScanner

from conftest import make_jpeg, make_video


def test_refresh_is_recursive_incremental_and_reconciles_deletions(settings, monkeypatch):
    nested = settings.photo_root / "year" / "image.jpg"
    video = settings.video_root / "clip.mp4"
    ignored = settings.photo_root / "notes.txt"
    make_jpeg(nested, captured_at="2024:05:06 07:08:09")
    make_video(video)
    ignored.write_text("not media")

    index = MediaIndex(settings.database_path)
    settings.prepare()
    index.initialize()
    scanner = MediaScanner(settings, index)

    first = scanner.refresh()
    assert (first.added, first.unchanged, first.unsupported) == (2, 0, 1)
    assert first.extension_counts == {".jpg": 1, ".mp4": 1, ".txt": 1}
    assert index.stats()["total"] == 2

    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("unchanged media must not be reprobed")

    monkeypatch.setattr("home_media.scanner.probe", unexpected_probe)
    second = scanner.refresh()
    assert (second.added, second.updated, second.unchanged) == (0, 0, 2)

    monkeypatch.undo()
    make_jpeg(nested, size=(60, 20), color="red")
    stat = nested.stat()
    os.utime(nested, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    third = scanner.refresh()
    assert (third.updated, third.unchanged) == (1, 1)

    video.unlink()
    fourth = scanner.refresh()
    assert fourth.removed == 1
    assert index.stats()["total"] == 1


def test_corrupt_media_is_indexed_as_a_controlled_metadata_error(settings):
    broken = settings.photo_root / "broken.jpg"
    broken.write_bytes(b"not a jpeg")
    settings.prepare()
    index = MediaIndex(settings.database_path)
    index.initialize()

    result = MediaScanner(settings, index).refresh()
    rows, _ = index.list_items(media_type="all", limit=10, offset=0)
    assert result.errors == 1
    assert rows[0]["metadata_status"].startswith("error:")


def test_symlink_escape_is_not_indexed(settings, tmp_path: Path):
    outside = tmp_path / "outside.jpg"
    make_jpeg(outside)
    (settings.photo_root / "escape.jpg").symlink_to(outside)
    settings.prepare()
    index = MediaIndex(settings.database_path)
    index.initialize()

    result = MediaScanner(settings, index).refresh()
    assert result.discovered == 0
    assert result.unsupported_extension_counts == {"<unsafe-symlink>": 1}
    assert index.stats()["total"] == 0

