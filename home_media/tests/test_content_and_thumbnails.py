from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from home_media.content import UnsafeMediaPath, resolve_media_path
from home_media.index import MediaIndex
from home_media.scanner import MediaScanner
from home_media.thumbnails import ThumbnailService

from conftest import make_jpeg, make_video


@pytest.mark.parametrize("relative", ["../secret.jpg", "/etc/passwd", "%2e%2e/secret.jpg"])
def test_resolver_rejects_traversal_and_absolute_paths(settings, relative):
    with pytest.raises(UnsafeMediaPath):
        resolve_media_path({"source": "photo", "relative_path": relative}, settings)


def test_resolver_rejects_symlink_escape_after_indexing(settings, tmp_path: Path):
    path = settings.photo_root / "safe.jpg"
    make_jpeg(path)
    item = {"source": "photo", "relative_path": "safe.jpg"}
    assert resolve_media_path(item, settings) == path.resolve()
    path.unlink()
    outside = tmp_path / "secret.jpg"
    make_jpeg(outside)
    path.symlink_to(outside)
    with pytest.raises(UnsafeMediaPath):
        resolve_media_path(item, settings)


def test_photo_thumbnail_is_oriented_and_cache_is_reused(settings):
    path = settings.photo_root / "rotated.jpg"
    make_jpeg(path, size=(20, 40), orientation=6)
    settings.prepare()
    index = MediaIndex(settings.database_path)
    index.initialize()
    MediaScanner(settings, index).refresh()
    item = index.list_items(media_type="photo", limit=10, offset=0)[0][0]
    service = ThumbnailService(settings, index)

    first = service.get_or_create(item)
    first_mtime = first.stat().st_mtime_ns
    with Image.open(first) as thumb:
        assert thumb.width > thumb.height
    second = service.get_or_create(item)
    assert second == first
    assert second.stat().st_mtime_ns == first_mtime


def test_video_poster_is_generated(settings):
    path = settings.video_root / "poster.mp4"
    make_video(path)
    settings.prepare()
    index = MediaIndex(settings.database_path)
    index.initialize()
    MediaScanner(settings, index).refresh()
    item = index.list_items(media_type="video", limit=10, offset=0)[0][0]
    poster = ThumbnailService(settings, index).get_or_create(item)
    with Image.open(poster) as image:
        assert image.width == 64
        assert image.height < image.width

