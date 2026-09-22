from __future__ import annotations

from datetime import datetime, timezone

from home_media.metadata import photo_metadata, video_metadata

from conftest import make_jpeg, make_video


def test_photo_dimensions_orientation_and_capture_time(settings):
    path = settings.photo_root / "rotated.jpg"
    make_jpeg(
        path,
        size=(20, 40),
        orientation=6,
        captured_at="2023:02:03 04:05:06",
    )
    metadata = photo_metadata(path)
    assert (metadata.width, metadata.height, metadata.orientation) == (20, 40, 6)
    assert metadata.captured_at.year == 2023


def test_photo_capture_time_falls_back_to_mtime(settings):
    path = settings.photo_root / "plain.jpg"
    make_jpeg(path)
    expected = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    assert photo_metadata(path).captured_at == expected


def test_video_dimensions_and_duration(settings):
    path = settings.video_root / "tiny.mp4"
    make_video(path)
    metadata = video_metadata(path, settings.ffprobe_path)
    assert (metadata.width, metadata.height) == (48, 32)
    assert metadata.duration_seconds is not None
    assert 0.2 <= metadata.duration_seconds <= 1.0

