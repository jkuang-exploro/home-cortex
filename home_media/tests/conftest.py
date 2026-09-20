from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from home_media.app import create_app
from home_media.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    photo_root = tmp_path / "photos"
    video_root = tmp_path / "videos"
    photo_root.mkdir()
    video_root.mkdir()
    return Settings(
        photo_root=photo_root,
        video_root=video_root,
        data_root=tmp_path / "data",
        thumbnail_size=64,
        page_size=2,
        ffprobe_path=shutil.which("ffprobe") or "ffprobe",
        ffmpeg_path=shutil.which("ffmpeg") or "ffmpeg",
    )


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def make_jpeg(
    path: Path,
    *,
    size: tuple[int, int] = (40, 30),
    orientation: int | None = None,
    captured_at: str | None = None,
    color: str = "#4d8f8b",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    if captured_at is not None:
        exif[36867] = captured_at
    image.save(path, exif=exif)


def make_video(path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=48x32:d=0.4",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ],
        check=True,
        timeout=30,
    )

