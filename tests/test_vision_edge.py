"""EdgeVision camera source and MJPEG runtime. No hardware required."""
import json
import os
import time
from datetime import datetime
from urllib.request import urlopen

import pytest

from home_cortex.vision.edge.frames import (
    DEV_CAMERA_ID,
    DEV_DEVICE_ID,
    CameraFrame,
    capture_timestamp,
)
from home_cortex.vision.edge.runtime import EdgeRuntime
from home_cortex.vision.edge.sources import MacCameraSource, SyntheticCameraSource
from home_cortex.vision.edge.stream import StreamConfig
from home_cortex.vision.edge.__main__ import build_parser


def test_synthetic_source_timestamps_and_dimensions() -> None:
    source = SyntheticCameraSource(width=320, height=240, fps=5)
    source.open()
    try:
        first = source.read()
        second = source.read()
    finally:
        source.close()
    assert first.width == 320 and first.height == 240
    assert first.jpeg.startswith(b"\xff\xd8")
    parsed = datetime.fromisoformat(first.captured_at)
    assert parsed.tzinfo is not None
    assert datetime.fromisoformat(second.captured_at) > parsed
    datetime.fromisoformat(capture_timestamp())


def test_capture_timestamp_matches_observation_iso8601() -> None:
    stamp = capture_timestamp()
    datetime.fromisoformat(stamp)
    assert "T" in stamp


def test_mac_source_is_a_replaceable_camera_source() -> None:
    mac = MacCameraSource()
    synthetic = SyntheticCameraSource()
    for source in (mac, synthetic):
        assert source.device_id == DEV_DEVICE_ID
        assert source.camera_id == DEV_CAMERA_ID
        assert callable(source.open)
        assert callable(source.read)
        assert callable(source.close)


def test_stream_config_endpoint() -> None:
    config = StreamConfig(host="127.0.0.1", port=8088)
    assert config.transport == "mjpeg-http"
    assert config.endpoint == "http://127.0.0.1:8088/live.mjpg"
    assert config.viewer == "http://127.0.0.1:8088/"


def test_runtime_start_health_and_clean_shutdown() -> None:
    source = SyntheticCameraSource(fps=20)
    runtime = EdgeRuntime(source, config=StreamConfig(host="127.0.0.1", port=0), fps=20)
    endpoint = runtime.start()
    try:
        assert endpoint.startswith("http://127.0.0.1:")
        assert endpoint.endswith("/live.mjpg")
        deadline = time.time() + 2
        while runtime.latest_frame() is None and time.time() < deadline:
            runtime.wait(0.02)
        frame = runtime.latest_frame()
        assert isinstance(frame, CameraFrame)
        assert frame.jpeg.startswith(b"\xff\xd8")
        with urlopen(endpoint.rsplit("/", 1)[0] + "/health", timeout=2) as response:
            health = json.loads(response.read().decode())
        assert health["device_id"] == DEV_DEVICE_ID
        assert health["camera_id"] == DEV_CAMERA_ID
        assert health["transport"] == "mjpeg-http"
        assert health["running"] is True
        assert health["source_status"] == "online"
        assert health["failure_code"] is None
        assert health["last_captured_at"] == frame.captured_at
        with urlopen(endpoint, timeout=2) as stream:
            chunk = stream.read(160)
        assert b"--edgeframe" in chunk or chunk.startswith(b"--") or b"\xff\xd8" in chunk
    finally:
        runtime.stop()
    assert runtime.running is False


def test_capture_failure_has_stable_health_state() -> None:
    class FailingSource(SyntheticCameraSource):
        def read(self) -> CameraFrame:
            raise RuntimeError("native camera detail must not cross the boundary")

    source = FailingSource()
    runtime = EdgeRuntime(source, config=StreamConfig(host="127.0.0.1", port=0))
    runtime.start()
    try:
        deadline = time.time() + 2
        while runtime.running and time.time() < deadline:
            runtime.wait(0.01)
        health = json.loads(runtime.health_json())
        assert health["running"] is False
        assert health["source_status"] == "camera_disconnected"
        assert health["failure_code"] == "camera_read_failed"
        assert "native camera detail" not in runtime.health_json()
    finally:
        runtime.stop()


def test_stream_start_failure_closes_capture_source(monkeypatch) -> None:
    class RecordingSource(SyntheticCameraSource):
        closed = False

        def close(self) -> None:
            self.closed = True
            super().close()

    source = RecordingSource()
    runtime = EdgeRuntime(source)

    def fail_start() -> None:
        raise OSError("bind failed")

    monkeypatch.setattr(runtime._stream, "start", fail_start)
    with pytest.raises(OSError, match="bind failed"):
        runtime.start()

    assert source.closed is True
    assert runtime.running is False


def test_cli_help_and_synthetic_main(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(["--source", "synthetic", "--port", "0"])
    assert args.source == "synthetic"
    # Do not run main() here: it loops until interrupt.


@pytest.mark.manual
@pytest.mark.skipif(os.environ.get("VISION_EDGE_SMOKE") != "1", reason="opt-in Mac camera smoke")
def test_mac_camera_smoke() -> None:
    source = MacCameraSource()
    source.open()
    try:
        frame = source.read()
    finally:
        source.close()
    assert frame.width > 0 and frame.height > 0
    assert frame.jpeg.startswith(b"\xff\xd8")
    datetime.fromisoformat(frame.captured_at)
