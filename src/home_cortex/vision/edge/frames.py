"""Camera frame contract for the EdgeVision runtime. No detector, no API server."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

DEV_DEVICE_ID = "device:dev_macbook"
DEV_CAMERA_ID = "camera:built_in"


@dataclass(frozen=True)
class CameraFrame:
    """One captured frame. Camera-native buffers stay inside the source."""

    captured_at: str
    width: int
    height: int
    jpeg: bytes


class CameraSource(Protocol):
    device_id: str
    camera_id: str

    def open(self) -> None: ...

    def read(self) -> CameraFrame: ...

    def close(self) -> None: ...


def capture_timestamp(when: datetime | None = None) -> str:
    """Timezone-aware ISO-8601, matching VisualObservation.captured_at."""
    moment = when if when is not None else datetime.now().astimezone()
    return moment.isoformat(timespec="milliseconds")
