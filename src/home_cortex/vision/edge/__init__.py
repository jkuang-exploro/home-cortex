"""Mac/MicroDuck EdgeVision runtime. Separate from the semantic API process."""

from .frames import (
    DEV_CAMERA_ID,
    DEV_DEVICE_ID,
    CameraFrame,
    CameraSource,
    capture_timestamp,
)
from .runtime import EdgeRuntime
from .sources import MacCameraSource, SyntheticCameraSource
from .stream import StreamConfig

__all__ = (
    "DEV_CAMERA_ID",
    "DEV_DEVICE_ID",
    "CameraFrame",
    "CameraSource",
    "EdgeRuntime",
    "MacCameraSource",
    "StreamConfig",
    "SyntheticCameraSource",
    "capture_timestamp",
)
