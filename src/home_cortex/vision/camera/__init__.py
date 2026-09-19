"""In-process camera adapters for Vision. Isolated from observation and identity."""

from .errors import CameraError
from .hosts import validate_camera_host
from .sources import CameraSource, CameraStreamSource

__all__ = (
    "CameraError",
    "CameraSource",
    "CameraStreamSource",
    "validate_camera_host",
)
