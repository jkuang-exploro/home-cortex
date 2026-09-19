"""Validated camera source configuration and the MPEG-TS producer protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import SecretStr

from .errors import CameraError
from .hosts import validate_camera_host

CameraKind = Literal["http_mjpeg", "tapo"]


@dataclass(frozen=True)
class CameraSource:
    kind: CameraKind
    http_url: str | None = None
    host: str | None = None
    port: int = 8800
    subtype: int = 0
    cloud_password: SecretStr | None = None
    allow_loopback: bool = False

    def tapo_endpoint(self) -> tuple[str, int, int]:
        if self.kind != "tapo" or not self.host:
            raise CameraError("vision_not_configured", "Tapo camera is not configured")
        host = validate_camera_host(self.host, allow_loopback=self.allow_loopback)
        if not 1 <= self.port <= 65535:
            raise CameraError("vision_not_configured", "Tapo camera is not configured")
        subtype = 1 if self.subtype == 1 else 0
        return host, self.port, subtype

    def secret(self) -> str:
        if self.cloud_password is None:
            raise CameraError("vision_not_configured", "Tapo camera is not configured")
        secret = self.cloud_password.get_secret_value()
        if not secret:
            raise CameraError("vision_not_configured", "Tapo camera is not configured")
        return secret


class CameraStreamSource(Protocol):
    async def open(self) -> None: ...

    def iter_mpegts(self) -> AsyncIterator[bytes]: ...

    async def close(self) -> None: ...


def source_from_settings(settings: Mapping[str, object] | object) -> CameraSource | None:
    kind = str(getattr(settings, "vision_camera_kind", "http_mjpeg") or "http_mjpeg")
    if kind == "tapo":
        host = getattr(settings, "vision_camera_host", None)
        if not host:
            return None
        return CameraSource(
            kind="tapo",
            host=str(host),
            port=int(getattr(settings, "vision_camera_port", 8800) or 8800),
            subtype=int(getattr(settings, "vision_camera_subtype", 0) or 0),
            cloud_password=getattr(settings, "vision_tapo_cloud_password", None),
        )
    url = getattr(settings, "vision_stream_url", None)
    if url:
        return CameraSource(kind="http_mjpeg", http_url=str(url))
    return None
