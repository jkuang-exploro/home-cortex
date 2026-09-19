"""Private/local camera host policy. Never log the raw address."""

from __future__ import annotations

import ipaddress

from .errors import CameraError

_LOCAL_SUFFIXES = (".local", ".lan", ".home", ".internal")


def validate_camera_host(host: str, *, allow_loopback: bool = False) -> str:
    raw = host.strip().rstrip(".").casefold()
    if not raw or raw in {"0.0.0.0", "::", "localhost", "metadata.google.internal"}:
        raise CameraError("camera_unreachable", "Camera address is not allowed")
    try:
        address = ipaddress.ip_address(raw)
    except ValueError:
        if any(raw.endswith(suffix) for suffix in _LOCAL_SUFFIXES) and "/" not in raw:
            return raw
        raise CameraError("camera_unreachable", "Camera address is not allowed") from None
    if address.is_multicast or address.is_unspecified or address.is_link_local:
        raise CameraError("camera_unreachable", "Camera address is not allowed")
    if address.is_loopback:
        if not allow_loopback:
            raise CameraError("camera_unreachable", "Camera address is not allowed")
        return raw
    if not address.is_private:
        raise CameraError("camera_unreachable", "Camera address is not allowed")
    return raw
