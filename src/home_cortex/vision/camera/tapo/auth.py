"""Tapo digest authentication and cloud-password credential derivation."""

from __future__ import annotations

import hashlib
import os
import re

from ..errors import CameraError

_HEADER_FIELD = re.compile(r'([a-zA-Z][a-zA-Z0-9_-]*)="([^"]*)"')


def parse_digest_challenge(header: str) -> dict[str, str]:
    if not header or not header.lstrip().lower().startswith("digest"):
        raise CameraError("camera_auth_failed", "Camera rejected the credentials")
    fields = {name.casefold(): value for name, value in _HEADER_FIELD.findall(header)}
    if "realm" not in fields or "nonce" not in fields:
        raise CameraError("camera_auth_failed", "Camera rejected the credentials")
    return fields


def derive_cloud_credentials(cloud_password: str, encrypt_type: str | None) -> tuple[str, str]:
    """Return (username, hashed_password) for the Tapo cloud-password form."""
    if not cloud_password:
        raise CameraError("camera_auth_failed", "Camera rejected the credentials")
    digest = hashlib.sha256 if encrypt_type == "3" else hashlib.md5
    return "admin", digest(cloud_password.encode()).hexdigest().upper()


def _md5_hex(*parts: str) -> str:
    return hashlib.md5(":".join(parts).encode()).hexdigest()


def digest_authorization(
    *,
    username: str,
    password: str,
    realm: str,
    nonce: str,
    uri: str,
    method: str = "POST",
    qop: str = "auth",
    nc: str = "00000001",
    cnonce: str | None = None,
    opaque: str | None = None,
) -> str:
    token = cnonce if cnonce is not None else os.urandom(16).hex()
    ha1 = _md5_hex(username, realm, password)
    ha2 = _md5_hex(method, uri)
    response = _md5_hex(ha1, nonce, nc, token, qop or "auth", ha2)
    header = (
        f'Digest username="{username}", realm="{realm}", nonce="{nonce}", '
        f'uri="{uri}", qop={qop or "auth"}, nc={nc}, cnonce="{token}", '
        f'response="{response}"'
    )
    if opaque:
        header += f', opaque="{opaque}", algorithm=MD5'
    return header
