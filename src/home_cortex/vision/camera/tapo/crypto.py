"""Tapo key-exchange parsing and AES-CBC payload decryption."""

from __future__ import annotations

import hashlib
import re
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from ..errors import CameraError

_FIELD = re.compile(r'([a-zA-Z][a-zA-Z0-9_-]*)="([^"]*)"')


def parse_key_exchange(header: str) -> dict[str, str]:
    if not header:
        raise CameraError("camera_stream_unsupported", "Camera did not return a usable stream")
    return {name.casefold(): value for name, value in _FIELD.findall(header)}


def derive_session_keys(username: str, password: str, nonce: str) -> tuple[bytes, bytes]:
    key = hashlib.md5(f"{nonce}:{password}".encode()).digest()
    iv = hashlib.md5(f"{username}:{nonce}".encode()).digest()
    return key, iv


def decrypt_payload(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    if not ciphertext or len(ciphertext) % 16 != 0:
        raise CameraError("camera_stream_failed", "Camera stream failed")
    decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    pad = padded[-1]
    if pad < 1 or pad > 16 or padded[-pad:] != bytes([pad]) * pad:
        raise CameraError("camera_stream_failed", "Camera stream failed")
    return padded[:-pad]
