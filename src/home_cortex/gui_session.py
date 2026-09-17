"""HttpOnly household GUI session. Identity still goes through CORTEX_IDENTITY_MAP."""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from typing import Literal

COOKIE_NAME = "cortex_session"
SESSION_SECONDS = 12 * 60 * 60
IdentityKind = Literal["email", "id", "none"]


def _b64(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def _unb64(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}").decode()


def session_token(
    key: str,
    kind: IdentityKind,
    value: str = "",
    expires: int | None = None,
) -> str:
    expiry = expires if expires is not None else int(time.time()) + SESSION_SECONDS
    payload = f"gui:{kind}:{value}:{expiry}"
    signature = hmac.new(key.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{kind}.{_b64(value)}.{expiry}.{signature}"


def parse_session(token: str, key: str) -> tuple[IdentityKind, str] | None:
    try:
        kind, encoded, expiry_text, _signature = token.split(".", 3)
        if kind not in {"email", "id", "none"}:
            return None
        value = _unb64(encoded)
        expiry = int(expiry_text)
        if not int(time.time()) < expiry <= int(time.time()) + SESSION_SECONDS:
            return None
        expected = session_token(key, kind, value, expiry)  # type: ignore[arg-type]
        if not hmac.compare_digest(token, expected):
            return None
        return kind, value  # type: ignore[return-value]
    except (ValueError, TypeError, UnicodeDecodeError):
        return None


def valid_session(token: str, key: str) -> bool:
    return parse_session(token, key) is not None
