"""MJPEG relay and short-lived Vision credentials. Source URL is session-bound."""
from __future__ import annotations

import base64
import hashlib
import hmac
import time
from email.message import Message
from urllib.parse import urlparse, urlunparse

import httpx
from fastapi import HTTPException
from collections.abc import AsyncIterator

from starlette.responses import StreamingResponse

from .camera.errors import CameraError
from .camera.hub import MJPEG_TYPE, SharedTapoHub
from .camera.sources import CameraSource

COOKIE_NAME = "cortex_vision_session"
SESSION_SECONDS = 3600
DEFAULT_STREAM_PORT = 8088
DEFAULT_STREAM_PATH = "/live.mjpg"
BLOCKED_HOSTS = frozenset({"0.0.0.0", "169.254.169.254", "metadata.google.internal"})


def _b64(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def _unb64(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}").decode()


def normalize_stream_source(value: str) -> str:
    """Accept an IP, host:port, or http(s) URL. Never credentials, fragments, or metadata hosts."""
    raw = value.strip()
    if not raw:
        raise ValueError("Camera address is required")
    if "://" not in raw:
        hostport = raw.split("/", 1)[0]
        if hostport.count(":") == 1:
            raw = f"http://{hostport}{DEFAULT_STREAM_PATH}"
        else:
            raw = f"http://{hostport}:{DEFAULT_STREAM_PORT}{DEFAULT_STREAM_PATH}"
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Camera URL must be http or https")
    if parsed.username or parsed.password:
        raise ValueError("Camera URL must not contain credentials")
    if parsed.fragment or parsed.query:
        raise ValueError("Camera URL must not contain a query or fragment")
    host = (parsed.hostname or "").casefold()
    if not host or host in BLOCKED_HOSTS or host.startswith("169.254."):
        raise ValueError("Camera address is not allowed")
    path = parsed.path if parsed.path and parsed.path != "/" else DEFAULT_STREAM_PATH
    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"
    return urlunparse((parsed.scheme, netloc, path, "", "", ""))


def session_token(key: str, source: str = "", *, expires: int | None = None) -> str:
    expiry = expires if expires is not None else int(time.time()) + SESSION_SECONDS
    payload = f"vision:{expiry}:{source}"
    signature = hmac.new(key.encode(), payload.encode(), hashlib.sha256).hexdigest()
    if source:
        return f"{expiry}.{_b64(source)}.{signature}"
    return f"{expiry}.{signature}"


def parse_session(token: str, key: str) -> str | None:
    try:
        parts = token.split(".")
        if len(parts) == 2:
            expiry_text, _signature = parts
            source = ""
        elif len(parts) == 3:
            expiry_text, encoded, _signature = parts
            source = _unb64(encoded)
        else:
            return None
        expiry = int(expiry_text)
        if not int(time.time()) < expiry <= int(time.time()) + SESSION_SECONDS:
            return None
        expected = session_token(key, source, expires=expiry)
        if not hmac.compare_digest(token, expected):
            return None
        return source
    except (ValueError, TypeError, UnicodeDecodeError):
        return None


def valid_session(token: str, key: str) -> bool:
    return parse_session(token, key) is not None


def session_source(token: str, key: str) -> str:
    return parse_session(token, key) or ""


class RelayResponse(StreamingResponse):
    """Own upstream resources even if downstream disconnects before iteration."""

    def __init__(self, body, content_type, upstream, client):
        super().__init__(body, headers={
            "Content-Type": content_type,
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        })
        self.upstream = upstream
        self.client = client

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await self.upstream.aclose()
            finally:
                await self.client.aclose()


async def open_relay(
    source: str | CameraSource,
    *,
    hub: SharedTapoHub | None = None,
) -> RelayResponse:
    if isinstance(source, CameraSource) and source.kind == "tapo":
        return await _open_tapo(source, hub)
    url = source if isinstance(source, str) else source.http_url
    if not url:
        raise HTTPException(503, "Provide the camera IP or URL when connecting")
    return await _open_http_mjpeg(url)


async def _open_tapo(source: CameraSource, hub: SharedTapoHub | None) -> RelayResponse:
    session = hub or SharedTapoHub()
    owned = hub is None
    chunks = session.subscribe(source)
    try:
        first = await anext(chunks)
    except CameraError:
        if owned:
            await session.close()
        raise
    except StopAsyncIteration as error:
        if owned:
            await session.close()
        raise HTTPException(502, "Camera stream unreachable or stalled") from error

    async def body() -> AsyncIterator[bytes]:
        yield first
        try:
            async for chunk in chunks:
                yield chunk
        finally:
            aclose = getattr(chunks, "aclose", None)
            if aclose is not None:
                await aclose()
            if owned:
                await session.close()

    return RelayResponse(body(), MJPEG_TYPE, _NullCloseable(), _NullCloseable())


class _NullCloseable:
    async def aclose(self) -> None:
        return None


async def _open_http_mjpeg(url: str) -> RelayResponse:
    # Only a validated HTTP(S) MJPEG URL reaches here. Never forward browser
    # credentials, proxy environment variables, redirects, or query parameters.
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=3.0),
        follow_redirects=False,
        trust_env=False,
    )
    upstream = None
    try:
        upstream = await client.send(client.build_request(
            "GET", url, headers={"Accept-Encoding": "identity"},
        ), stream=True)
        content_type = upstream.headers.get("content-type", "")
        message = Message()
        message["content-type"] = content_type
        if (upstream.status_code != 200
                or message.get_content_type() != "multipart/x-mixed-replace"
                or not message.get_param("boundary")):
            raise HTTPException(502, "Camera did not return an MJPEG stream")
        chunks = upstream.aiter_raw()
        first = await anext(chunks)  # Fail before HTTP 200 if the source stalls immediately.
    except BaseException as error:
        if upstream is not None:
            await upstream.aclose()
        await client.aclose()
        if isinstance(error, (httpx.HTTPError, StopAsyncIteration)):
            raise HTTPException(502, "Camera stream unreachable or stalled; check the Mac listener and server network") from None
        raise

    async def body():
        yield first
        try:
            async for chunk in chunks:
                yield chunk
        except httpx.HTTPError:
            # Headers have already gone out. End the stream; the browser can reconnect.
            return

    return RelayResponse(body(), content_type, upstream, client)
