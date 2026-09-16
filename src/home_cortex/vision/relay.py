"""Fixed-source MJPEG relay and short-lived, Vision-only browser credentials."""
from __future__ import annotations

import hashlib
import hmac
import time
from email.message import Message

import httpx
from fastapi import HTTPException
from starlette.responses import StreamingResponse

COOKIE_NAME = "cortex_vision_session"
SESSION_SECONDS = 3600


def session_token(key: str, expires: int | None = None) -> str:
    expiry = str(expires if expires is not None else int(time.time()) + SESSION_SECONDS)
    signature = hmac.new(key.encode(), f"vision:{expiry}".encode(), hashlib.sha256).hexdigest()
    return f"{expiry}.{signature}"


def valid_session(token: str, key: str) -> bool:
    try:
        expiry, _ = token.split(".", 1)
        if not int(time.time()) < int(expiry) <= int(time.time()) + SESSION_SECONDS:
            return False
        return hmac.compare_digest(token, session_token(key, int(expiry)))
    except (ValueError, TypeError):
        return False


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


async def open_relay(url: str) -> RelayResponse:
    # Only a server-configured URL reaches here. Never forward browser credentials,
    # proxy environment variables, redirects, or arbitrary request query parameters.
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
