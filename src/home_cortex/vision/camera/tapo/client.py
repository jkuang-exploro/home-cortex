"""In-process Tapo C610 stream client. Never logs secrets or camera addresses."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from ..errors import (
    CameraError,
    auth_failed,
    failed,
    timed_out,
    unreachable,
    unsupported,
)
from ..sources import CameraSource
from .auth import derive_cloud_credentials, digest_authorization, parse_digest_challenge
from .crypto import decrypt_payload, derive_session_keys, parse_key_exchange
from .mpegts import realign_mpegts

CONNECT_TIMEOUT = 3.0
HANDSHAKE_TIMEOUT = 5.0
FIRST_FRAME_TIMEOUT = 15.0
IDLE_TIMEOUT = 10.0
CLIENT_BOUNDARY = b"--client-stream-boundary--"
DEVICE_BOUNDARY = b"--device-stream-boundary--"
STREAM_URI = "/stream"


class TapoClient:
    def __init__(self, source: CameraSource, *, cnonce: str | None = None) -> None:
        self._source = source
        self._cnonce = cnonce
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._key = b""
        self._iv = b""
        self._seq = 1
        self._open = False

    async def open(self) -> None:
        host, port, subtype = self._source.tapo_endpoint()
        self._http_host = f"{host}:{port}"
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=CONNECT_TIMEOUT,
            )
        except TimeoutError as error:
            raise timed_out() from error
        except OSError as error:
            raise unreachable() from error
        try:
            await asyncio.wait_for(self._handshake(subtype), timeout=HANDSHAKE_TIMEOUT)
        except TimeoutError as error:
            await self.close()
            raise timed_out() from error
        except CameraError:
            await self.close()
            raise
        except Exception as error:
            await self.close()
            raise failed() from error
        self._open = True

    async def _handshake(self, subtype: int) -> None:
        assert self._reader is not None and self._writer is not None
        await self._write_http(STREAM_URI, extra_headers=())
        challenge = await self._read_http()
        if challenge["status"] != 401:
            raise auth_failed()
        www = challenge["headers"].get("www-authenticate", "")
        fields = parse_digest_challenge(www)
        username, password = derive_cloud_credentials(
            self._source.secret(),
            fields.get("encrypt_type"),
        )
        authorization = digest_authorization(
            username=username,
            password=password,
            realm=fields["realm"],
            nonce=fields["nonce"],
            uri=STREAM_URI,
            qop=fields.get("qop", "auth"),
            opaque=fields.get("opaque"),
            cnonce=self._cnonce,
        )
        await self._write_http(STREAM_URI, extra_headers=(f"Authorization: {authorization}",))
        accepted = await self._read_http()
        if accepted["status"] != 200:
            raise auth_failed()
        exchange = parse_key_exchange(accepted["headers"].get("key-exchange", ""))
        nonce = exchange.get("nonce")
        if not nonce:
            raise unsupported()
        if exchange.get("encrypt_type") == "3" and fields.get("encrypt_type") != "3":
            username, password = derive_cloud_credentials(self._source.secret(), "3")
        self._key, self._iv = derive_session_keys(username, password, nonce)
        resolution = "VGA" if subtype == 1 else "HD"
        body = json.dumps(
            {
                "params": {
                    "preview": {
                        "audio": ["default"],
                        "channels": [0],
                        "resolutions": [resolution],
                    },
                    "method": "get",
                },
                "seq": self._seq,
                "type": "request",
            },
            separators=(",", ":"),
        ).encode()
        self._seq += 1
        await self._write_part(body)

    async def iter_mpegts(self) -> AsyncIterator[bytes]:
        if not self._open or self._reader is None:
            raise failed()
        leftover = bytearray()
        first = True
        timeout = FIRST_FRAME_TIMEOUT
        while self._open:
            try:
                headers, payload = await asyncio.wait_for(
                    self._read_part(),
                    timeout=timeout,
                )
            except TimeoutError as error:
                raise timed_out() from error
            except (asyncio.IncompleteReadError, ConnectionError) as error:
                raise failed() from error
            timeout = IDLE_TIMEOUT
            content_type = headers.get("content-type", "")
            if content_type != "video/mp2t":
                continue
            try:
                plain = decrypt_payload(payload, self._key, self._iv)
            except CameraError:
                raise
            packets, leftover = realign_mpegts(leftover, plain)
            await self._acknowledge()
            if packets:
                first = False
                yield packets
            elif first:
                continue

    async def close(self) -> None:
        self._open = False
        writer = self._writer
        self._writer = None
        self._reader = None
        if writer is None:
            return
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            return

    async def _acknowledge(self) -> None:
        payload = json.dumps(
            {"type": "notification", "seq": self._seq, "params": {"eventType": "streamAck"}},
            separators=(",", ":"),
        ).encode()
        self._seq += 1
        await self._write_part(payload)

    async def _write_http(self, uri: str, extra_headers: tuple[str, ...]) -> None:
        assert self._writer is not None
        lines = [
            f"POST {uri} HTTP/1.1",
            f"Host: {getattr(self, '_http_host', 'camera')}",
            "Content-Type: multipart/mixed; boundary=--client-stream-boundary--",
            "Connection: keep-alive",
            *extra_headers,
            "",
            "",
        ]
        self._writer.write("\r\n".join(lines).encode())
        await self._writer.drain()

    async def _read_http(self) -> dict[str, object]:
        assert self._reader is not None
        header_block = await self._reader.readuntil(b"\r\n\r\n")
        lines = header_block.decode("latin-1").split("\r\n")
        status_line = lines[0]
        try:
            status = int(status_line.split(" ")[1])
        except (IndexError, ValueError) as error:
            raise unsupported() from error
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            headers[name.strip().casefold()] = value.strip()
        length = int(headers.get("content-length", "0") or 0)
        if length:
            await self._reader.readexactly(length)
        return {"status": status, "headers": headers}

    async def _write_part(self, body: bytes) -> None:
        assert self._writer is not None
        header = (
            b"----client-stream-boundary--\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n"
        )
        self._writer.write(header + body + b"\r\n")
        await self._writer.drain()

    async def _read_part(self) -> tuple[dict[str, str], bytes]:
        assert self._reader is not None
        while True:
            line = await self._reader.readline()
            if not line:
                raise failed()
            if DEVICE_BOUNDARY in line or line.strip() == b"----device-stream-boundary--":
                break
        headers: dict[str, str] = {}
        while True:
            line = await self._reader.readline()
            if line in {b"\r\n", b"\n", b""}:
                break
            decoded = line.decode("latin-1").strip()
            if ":" not in decoded:
                continue
            name, value = decoded.split(":", 1)
            headers[name.strip().casefold()] = value.strip()
        length = int(headers.get("content-length", "0") or 0)
        payload = await self._reader.readexactly(length) if length else b""
        return headers, payload
