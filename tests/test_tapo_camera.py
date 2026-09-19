import asyncio
import hashlib

import pytest
from pydantic import SecretStr

from home_cortex.vision.camera.codec import ScriptedMjpegAdapter
from home_cortex.vision.camera.errors import CameraError
from home_cortex.vision.camera.hosts import validate_camera_host
from home_cortex.vision.camera.hub import SharedTapoHub
from home_cortex.vision.camera.sources import CameraSource
from home_cortex.vision.camera.tapo.auth import (
    derive_cloud_credentials,
    digest_authorization,
    parse_digest_challenge,
)
from home_cortex.vision.camera.tapo.client import TapoClient
from home_cortex.vision.camera.tapo.crypto import (
    decrypt_payload,
    derive_session_keys,
    parse_key_exchange,
)
from home_cortex.vision.camera.tapo.mpegts import TS_PACKET_SIZE, realign_mpegts
from home_cortex.vision.relay import open_relay


def test_digest_challenge_parsing() -> None:
    header = (
        'Digest realm="TP-LINK", nonce="abc123", qop="auth", encrypt_type="3"'
    )
    fields = parse_digest_challenge(header)
    assert fields["realm"] == "TP-LINK"
    assert fields["nonce"] == "abc123"
    assert fields["encrypt_type"] == "3"


def test_cloud_password_md5_and_sha256() -> None:
    user, md5_secret = derive_cloud_credentials("cloud-secret", None)
    assert user == "admin"
    assert md5_secret == hashlib.md5(b"cloud-secret").hexdigest().upper()
    user, sha = derive_cloud_credentials("cloud-secret", "3")
    assert user == "admin"
    assert sha == hashlib.sha256(b"cloud-secret").hexdigest().upper()
    assert md5_secret != sha


def test_digest_authorization_is_deterministic_with_cnonce() -> None:
    header = digest_authorization(
        username="admin",
        password="AABBCC",
        realm="TP-LINK",
        nonce="n1",
        uri="/stream",
        cnonce="fixed-cnonce",
    )
    assert "username=\"admin\"" in header
    assert "response=" in header
    assert "cloud-secret" not in header


def test_key_exchange_and_aes_roundtrip() -> None:
    exchange = 'key_size="128", nonce="nonce-value", encrypt_type="3"'
    fields = parse_key_exchange(exchange)
    key, iv = derive_session_keys("admin", "HASH", fields["nonce"])
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    plain = b"hello tapo stream"
    pad = 16 - (len(plain) % 16)
    padded = plain + bytes([pad]) * pad
    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    cipher = encryptor.update(padded) + encryptor.finalize()
    assert decrypt_payload(cipher, key, iv) == plain


def test_decrypt_rejects_truncated_and_malformed_packets() -> None:
    key, iv = derive_session_keys("admin", "HASH", "nonce")
    with pytest.raises(CameraError) as error:
        decrypt_payload(b"short", key, iv)
    assert error.value.code == "camera_stream_failed"
    with pytest.raises(CameraError):
        decrypt_payload(b"\x00" * 16, key, iv)


def test_mpegts_realigns_split_packets() -> None:
    packet = bytes([0x47]) + b"\x11" * (TS_PACKET_SIZE - 1)
    first, rest = realign_mpegts(bytearray(), b"xx" + packet[:20])
    assert first == b""
    second, rest = realign_mpegts(rest, packet[20:] + packet)
    assert second == packet * 2
    assert rest == bytearray()


def test_private_network_validation() -> None:
    assert validate_camera_host("192.168.1.20") == "192.168.1.20"
    assert validate_camera_host("10.0.0.8") == "10.0.0.8"
    assert validate_camera_host("camera.local") == "camera.local"
    for host in ("8.8.8.8", "127.0.0.1", "169.254.169.254", "224.0.0.1", "0.0.0.0"):
        with pytest.raises(CameraError):
            validate_camera_host(host)
    assert validate_camera_host("127.0.0.1", allow_loopback=True) == "127.0.0.1"


def _encrypt(plain: bytes, username: str, password: str, nonce: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    key, iv = derive_session_keys(username, password, nonce)
    pad = 16 - (len(plain) % 16)
    padded = plain + bytes([pad]) * pad
    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    return encryptor.update(padded) + encryptor.finalize()


async def _tapo_camera(handler) -> tuple[asyncio.AbstractServer, int]:
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


def _http_message(status: int, headers: list[str], body: bytes = b"") -> bytes:
    lines = [f"HTTP/1.1 {status} X"] + headers + [f"Content-Length: {len(body)}", "", ""]
    return "\r\n".join(lines).encode() + body


@pytest.mark.asyncio
async def test_tapo_client_auth_key_exchange_ack_and_ts(monkeypatch: pytest.MonkeyPatch) -> None:
    password = "fixture-cloud-password"
    _, hashed = derive_cloud_credentials(password, "3")
    packet = bytes([0x47]) + b"\x22" * (TS_PACKET_SIZE - 1)
    cipher = _encrypt(packet, "admin", hashed, "session-nonce")
    acks: list[bytes] = []

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(_http_message(401, [
            'WWW-Authenticate: Digest realm="TP", nonce="n1", qop="auth", encrypt_type="3"',
        ]))
        await writer.drain()
        second = await reader.readuntil(b"\r\n\r\n")
        assert b"Authorization: Digest" in second
        writer.write(_http_message(200, [
            'Key-Exchange: nonce="session-nonce", encrypt_type="3"',
            "Content-Type: multipart/mixed; boundary=--device-stream-boundary--",
        ]))
        await writer.drain()
        await reader.readuntil(b"Content-Length:")
        await reader.readline()
        await reader.readuntil(b"\r\n")
        part = (
            b"----device-stream-boundary--\r\n"
            b"Content-Type: video/mp2t\r\n"
            b"Content-Length: " + str(len(cipher)).encode() + b"\r\n\r\n" + cipher
        )
        writer.write(part)
        await writer.drain()
        ack = await reader.readuntil(b"streamAck")
        acks.append(ack)
        writer.close()
        await writer.wait_closed()

    server, port = await _tapo_camera(handler)
    source = CameraSource(
        kind="tapo",
        host="127.0.0.1",
        port=port,
        cloud_password=SecretStr(password),
        allow_loopback=True,
    )
    client = TapoClient(source, cnonce="cnonce")
    try:
        await client.open()
        packets = []
        async for chunk in client.iter_mpegts():
            packets.append(chunk)
            break
        assert packets[0].startswith(b"\x47")
        for _ in range(50):
            if acks:
                break
            await asyncio.sleep(0.01)
        assert acks and b"streamAck" in acks[0]
        assert password.encode() not in acks[0]
    finally:
        await client.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_tapo_handshake_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("home_cortex.vision.camera.tapo.client.HANDSHAKE_TIMEOUT", 0.05)

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await asyncio.sleep(1)

    server, port = await _tapo_camera(handler)
    source = CameraSource(
        kind="tapo",
        host="127.0.0.1",
        port=port,
        cloud_password=SecretStr("secret"),
        allow_loopback=True,
    )
    client = TapoClient(source)
    with pytest.raises(CameraError) as error:
        await client.open()
    assert error.value.code == "camera_stream_timeout"
    await client.close()
    server.close()
    await server.wait_closed()


@pytest.mark.asyncio
async def test_tapo_auth_failure_is_bounded() -> None:
    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(_http_message(401, [
            'WWW-Authenticate: Digest realm="TP", nonce="n1", qop="auth"',
        ]))
        await writer.drain()
        await reader.readuntil(b"\r\n\r\n")
        writer.write(_http_message(401, ['WWW-Authenticate: Digest realm="TP", nonce="n2"']))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server, port = await _tapo_camera(handler)
    source = CameraSource(
        kind="tapo",
        host="127.0.0.1",
        port=port,
        cloud_password=SecretStr("wrong"),
        allow_loopback=True,
    )
    client = TapoClient(source, cnonce="cnonce")
    with pytest.raises(CameraError) as error:
        await client.open()
    assert error.value.code == "camera_auth_failed"
    assert "wrong" not in str(error.value)
    server.close()
    await server.wait_closed()


@pytest.mark.asyncio
async def test_shared_session_fanout_and_shutdown() -> None:
    packet = bytes([0x47]) + b"\x33" * (TS_PACKET_SIZE - 1)
    ready = asyncio.Event()
    stop = asyncio.Event()
    clients: list[object] = []

    class FakeClient:
        def __init__(self, source: CameraSource) -> None:
            self.source = source
            self.closed = False
            clients.append(self)

        async def open(self) -> None:
            return None

        async def iter_mpegts(self):
            await ready.wait()
            yield packet
            await stop.wait()

        async def close(self) -> None:
            self.closed = True

    hub = SharedTapoHub(client_factory=FakeClient, codec_factory=ScriptedMjpegAdapter)
    source = CameraSource(
        kind="tapo",
        host="192.168.1.20",
        cloud_password=SecretStr("secret"),
    )
    first = hub.subscribe(source)
    second = hub.subscribe(source)
    waiter1 = asyncio.create_task(anext(first))
    waiter2 = asyncio.create_task(anext(second))
    await asyncio.sleep(0.05)
    ready.set()
    chunk1, chunk2 = await asyncio.wait_for(asyncio.gather(waiter1, waiter2), timeout=2)
    assert b"\xff\xd8" in chunk1 and chunk1 == chunk2
    stop.set()
    await first.aclose()
    await second.aclose()
    await asyncio.sleep(0.05)
    assert len(clients) == 1
    await hub.close()


@pytest.mark.asyncio
async def test_open_relay_tapo_fixture_to_multipart(monkeypatch: pytest.MonkeyPatch) -> None:
    packet = bytes([0x47]) + b"\x44" * (TS_PACKET_SIZE - 1)

    class FakeClient:
        def __init__(self, source: CameraSource) -> None:
            self.source = source

        async def open(self) -> None:
            return None

        async def iter_mpegts(self):
            yield packet

        async def close(self) -> None:
            return None

    hub = SharedTapoHub(client_factory=FakeClient, codec_factory=ScriptedMjpegAdapter)
    source = CameraSource(kind="tapo", host="192.168.1.20", cloud_password=SecretStr("secret"))
    response = await open_relay(source, hub=hub)
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
    body = b""
    iterator = response.body_iterator
    try:
        async for chunk in iterator:
            body += chunk
            if b"\xff\xd8" in body:
                break
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()
        await hub.close()
    assert b"\xff\xd8" in body and b"\xff\xd9" in body


@pytest.mark.asyncio
async def test_http_mjpeg_relay_still_used_for_string_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    from home_cortex.vision import relay
    import httpx

    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"--edgeframe\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8\xff\xd9\r\n"

        async def aclose(self) -> None:
            return None

    real_client = httpx.AsyncClient

    def factory(**kwargs):
        def upstream(request):
            return httpx.Response(
                200,
                headers={"Content-Type": "multipart/x-mixed-replace; boundary=edgeframe"},
                stream=Chunks(),
            )

        return real_client(transport=httpx.MockTransport(upstream), **kwargs)

    monkeypatch.setattr(relay.httpx, "AsyncClient", factory)
    response = await relay.open_relay("http://camera/live.mjpg")
    assert "multipart/x-mixed-replace" in response.headers["content-type"]
