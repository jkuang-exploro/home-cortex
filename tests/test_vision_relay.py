import asyncio
import time

import httpx
import pytest
from fastapi import HTTPException

from home_cortex.vision import relay


def test_session_token_expiry_and_tampering() -> None:
    key = 'test-key'
    assert relay.valid_session(relay.session_token(key), key)
    assert not relay.valid_session(relay.session_token(key, expires=int(time.time()) - 1), key)
    assert not relay.valid_session(relay.session_token(key) + 'x', key)
    assert not relay.valid_session('invalid', key)


@pytest.mark.parametrize(
    'raw, expected',
    [
        ('192.168.68.65', 'http://192.168.68.65:8088/live.mjpg'),
        ('10.0.0.8:9000', 'http://10.0.0.8:9000/live.mjpg'),
        ('http://cam.local:8088/live.mjpg', 'http://cam.local:8088/live.mjpg'),
    ],
)
def test_session_source_normalization(raw, expected):
    assert relay.normalize_stream_source(raw) == expected


@pytest.mark.parametrize('raw', ['file:///tmp/x', 'http://user:pass@camera/live', 'http://camera/live#x', '169.254.169.254'])
def test_session_source_rejects_unsafe_targets(raw):
    with pytest.raises(ValueError):
        relay.normalize_stream_source(raw)


class Chunks(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False
    async def __aiter__(self):
        yield b'--edgeframe\r\n'
        yield b'Content-Type: image/jpeg\r\n\r\n\xff\xd8\xff\xd9\r\n'
    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_relay_streams_and_closes_upstream_on_downstream_failure(monkeypatch):
    stream = Chunks()
    real_client = httpx.AsyncClient
    clients = []
    def factory(**kwargs):
        assert kwargs['trust_env'] is False
        assert kwargs['follow_redirects'] is False
        def upstream(request):
            assert 'authorization' not in request.headers
            assert 'cookie' not in request.headers
            return httpx.Response(200, headers={'Content-Type': 'multipart/x-mixed-replace; boundary=edgeframe'}, stream=stream)
        client = real_client(transport=httpx.MockTransport(upstream), **kwargs)
        clients.append(client)
        return client
    monkeypatch.setattr(relay.httpx, 'AsyncClient', factory)
    response = await relay.open_relay('http://camera/live.mjpg')
    assert response.headers['x-accel-buffering'] == 'no'
    sent = []
    async def send(event):
        sent.append(event)
        if event['type'] == 'http.response.body':
            raise OSError('browser disconnected')
    async def receive():
        await asyncio.sleep(1)
    with pytest.raises(Exception):
        await response({'type': 'http', 'asgi': {'spec_version': '2.4'}}, receive, send)
    assert sent[1]['body'] == b'--edgeframe\r\n'
    assert stream.closed and clients[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['timeout', 'redirect', 'html', 'missing-boundary', 'empty'])
async def test_upstream_failures_return_502_and_close(monkeypatch, mode):
    real_client = httpx.AsyncClient
    clients = []
    def factory(**kwargs):
        def handler(request):
            if mode == 'timeout':
                raise httpx.ReadTimeout('private upstream details', request=request)
            if mode == 'redirect':
                return httpx.Response(302, headers={'location': 'http://other/secret'})
            if mode == 'html':
                return httpx.Response(200, headers={'content-type': 'text/html'})
            if mode == 'missing-boundary':
                return httpx.Response(200, headers={'content-type': 'multipart/x-mixed-replace'})
            class Empty(httpx.AsyncByteStream):
                async def __aiter__(self):
                    if False:
                        yield b''
            return httpx.Response(200, headers={'content-type': 'multipart/x-mixed-replace; boundary=x'}, stream=Empty())
        client = real_client(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client
    monkeypatch.setattr(relay.httpx, 'AsyncClient', factory)
    with pytest.raises(HTTPException) as error:
        await relay.open_relay('http://camera/live.mjpg')
    assert error.value.status_code == 502
    assert 'private upstream details' not in error.value.detail
    assert clients[0].is_closed
