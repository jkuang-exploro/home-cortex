import asyncio
import socket
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from fastapi import HTTPException
from fastapi.testclient import TestClient

from home_cortex.api import app
from home_cortex.config import Settings
from home_cortex.vision import relay
from home_cortex.vision.edge.runtime import EdgeRuntime
from home_cortex.vision.edge.sources import SyntheticCameraSource
from home_cortex.vision.edge.stream import StreamConfig


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app.state, 'settings', SimpleNamespace(
        cortex_api_key='test-key', vision_stream_url='http://camera:8088/live.mjpg',
    ), raising=False)
    test_client = TestClient(app)
    yield test_client
    test_client.close()


def test_session_protects_media_without_changing_other_api_auth(client, monkeypatch):
    calls = []
    async def fake_relay(url):
        from starlette.responses import Response
        calls.append(url)
        return Response(b'frame', media_type='multipart/x-mixed-replace; boundary=test')
    monkeypatch.setattr('home_cortex.api.open_relay', fake_relay)
    assert client.get('/vision/stream').status_code == 401
    assert client.post('/vision/session').status_code == 401
    assert calls == []
    response = client.post('/vision/session', headers={'Authorization': 'Bearer test-key'})
    assert response.status_code == 200
    assert 'HttpOnly' in response.headers['set-cookie']
    assert 'SameSite=strict' in response.headers['set-cookie']
    assert 'Path=/vision' in response.headers['set-cookie']
    assert 'test-key' not in response.headers['set-cookie']
    assert client.get('/vision/stream?url=http://other-host/secret').content == b'frame'
    assert calls == ['http://camera:8088/live.mjpg']
    assert client.get('/v1/models').status_code == 401
    assert client.post('/vision/session').status_code == 200
    app.state.settings.cortex_api_key = 'rotated-key'
    assert client.get('/vision/stream').status_code == 401


def test_missing_config_and_no_key_development(client):
    app.state.settings.cortex_api_key = None
    app.state.settings.vision_stream_url = None
    assert client.post('/vision/session').status_code == 503
    assert client.get('/vision/stream').status_code == 503
    app.state.settings.vision_stream_url = 'http://camera/live.mjpg'
    assert client.post('/vision/session').status_code == 200


def test_cookie_expiry_tampering_and_secure_flag(client):
    key = 'test-key'
    assert relay.valid_session(relay.session_token(key), key)
    assert not relay.valid_session(relay.session_token(key, int(time.time()) - 1), key)
    assert not relay.valid_session(relay.session_token(key) + 'x', key)
    assert not relay.valid_session('invalid', key)
    response = client.post('https://testserver/vision/session', headers={'Authorization': 'Bearer test-key'})
    assert '; Secure' in response.headers['set-cookie']


@pytest.mark.parametrize('url', ['file:///tmp/x', 'http://user:pass@camera/live', 'http://camera/live#fragment'])
def test_source_validation(url):
    with pytest.raises(ValueError):
        Settings(_env_file=None, vision_stream_url=url)


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


def test_real_http_stream_through_cortex_server(client):
    # Actual sockets, actual MJPEG source, actual Cortex ASGI app; no camera required.
    runtime = EdgeRuntime(SyntheticCameraSource(fps=5), config=StreamConfig(host='127.0.0.1', port=0), fps=5)
    endpoint = runtime.start()
    app.state.settings.vision_stream_url = endpoint
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, lifespan='off', log_level='error', timeout_graceful_shutdown=1))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(.01)
        assert server.started
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=5) as browser:
            assert browser.get('/vision').status_code == 200
            assert browser.get('/vision/stream').status_code == 401
            assert browser.post('/vision/session', headers={'Authorization': 'Bearer test-key'}).status_code == 200
            with browser.stream('GET', '/vision/stream') as response:
                assert response.status_code == 200
                assert 'boundary=edgeframe' in response.headers['content-type']
                received = b''
                for chunk in response.iter_raw():
                    received += chunk
                    if b'\xff\xd9' in received:
                        break
                assert b'--edgeframe' in received
                assert b'Content-Type: image/jpeg' in received
                assert b'\xff\xd8' in received
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        runtime.stop()
        sock.close()
    assert not thread.is_alive()
