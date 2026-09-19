"""Share one Tapo upstream among concurrent Vision viewers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

from .codec import FfmpegMjpegAdapter, MpegTsToMjpeg
from .sources import CameraSource
from .tapo.client import TapoClient

MJPEG_TYPE = "multipart/x-mixed-replace; boundary=ffmpeg"


class SharedTapoHub:
    def __init__(
        self,
        *,
        client_factory: Callable[[CameraSource], TapoClient] | None = None,
        codec_factory: Callable[[], MpegTsToMjpeg] | None = None,
    ) -> None:
        self._client_factory = client_factory or TapoClient
        self._codec_factory = codec_factory or FfmpegMjpegAdapter
        self._lock = asyncio.Lock()
        self._source: CameraSource | None = None
        self._queues: set[asyncio.Queue[bytes | None]] = set()
        self._pump: asyncio.Task[None] | None = None
        self._client: TapoClient | None = None

    async def subscribe(self, source: CameraSource) -> AsyncIterator[bytes]:
        queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=8)
        async with self._lock:
            if self._pump is None or self._source != source:
                await self._stop()
                self._queues.add(queue)
                await self._start(source)
            else:
                self._queues.add(queue)
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    return
                yield chunk
        finally:
            async with self._lock:
                self._queues.discard(queue)
                if not self._queues:
                    await self._stop()

    async def _start(self, source: CameraSource) -> None:
        client = self._client_factory(source)
        await client.open()
        self._client = client
        self._source = source
        self._pump = asyncio.create_task(self._run(client))

    async def _run(self, client: TapoClient) -> None:
        codec = self._codec_factory()
        try:
            async for chunk in codec.transcode(client.iter_mpegts()):
                await self._broadcast(chunk)
        except Exception:
            await self._broadcast(None)
        finally:
            await client.close()

    async def _broadcast(self, chunk: bytes | None) -> None:
        for queue in list(self._queues):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(chunk)
            except asyncio.QueueFull:
                continue

    async def _stop(self) -> None:
        pump = self._pump
        self._pump = None
        self._source = None
        if pump is not None:
            pump.cancel()
            try:
                await pump
            except (asyncio.CancelledError, Exception):
                pass
        if self._client is not None:
            await self._client.close()
            self._client = None
        for queue in list(self._queues):
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                continue

    async def close(self) -> None:
        async with self._lock:
            await self._stop()
