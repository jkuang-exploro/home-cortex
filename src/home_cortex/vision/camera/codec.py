"""Bounded MPEG-TS to multipart MJPEG conversion owned by cortex-api."""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import AsyncIterator
from typing import Protocol

from .errors import failed, timed_out, unsupported

IDLE_TIMEOUT = 10.0
FIRST_FRAME_TIMEOUT = 15.0


class MpegTsToMjpeg(Protocol):
    def transcode(self, packets: AsyncIterator[bytes]) -> AsyncIterator[bytes]: ...


class FfmpegMjpegAdapter:
    def __init__(self, ffmpeg: str | None = None) -> None:
        self.ffmpeg = ffmpeg or shutil.which("ffmpeg") or "ffmpeg"

    async def transcode(self, packets: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        try:
            process = await asyncio.create_subprocess_exec(
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-fflags",
                "nobuffer",
                "-flags",
                "low_delay",
                "-f",
                "mpegts",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "mjpeg",
                "-q:v",
                "5",
                "-f",
                "mpjpeg",
                "pipe:1",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as error:
            raise unsupported() from error
        assert process.stdin is not None and process.stdout is not None

        async def feed() -> None:
            try:
                async for packet in packets:
                    process.stdin.write(packet)
                    await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                return
            finally:
                try:
                    process.stdin.close()
                except Exception:
                    return

        feeder = asyncio.create_task(feed())
        timeout = FIRST_FRAME_TIMEOUT
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(process.stdout.read(65536), timeout=timeout)
                except TimeoutError as error:
                    raise timed_out() from error
                if not chunk:
                    break
                timeout = IDLE_TIMEOUT
                yield chunk
        finally:
            feeder.cancel()
            await _stop_process(process)


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        process.terminate()
        await asyncio.wait_for(process.wait(), timeout=1.0)
    except (ProcessLookupError, TimeoutError):
        try:
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except (ProcessLookupError, TimeoutError):
            if process.pid:
                with suppress_os_error():
                    os.killpg(process.pid, 9)


class suppress_os_error:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return True


class ScriptedMjpegAdapter:
    """Deterministic test double: wrap TS bytes as multipart JPEG parts."""

    boundary = "frame"

    async def transcode(self, packets: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        async for packet in packets:
            payload = b"\xff\xd8" + packet[:32] + b"\xff\xd9"
            yield (
                f"--{self.boundary}\r\nContent-Type: image/jpeg\r\n"
                f"Content-Length: {len(payload)}\r\n\r\n"
            ).encode() + payload + b"\r\n"
