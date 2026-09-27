"""Bounded background residency checks for configured local language models."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol


WARMUP_TIMEOUT_SECONDS = 90.0
RESIDENCY_CHECK_SECONDS = 15.0
LOCK_DIRECTORY = Path("/tmp/home-cortex-model-warmup")


class WarmableModel(Protocol):
    base_url: str
    model: str

    async def is_resident(self) -> bool: ...

    async def warmup(self) -> None: ...


@dataclass
class _ModelState:
    provider: WarmableModel
    status: str = "not_started"
    attempts: int = 0
    ready_ms: float | None = None
    last_error_type: str | None = None
    task: asyncio.Task[None] | None = None


class ModelWarmup:
    """Warm each local model once and recheck residency after runtime eviction."""

    def __init__(self, providers: Sequence[WarmableModel]) -> None:
        unique: dict[tuple[str, str, str], _ModelState] = {}
        for provider in providers:
            key = (type(provider).__name__, provider.base_url, provider.model)
            unique.setdefault(key, _ModelState(provider))
        self._models = tuple(unique.values())

    def start(self) -> None:
        for state in self._models:
            if state.task is None:
                state.task = asyncio.create_task(self._run(state))

    async def close(self) -> None:
        tasks = [state.task for state in self._models if state.task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def snapshot(self) -> dict[str, object]:
        items = [
            {
                "model": state.provider.model,
                "status": state.status,
                "attempts": state.attempts,
                "ready_ms": state.ready_ms,
                "last_error_type": state.last_error_type,
            }
            for state in self._models
        ]
        return {
            "status": (
                "disabled" if not items else
                "warm" if all(item["status"] == "warm" for item in items) else
                "failed" if any(item["status"] == "failed" for item in items) else
                "warming"
            ),
            "models": items,
        }

    async def _run(self, state: _ModelState) -> None:
        failures = 0
        while True:
            if state.status == "warm":
                try:
                    await asyncio.sleep(RESIDENCY_CHECK_SECONDS)
                    if await state.provider.is_resident():
                        continue
                    state.status = "not_started"
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    state.status = "failed"
                    state.last_error_type = type(error).__name__
            state.status = "warming"
            started = perf_counter()
            try:
                async with _model_lock(state.provider):
                    if not await state.provider.is_resident():
                        state.attempts += 1
                        await asyncio.wait_for(
                            state.provider.warmup(), WARMUP_TIMEOUT_SECONDS
                        )
                        if not await state.provider.is_resident():
                            raise RuntimeError("Model residency was not verified")
                state.ready_ms = (perf_counter() - started) * 1000
                state.last_error_type = None
                state.status = "warm"
                failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failures += 1
                state.status = "failed"
                state.last_error_type = type(error).__name__
                state.ready_ms = None
                await asyncio.sleep(min(0.5 * 2 ** min(failures - 1, 5), 10.0))


@asynccontextmanager
async def _model_lock(provider: WarmableModel) -> AsyncIterator[None]:
    """Share one warmup across API workers in the same container."""
    key = f"{type(provider).__name__}\0{provider.base_url}\0{provider.model}"
    name = hashlib.sha256(key.encode()).hexdigest()[:24] + ".lock"
    LOCK_DIRECTORY.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(LOCK_DIRECTORY / name, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        while not acquired:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                await asyncio.sleep(0.1)
        yield
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
