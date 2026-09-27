"""Automatic model residency with synthetic providers and no household data."""

import asyncio

import pytest

from home_cortex.runtime import model_warmup


class FakeModel:
    base_url = "http://model.test"
    model = "test-model"

    def __init__(self) -> None:
        self.resident = False
        self.warm_calls = 0
        self.fail = False

    async def is_resident(self) -> bool:
        return self.resident

    async def warmup(self) -> None:
        self.warm_calls += 1
        await asyncio.sleep(0.01)
        if self.fail:
            raise RuntimeError("secret provider error")
        self.resident = True


async def _until(predicate, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_warmup_deduplicates_workers_and_rewarms_after_eviction(monkeypatch, tmp_path):
    monkeypatch.setattr(model_warmup, "LOCK_DIRECTORY", tmp_path)
    monkeypatch.setattr(model_warmup, "RESIDENCY_CHECK_SECONDS", 0.02)
    provider = FakeModel()
    first = model_warmup.ModelWarmup([provider, provider])
    second = model_warmup.ModelWarmup([provider])
    first.start()
    second.start()
    try:
        await _until(lambda: first.snapshot()['status'] == second.snapshot()['status'] == 'warm')
        assert provider.warm_calls == 1
        provider.resident = False
        await _until(lambda: provider.warm_calls == 2 and first.snapshot()['status'] == 'warm')
        assert first.snapshot()['models'][0]['ready_ms'] >= 0
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_warmup_failure_is_visible_and_shutdown_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr(model_warmup, "LOCK_DIRECTORY", tmp_path)
    provider = FakeModel()
    provider.fail = True
    warmup = model_warmup.ModelWarmup([provider])
    warmup.start()
    try:
        await _until(lambda: warmup.snapshot()['status'] == 'failed')
        model = warmup.snapshot()['models'][0]
        assert model['last_error_type'] == 'RuntimeError'
        assert 'secret' not in str(warmup.snapshot())
    finally:
        await asyncio.wait_for(warmup.close(), 0.2)


@pytest.mark.asyncio
async def test_warmup_timeout_does_not_block_startup(monkeypatch, tmp_path):
    monkeypatch.setattr(model_warmup, "LOCK_DIRECTORY", tmp_path)
    monkeypatch.setattr(model_warmup, "WARMUP_TIMEOUT_SECONDS", 0.02)

    class StalledModel(FakeModel):
        async def warmup(self) -> None:
            self.warm_calls += 1
            await asyncio.sleep(10)

    provider = StalledModel()
    warmup = model_warmup.ModelWarmup([provider])
    warmup.start()
    try:
        await _until(lambda: warmup.snapshot()["status"] == "failed")
        assert warmup.snapshot()["models"][0]["last_error_type"] == "TimeoutError"
        assert provider.warm_calls == 1
    finally:
        await asyncio.wait_for(warmup.close(), 0.2)
