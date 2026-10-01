"""Runtime registration resolves a SurrealDB embodiment and does not assign it."""
from contextlib import asynccontextmanager
from dataclasses import fields
from datetime import datetime, timezone

import pytest
from surrealdb import AsyncSurreal, RecordID

from home_cortex.agents.embodiments import RuntimeSession, SessionProtocolError
from home_cortex.agents.registration import open_embodiment_runtime
from home_cortex.mutation.embodiments import EmbodimentWritingService


MAC = "embodiment:macbook-0"
NOW = datetime(2026, 9, 30, 16, 0, tzinfo=timezone.utc)


class MemoryDatabase:
    def __init__(self) -> None:
        self.client = AsyncSurreal("mem://")

    async def connect(self) -> None:
        await self.client.connect()
        await self.client.use("test", "registration")

    async def close(self) -> None:
        await self.client.close()

    async def query(self, statement, variables=None):
        return await self.client.query(statement, variables or {})

    async def upsert(self, record, data):
        return await self.client.upsert(record, data)


def macbook() -> dict:
    return {
        "id": MAC, "name": "MacBook", "embodiment_type": "computer",
        "geometry": {"box": {"length_m": 0.31, "width_m": 0.22, "height_m": 0.016,
                             "center": {"x": 0.0, "y": 0.0, "z": 0.008}}},
        "local_frame": {"forward": "+x", "left": "+y", "up": "+z"},
        "capabilities": ["audio.speak", "vision.observe"],
    }


@asynccontextmanager
async def started_runtime():
    database = MemoryDatabase()
    await database.connect()
    try:
        writing = EmbodimentWritingService(database)
        runtime = await open_embodiment_runtime(writing, space_ids=["space:kitchen"])
        yield database, writing, runtime
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_startup_does_not_require_an_online_embodiment() -> None:
    async with started_runtime() as (database, writing, runtime):
        assert runtime.catalog.embodiment_ids == frozenset()
        assert await writing.list() == ()
        assert await database.query("SELECT * FROM assigned_to;") == []
        assert not runtime.connections.is_currently_embodied("agent:butler")
        with pytest.raises(SessionProtocolError, match="unknown embodiment") as error:
            await runtime.registration.connect("embodiment:unknown-device", [], now=NOW)
        assert error.value.code == "unknown_embodiment"
        assert await writing.list() == ()
        assert await database.query("SELECT * FROM assigned_to;") == []


@pytest.mark.asyncio
async def test_known_embodiment_connects_without_writing_identity() -> None:
    async with started_runtime() as (_, writing, runtime):
        await writing.create(macbook())
        await writing.assign(MAC, "agent:butler")
        stored = await writing.get(MAC)
        opened = await runtime.registration.connect(MAC, ["audio.speak"], now=NOW)
        assert opened["disposition"] == "connected"
        assert opened["embodiment_id"] == MAC
        assert opened["agent_id"] == "agent:butler"
        assert opened["session"]["online"] is True
        assert "agent_id" not in opened["session"]
        assert "vision.observe" in opened["configured_capabilities"]
        assert opened["session"]["available_capabilities"] == ["audio.speak"]
        assert "agent_id" not in {item.name for item in fields(RuntimeSession)}
        state = runtime.presence.latest(MAC, now=NOW)
        assert state.telemetry is None and state.status.unavailable
        posted = await runtime.registration.submit_telemetry({
            "embodiment_id": MAC, "space_id": None, "measured_at": NOW.isoformat(),
            "validity": "no_estimate", "transform": None,
        }, session_id=opened["session"]["session_id"], now=NOW)
        assert posted["presence"]["session"]["online"] is True
        assert runtime.presence.latest(MAC, now=NOW).status.unavailable
        assert await writing.get(MAC) == stored
        assert len(await writing.list()) == 1


@pytest.mark.asyncio
async def test_unknown_embodiment_is_rejected() -> None:
    async with started_runtime() as (database, writing, runtime):
        await writing.create(macbook())
        with pytest.raises(SessionProtocolError, match="unknown embodiment") as error:
            await runtime.registration.connect(
                "embodiment:unknown-device", ["vision.observe"], now=NOW,
            )
        assert error.value.code == "unknown_embodiment"
        assert [body.id for body in await writing.list()] == [MAC]
        assert await database.query("SELECT * FROM assigned_to;") == []
        assert (await writing.get(MAC)).agent_id is None


@pytest.mark.asyncio
async def test_runtime_cannot_create_or_overwrite_assignment() -> None:
    async with started_runtime() as (database, writing, runtime):
        await database.upsert(RecordID("agent", "other"), {"name": {"en": "Other"}})
        await writing.create(macbook())
        await runtime.registration.connect(MAC, [], now=NOW)
        assert (await writing.get(MAC)).agent_id is None
        assert await database.query("SELECT * FROM assigned_to;") == []

        await writing.assign(MAC, "agent:other")
        assigned = await writing.get(MAC)
        await runtime.registration.connect(MAC, ["vision.observe"], now=NOW)
        assert await writing.get(MAC) == assigned
        edges = await database.query("SELECT * FROM assigned_to;")
        assert len(edges) == 1


@pytest.mark.asyncio
async def test_disconnect_and_reconnect_preserve_the_same_semantic_entity() -> None:
    async with started_runtime() as (_, writing, runtime):
        await writing.create(macbook())
        await writing.assign(MAC, "agent:butler")
        stored = await writing.get(MAC)
        first = await runtime.registration.connect(MAC, ["vision.observe"], now=NOW)
        offline = await runtime.registration.disconnect(
            MAC, session_id=first["session"]["session_id"], now=NOW,
        )
        assert offline["session"]["online"] is False
        assert await writing.get(MAC) == stored
        second = await runtime.registration.connect(MAC, ["audio.speak"], now=NOW)
        assert second["disposition"] == "connected"
        assert second["session"]["online"] is True
        assert second["embodiment_id"] == first["embodiment_id"] == MAC
        assert second["agent_id"] == "agent:butler"
        assert second["session"]["session_id"] != first["session"]["session_id"]
        assert await writing.get(MAC) == stored
        current = second
        for _ in range(3):
            await runtime.registration.disconnect(
                MAC, session_id=current["session"]["session_id"], now=NOW,
            )
            current = await runtime.registration.connect(MAC, ["vision.observe"], now=NOW)
        assert [body.id for body in await writing.list()] == [MAC]
        assert await writing.get(MAC) == stored
