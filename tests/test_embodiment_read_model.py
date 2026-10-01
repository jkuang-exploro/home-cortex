"""Embodiment views join a live SurrealDB record to optional runtime state."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
from surrealdb import AsyncSurreal

from home_cortex.agents.registration import open_embodiment_runtime
from home_cortex.mutation.embodiments import EmbodimentWritingService
from home_cortex.spatial.primitives import SpatialContractError


MAC = "embodiment:macbook-0"
PHONE = "embodiment:phone-0"
NOW = datetime(2026, 9, 30, 18, 0, tzinfo=timezone.utc)


class MemoryDatabase:
    def __init__(self) -> None:
        self.client = AsyncSurreal("mem://")

    async def connect(self) -> None:
        await self.client.connect()
        await self.client.use("test", "read-model")

    async def close(self) -> None:
        await self.client.close()

    async def query(self, statement, variables=None):
        return await self.client.query(statement, variables or {})


def _body(embodiment_id: str, name: str, capabilities: list[str]) -> dict:
    return {
        "id": embodiment_id, "name": name, "embodiment_type": "computer",
        "capabilities": capabilities,
    }


@asynccontextmanager
async def started():
    database = MemoryDatabase()
    await database.connect()
    try:
        writing = EmbodimentWritingService(database)
        runtime = await open_embodiment_runtime(writing, space_ids=["space:kitchen"])
        yield database, writing, runtime
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_surrealdb_view_keeps_assignment_and_session_apart() -> None:
    async with started() as (database, writing, runtime):
        assert runtime.catalog.embodiment_ids == frozenset()
        await writing.create(_body(MAC, "MacBook", ["audio.speak", "vision.observe"]))
        await writing.create(_body(PHONE, "Phone", ["mobility.move"]))
        listed = await runtime.directory.list()
        assert [item["id"] for item in listed] == [MAC, PHONE]
        assert all(not item["connected"] and not item["linked"] for item in listed)

        await writing.assign(MAC, "agent:butler")
        stored = await writing.get(MAC)
        offline = await runtime.directory.get(MAC)
        assert offline["linked"] is True
        assert offline["connected"] is False
        assert offline["state"] == "linked_offline"
        assert offline["agent"] == {"id": "agent:butler", "name": "老管家"}
        assert offline["embodiment_type"] == "computer"
        assert offline["geometry"] is None
        assert offline["local_frame"] is None
        assert offline["telemetry"] == {
            "available": False, "valid": False, "fresh": False,
            "space_id": None, "measured_at": None, "estimate": None,
        }
        await runtime.directory.validate_selection("agent:butler", MAC)
        assert (await runtime.directory.action_availability(
            "agent:butler", MAC, "vision.observe")).code == "embodiment_offline"
        assert await writing.get(MAC) == stored

        opened = await runtime.registration.connect(MAC, ["audio.speak"], now=NOW)
        await runtime.registration.submit_telemetry({
            "embodiment_id": MAC, "space_id": None, "measured_at": NOW.isoformat(),
            "validity": "no_estimate", "transform": None,
        }, session_id=opened["session"]["session_id"], now=NOW)
        online = await runtime.directory.get(MAC)
        assert online["linked"] is True and online["connected"] is True
        assert online["telemetry"]["available"] is False
        capabilities = {item["name"]: item for item in online["capabilities"]}
        assert capabilities["vision.observe"] == {
            "name": "vision.observe", "supported": True, "available": False,
        }
        assert capabilities["audio.speak"]["available"] is True
        assert "session_id" not in online["runtime"]
        assert (await runtime.directory.action_availability(
            "agent:butler", MAC, "vision.observe")).code == "capability_unavailable"
        assert (await runtime.directory.action_availability(
            "agent:butler", MAC, "audio.speak")).available
        assert (await runtime.directory.action_availability(
            "agent:butler", MAC, "mobility.move")).code == "unsupported_capability"
        assert await writing.get(MAC) == stored

        await runtime.registration.connect(PHONE, [], now=NOW)
        unlinked = await runtime.directory.get(PHONE)
        assert unlinked["linked"] is False and unlinked["connected"] is True
        assert unlinked["agent"] is None
        with pytest.raises(SpatialContractError, match="not linked"):
            await runtime.directory.validate_selection("agent:butler", PHONE)
        assert (await runtime.directory.action_availability(
            "agent:butler", PHONE, "mobility.move")).code == "embodiment_not_linked"
        assert (await writing.get(PHONE)).agent_id is None
        assert len(await database.query("SELECT * FROM assigned_to;")) == 1

        with pytest.raises(SpatialContractError, match="unknown embodiment"):
            await runtime.directory.get("embodiment:unknown-device")
        assert (await runtime.directory.action_availability(
            "agent:butler", "embodiment:unknown-device", "vision.observe",
        )).code == "unknown_embodiment"
        assert [body.id for body in await writing.list()] == [MAC, PHONE]

        restarted = await open_embodiment_runtime(writing, space_ids=["space:kitchen"])
        revived = await restarted.directory.get(MAC)
        assert revived["linked"] is True
        assert revived["connected"] is False
        assert revived["agent"]["id"] == "agent:butler"
        await restarted.directory.validate_selection("agent:butler", MAC)
        restarted_list = await restarted.directory.list()
        assert [item["id"] for item in restarted_list] == [MAC, PHONE]
        assert restarted_list[0]["linked"] is True and restarted_list[0]["connected"] is False
        assert restarted_list[1]["linked"] is False and restarted_list[1]["connected"] is False
        assert await writing.get(MAC) == stored
        assert len(await database.query("SELECT * FROM assigned_to;")) == 1
