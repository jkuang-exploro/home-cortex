"""Persistent body identity and assignment use the same graph as household facts."""
import json
from pathlib import Path

import pytest
from surrealdb import AsyncSurreal, RecordID

from home_cortex.agents.embodiments import EmbodimentConnections
from home_cortex.mutation.embodiments import EmbodimentWritingService
from home_cortex.persistence.export import export_directory
from home_cortex.persistence.ingestion import ingest_directory
from home_cortex.persistence.retrieval import RetrievalService
from home_cortex.spatial.primitives import SpatialContractError


class MemoryDatabase:
    def __init__(self) -> None:
        self.client = AsyncSurreal("mem://")

    async def connect(self) -> None:
        await self.client.connect()
        await self.client.use("test", "embodiments")

    async def close(self) -> None:
        await self.client.close()

    async def query(self, statement, variables=None):
        return await self.client.query(statement, variables or {})

    async def upsert(self, record, data):
        return await self.client.upsert(record, data)


def body(key: str, *, name: str = "Duck", kind: str = "robot") -> dict:
    return {
        "id": f"embodiment:{key}", "name": name, "embodiment_type": kind,
        "geometry": {"box": {"length_m": 0.3, "width_m": 0.2, "height_m": 0.1,
                             "center": {"x": 0, "y": 0, "z": 0.05}}},
        "local_frame": {"forward": "+x", "left": "+y", "up": "+z"},
        "capabilities": ["mobility.move", "vision.observe"],
    }


@pytest.mark.asyncio
async def test_crud_and_assignment_survive_runtime_disconnect() -> None:
    db = MemoryDatabase()
    await db.connect()
    try:
        writer = EmbodimentWritingService(db)
        await writer.ensure_constraints()
        await writer.ensure_registered_agents()
        await db.upsert(RecordID("agent", "other"), {"name": {"en": "Other"}})
        await writer.create(body("duck"))
        await writer.create(body("humanoid", name="Humanoid"))
        linked = body("embedded")
        linked["agent_id"] = "agent:butler"
        with pytest.raises(SpatialContractError, match="assignment must use assign"):
            await writer.create(linked)
        untyped = body("untyped")
        del untyped["embodiment_type"]
        with pytest.raises(SpatialContractError, match="embodiment_type is required"):
            await writer.create(untyped)
        with pytest.raises(SpatialContractError, match="already exists"):
            await writer.create(body("duck"))
        with pytest.raises(SpatialContractError, match="unknown controlling agent"):
            await writer.assign("embodiment:duck", "agent:missing")
        assert (await writer.get("embodiment:duck")).embodiment_type == "robot"
        await writer.update(body("duck", name="New Duck", kind="computer"))
        assert (await writer.get("embodiment:duck")).name == "New Duck"
        assert (await writer.get("embodiment:duck")).embodiment_type == "computer"
        assert (await writer.get("embodiment:duck")).capabilities == (
            "mobility.move", "vision.observe"
        )
        await writer.assign("embodiment:duck", "agent:butler")
        await writer.assign("embodiment:humanoid", "agent:butler")
        retrieval = RetrievalService(db)
        assert (await retrieval.get_entity("embodiment:duck"))["name"] == "New Duck"
        assert (await retrieval.get_entity("agent:butler"))["agent_type"] == "steward"
        embodied = await retrieval.get_relationships("agent:butler", "embodied_by")
        assert {edge["semantic_relation"] for edge in embodied} == {"embodied_by"}
        assert {edge["related_entity"]["id"] for edge in embodied} == {
            "embodiment:duck", "embodiment:humanoid",
        }
        assert (await writer.catalog()).bodies_for_agent("agent:butler")
        with pytest.raises(SpatialContractError, match="already assigned"):
            await writer.assign("embodiment:duck", "agent:other")
        with pytest.raises(Exception, match="assigned_to_one_agent"):
            await db.query(
                "RELATE embodiment:duck->assigned_to:rogue->agent:other;"
            )
        assert (await writer.get("embodiment:duck")).agent_id == "agent:butler"

        # A new service instance sees the graph even before a client connects.
        reloaded = EmbodimentWritingService(db)
        catalog = await reloaded.catalog()
        sessions = EmbodimentConnections(catalog)
        assert catalog.get("embodiment:duck").agent_id == "agent:butler"
        assert not sessions.is_currently_embodied("agent:butler")
        sessions.connect("embodiment:duck")
        sessions.disconnect("embodiment:duck")
        assert (await reloaded.get("embodiment:duck")).agent_id == "agent:butler"
        await writer.unassign("embodiment:duck")
        assert (await writer.get("embodiment:duck")).agent_id is None
        await writer.assign("embodiment:duck", "agent:other")
        await writer.delete("embodiment:duck")
        with pytest.raises(SpatialContractError, match="unknown embodiment"):
            await writer.get("embodiment:duck")
        assert await db.query("SELECT * FROM assigned_to WHERE in = embodiment:duck;") == []
        assert (await writer.get("embodiment:humanoid")).agent_id == "agent:butler"
        with pytest.raises(SpatialContractError, match="unknown embodiment"):
            await writer.update(body("missing"))
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_export_ingest_round_trip_preserves_body_and_assignment(tmp_path: Path) -> None:
    source = MemoryDatabase()
    restored = MemoryDatabase()
    await source.connect()
    await restored.connect()
    try:
        writer = EmbodimentWritingService(source)
        await writer.ensure_registered_agents()
        await writer.create(body("duck"))
        await writer.assign("embodiment:duck", "agent:butler")
        await export_directory(source, tmp_path)
        node = json.loads((tmp_path / "nodes" / "embodiment.json").read_text())[0]
        edge = json.loads((tmp_path / "edges" / "assigned_to.json").read_text())[0]
        assert node["embodiment_type"] == "robot"
        assert node["geometry"]["box"]["length_m"] == 0.3
        assert node["capabilities"] == ["mobility.move", "vision.observe"]
        assert "agent_id" not in node
        assert edge["from"] == "embodiment:duck"
        assert edge["to"] == "agent:butler"
        await ingest_directory(restored, tmp_path)
        assert await EmbodimentWritingService(restored).list() == await writer.list()
    finally:
        await source.close()
        await restored.close()


@pytest.mark.asyncio
async def test_ingest_rejects_two_agents_for_one_body_before_writing(tmp_path: Path) -> None:
    (tmp_path / "nodes").mkdir()
    (tmp_path / "edges").mkdir()
    (tmp_path / "nodes" / "embodiment.json").write_text(json.dumps([body("duck")]))
    (tmp_path / "nodes" / "agent.json").write_text(json.dumps([
        {"id": "agent:butler", "name": {"en": "Butler"}},
        {"id": "agent:other", "name": {"en": "Other"}},
    ]))
    (tmp_path / "edges" / "assigned_to.json").write_text(json.dumps([
        {"from": "embodiment:duck", "to": "agent:butler"},
        {"from": "embodiment:duck", "to": "agent:other"},
    ]))
    db = MemoryDatabase()
    await db.connect()
    try:
        with pytest.raises(ValueError, match="only one target"):
            await ingest_directory(db, tmp_path)
        assert await EmbodimentWritingService(db).list() == ()
    finally:
        await db.close()
