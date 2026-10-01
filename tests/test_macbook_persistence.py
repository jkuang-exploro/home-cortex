"""MacBook #0 is household graph data even without a running client."""
import json
from pathlib import Path

import pytest
from surrealdb import AsyncSurreal

from home_cortex.agents.registration import open_embodiment_runtime
from home_cortex.mutation.embodiments import EmbodimentWritingService
from home_cortex.persistence.export import export_directory
from home_cortex.persistence.ingestion import ingest_directory


FIXTURE = Path(__file__).parent / "static_test_data"
MAC = "embodiment:macbook-0"


class MemoryDatabase:
    def __init__(self) -> None:
        self.client = AsyncSurreal("mem://")

    async def connect(self) -> None:
        await self.client.connect()
        await self.client.use("test", "macbook")

    async def close(self) -> None:
        await self.client.close()

    async def query(self, statement, variables=None):
        return await self.client.query(statement, variables or {})

    async def upsert(self, record, data):
        return await self.client.upsert(record, data)


@pytest.mark.asyncio
async def test_macbook_offline_online_offline_restart_and_restore(tmp_path: Path) -> None:
    source = MemoryDatabase()
    restored = MemoryDatabase()
    await source.connect()
    await restored.connect()
    try:
        await ingest_directory(source, FIXTURE)
        writer = EmbodimentWritingService(source)
        runtime = await open_embodiment_runtime(writer)
        mac = await writer.get(MAC)
        assert mac.name == "MacBook"
        assert mac.embodiment_type == "computer"
        assert mac.geometry is None and mac.local_frame is None
        assert mac.capabilities == ("vision.observe",)
        assert mac.agent_id == "agent:butler"
        offline = await runtime.directory.get(MAC)
        assert offline["linked"] and not offline["connected"]
        assert offline["agent"]["name"] == "老管家"
        assert offline["geometry"] is None and offline["local_frame"] is None
        assert offline["capabilities"] == [{
            "name": "vision.observe", "supported": True, "available": False,
        }]

        opened = await runtime.registration.connect(MAC, ["vision.observe"])
        online = await runtime.directory.get(MAC)
        assert online["linked"] and online["connected"]
        assert online["capabilities"][0]["available"] is True
        await runtime.registration.disconnect(
            MAC, session_id=opened["session"]["session_id"]
        )
        assert not (await runtime.directory.get(MAC))["connected"]
        assert await writer.get(MAC) == mac

        restarted = await open_embodiment_runtime(writer)
        assert (await restarted.directory.get(MAC))["state"] == "linked_offline"
        again = await restarted.registration.connect(MAC, ["vision.observe"])
        assert again["session"]["session_id"] != opened["session"]["session_id"]
        assert await writer.get(MAC) == mac
        assert len(await source.query("SELECT * FROM embodiment;")) == 1
        assert len(await source.query("SELECT * FROM assigned_to;")) == 1

        await export_directory(source, tmp_path)
        nodes = json.loads((tmp_path / "nodes" / "embodiment.json").read_text())
        edges = json.loads((tmp_path / "edges" / "assigned_to.json").read_text())
        assert len(nodes) == len(edges) == 1
        assert nodes[0] == {
            "id": MAC, "name": "MacBook", "embodiment_type": "computer",
            "capabilities": ["vision.observe"],
        }
        assert edges[0]["from"] == MAC and edges[0]["to"] == "agent:butler"
        assert all(field not in nodes[0] for field in (
            "connected", "last_seen", "runtime_session_id", "telemetry",
        ))
        await ingest_directory(restored, tmp_path)
        recovered = await open_embodiment_runtime(EmbodimentWritingService(restored))
        assert (await recovered.directory.get(MAC))["state"] == "linked_offline"
    finally:
        await source.close()
        await restored.close()
