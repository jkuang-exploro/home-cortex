"""Conversation selection persists independently of transient body sessions."""
from unittest.mock import AsyncMock

import pytest

from home_cortex.conversation.store import SurrealConversationStore


@pytest.mark.asyncio
async def test_surreal_store_keeps_selection_across_message_updates() -> None:
    database = type("DatabaseStub", (), {"upsert": AsyncMock()})()
    store = SurrealConversationStore(database)
    record = {
        "id": "chat", "agent_id": "steward", "model": "老管家",
        "person_id": "person:owner", "language": "en", "greeting": None,
        "title": "", "created_at": "2026-09-28T00:00:00+00:00",
        "updated_at": "2026-09-28T00:00:00+00:00",
        "active_embodiment_id": None,
    }

    async def load(_id: str):
        return dict(record)

    async def upsert(_id, data):
        record.update(data)

    store._load = load
    store.get = AsyncMock(side_effect=lambda _id: dict(record))
    store._write_message = AsyncMock()
    store._has_user_message = AsyncMock(return_value=False)
    database.upsert.side_effect = upsert

    selected = await store.set_active_embodiment("chat", "embodiment:duck")
    assert selected["active_embodiment_id"] == "embodiment:duck"
    assert record["agent_id"] == "steward"
    await store.append_message("chat", role="user", content="Hello")
    assert record["active_embodiment_id"] == "embodiment:duck"
    cleared = await store.set_active_embodiment("chat", None)
    assert cleared["active_embodiment_id"] is None
    assert record["agent_id"] == "steward"
