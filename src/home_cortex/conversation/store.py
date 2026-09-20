"""GUI transcript store. Not household graph truth; RetrievalService is unused."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ..persistence.db import Database
from ..persistence.record_ids import as_record_id
from ..persistence.retrieval import to_json_value

TITLE_MAX = 80
CONVERSATION_TABLE = "gui_conversation"
MESSAGE_TABLE = "gui_message"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def conversation_title(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= TITLE_MAX:
        return collapsed
    return collapsed[: TITLE_MAX - 1].rstrip() + "…"


def new_message(role: str, content: str, *, created_at: str | None = None) -> dict[str, Any]:
    return {
        "id": uuid4().hex,
        "role": role,
        "content": content,
        "created_at": created_at or utc_now(),
    }


class ConversationStore:
    """Bounded in-memory transcripts. Used by tests and as the store interface."""

    def __init__(self, maximum: int = 1_000) -> None:
        self.maximum = maximum
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()

    async def create(
        self,
        *,
        agent_id: str | None,
        model: str,
        person_id: str | None,
        language: str,
        greeting: str | None = None,
    ) -> dict[str, Any]:
        created = utc_now()
        messages = []
        if greeting:
            messages.append(new_message("assistant", greeting, created_at=created))
        conversation = {
            "id": uuid4().hex,
            "agent_id": agent_id,
            "model": model,
            "person_id": person_id,
            "language": language,
            "greeting": greeting,
            "title": greeting or "",
            "created_at": created,
            "updated_at": created,
            "messages": messages,
        }
        self._items[conversation["id"]] = conversation
        while len(self._items) > self.maximum:
            self._items.popitem(last=False)
        return self._public(conversation)

    async def get(self, conversation_id: str) -> dict[str, Any] | None:
        conversation = self._items.get(conversation_id)
        return self._public(conversation) if conversation is not None else None

    async def list_for_person(self, person_id: str | None) -> list[dict[str, Any]]:
        items = [
            self._summary(item)
            for item in self._items.values()
            if item["person_id"] == person_id
        ]
        items.sort(key=lambda item: item["updated_at"], reverse=True)
        return items

    async def delete(self, conversation_id: str) -> bool:
        return self._items.pop(conversation_id, None) is not None

    async def append_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        first_user: bool | None = None,
    ) -> dict[str, Any] | None:
        conversation = self._items.get(conversation_id)
        if conversation is None:
            return None
        message = new_message(role, content)
        conversation["messages"].append(message)
        conversation["updated_at"] = message["created_at"]
        is_first_user = first_user if first_user is not None else not any(
            item["role"] == "user" for item in conversation["messages"][:-1]
        )
        if role == "user" and is_first_user:
            conversation["title"] = conversation_title(content)
        self._items.move_to_end(conversation_id)
        return dict(message)

    def _public(self, conversation: dict[str, Any]) -> dict[str, Any]:
        copied = deepcopy(conversation)
        copied["messages"] = [dict(item) for item in copied["messages"]]
        return copied

    def _summary(self, conversation: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": conversation["id"],
            "object": "conversation",
            "agent_id": conversation["agent_id"],
            "model": conversation["model"],
            "language": conversation["language"],
            "greeting": conversation["greeting"],
            "title": conversation["title"],
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
        }


class SurrealConversationStore:
    """Durable transcripts in dedicated Surreal tables, not graph node/edge tables."""

    def __init__(self, database: Database, maximum: int = 1_000) -> None:
        self.database = database
        self.maximum = maximum

    async def create(
        self,
        *,
        agent_id: str | None,
        model: str,
        person_id: str | None,
        language: str,
        greeting: str | None = None,
    ) -> dict[str, Any]:
        created = utc_now()
        conversation_id = uuid4().hex
        conversation = {
            "id": conversation_id,
            "agent_id": agent_id,
            "model": model,
            "person_id": person_id,
            "language": language,
            "greeting": greeting,
            "title": greeting or "",
            "created_at": created,
            "updated_at": created,
        }
        await self.database.upsert(
            as_record_id(f"{CONVERSATION_TABLE}:{conversation_id}"),
            {key: value for key, value in conversation.items() if key != "id"},
        )
        messages: list[dict[str, Any]] = []
        if greeting:
            message = new_message("assistant", greeting, created_at=created)
            await self._write_message(conversation_id, message)
            messages.append(message)
        conversation["messages"] = messages
        return conversation

    async def get(self, conversation_id: str) -> dict[str, Any] | None:
        conversation = await self._load(conversation_id)
        if conversation is None:
            return None
        conversation["messages"] = await self._load_messages(conversation_id)
        return conversation

    async def list_for_person(self, person_id: str | None) -> list[dict[str, Any]]:
        result = await self.database.query(
            f"SELECT * FROM {CONVERSATION_TABLE} WHERE person_id = $person_id "
            "ORDER BY updated_at DESC;",
            {"person_id": person_id},
        )
        return [self._summary(record) for record in _records(result)]

    async def delete(self, conversation_id: str) -> bool:
        existing = await self._load(conversation_id)
        if existing is None:
            return False
        await self.database.query(
            f"DELETE {MESSAGE_TABLE} WHERE conversation_id = $conversation_id;",
            {"conversation_id": conversation_id},
        )
        await self.database.query(
            "DELETE $id;",
            {"id": as_record_id(f"{CONVERSATION_TABLE}:{conversation_id}")},
        )
        return True

    async def append_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        first_user: bool | None = None,
    ) -> dict[str, Any] | None:
        conversation = await self._load(conversation_id)
        if conversation is None:
            return None
        first_user_message = role == "user" and (
            first_user
            if first_user is not None
            else not await self._has_user_message(conversation_id)
        )
        message = new_message(role, content)
        await self._write_message(conversation_id, message)
        title = conversation["title"]
        if first_user_message:
            title = conversation_title(content)
        await self.database.upsert(
            as_record_id(f"{CONVERSATION_TABLE}:{conversation_id}"),
            {
                "agent_id": conversation["agent_id"],
                "model": conversation["model"],
                "person_id": conversation["person_id"],
                "language": conversation["language"],
                "greeting": conversation["greeting"],
                "title": title,
                "created_at": conversation["created_at"],
                "updated_at": message["created_at"],
            },
        )
        return dict(message)

    async def _has_user_message(self, conversation_id: str) -> bool:
        result = await self.database.query(
            f"SELECT VALUE count() FROM {MESSAGE_TABLE} "
            "WHERE conversation_id = $conversation_id AND role = 'user' GROUP ALL;",
            {"conversation_id": conversation_id},
        )
        normalized = to_json_value(result)
        while isinstance(normalized, list) and len(normalized) == 1:
            normalized = normalized[0]
        return isinstance(normalized, (int, float)) and normalized > 0

    async def _load(self, conversation_id: str) -> dict[str, Any] | None:
        result = await self.database.query(
            "SELECT * FROM $id;",
            {"id": as_record_id(f"{CONVERSATION_TABLE}:{conversation_id}")},
        )
        records = _records(result)
        if not records:
            return None
        return self._from_record(records[0], conversation_id)

    async def _load_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        result = await self.database.query(
            f"SELECT * FROM {MESSAGE_TABLE} WHERE conversation_id = $conversation_id "
            "ORDER BY created_at ASC;",
            {"conversation_id": conversation_id},
        )
        messages = []
        for record in _records(result):
            messages.append(
                {
                    "id": str(record.get("message_id") or record.get("id")),
                    "role": record["role"],
                    "content": record["content"],
                    "created_at": record["created_at"],
                }
            )
        return messages

    async def _write_message(self, conversation_id: str, message: dict[str, Any]) -> None:
        await self.database.upsert(
            as_record_id(f"{MESSAGE_TABLE}:{message['id']}"),
            {
                "conversation_id": conversation_id,
                "message_id": message["id"],
                "role": message["role"],
                "content": message["content"],
                "created_at": message["created_at"],
            },
        )

    def _from_record(self, record: dict[str, Any], conversation_id: str) -> dict[str, Any]:
        return {
            "id": conversation_id,
            "agent_id": record.get("agent_id"),
            "model": record.get("model"),
            "person_id": record.get("person_id"),
            "language": record.get("language"),
            "greeting": record.get("greeting"),
            "title": record.get("title") or "",
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
        }

    def _summary(self, record: dict[str, Any]) -> dict[str, Any]:
        record_id = record.get("id")
        conversation_id = str(record_id).split(":", 1)[-1] if record_id else ""
        public = self._from_record(record, conversation_id)
        public["object"] = "conversation"
        return public


def _records(value: Any) -> list[dict[str, Any]]:
    normalized = to_json_value(value)
    if normalized is None:
        return []
    if isinstance(normalized, dict):
        return [normalized]
    if isinstance(normalized, list):
        records: list[dict[str, Any]] = []
        for item in normalized:
            if isinstance(item, dict):
                records.append(item)
            elif isinstance(item, list):
                records.extend(_records(item))
        return records
    return []
