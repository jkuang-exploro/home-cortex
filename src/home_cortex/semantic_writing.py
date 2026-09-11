"""Resolve model-facing item names before invoking canonical mutations."""
from __future__ import annotations

import hashlib
from typing import Any

from .schema_catalog import normalize_entity_alias
from .semantic_facts import AgentRequestContext, HouseholdFactEngine, SemanticFactRequest, FactResult
from .writing import ItemWritingService
from .mutation_ir import NamedCreateItem, NamedDeleteItem, NamedWriteRequest


class NamedItemWritingService:
    def __init__(self, writing: ItemWritingService, engine: HouseholdFactEngine) -> None:
        self.writing = writing
        self.engine = engine

    async def _resolve(self, name: str, entity_type: str | None, context: AgentRequestContext) -> FactResult:
        request = SemanticFactRequest.model_validate({
            "operation": "resolve_reference",
            "subject": {"kind": "named_entity", "value": name, "entity_type": entity_type},
        })
        return (await self.engine.execute(request, context))[0]

    async def mutate(self, request: NamedWriteRequest, context: AgentRequestContext) -> dict[str, Any]:
        # Resolution and storage results stay server-side. The model receives
        # names and outcome only, never internal IDs or raw entity properties.
        def reply(status: str, reason: str | None = None) -> dict[str, Any]:
            return {"status": status, "operation": request.operation,
                    "item_name": request.item_name,
                    **({"location_name": request.location_name} if hasattr(request, "location_name") else {}),
                    "reason": reason}

        item = await self._resolve(request.item_name, "item", context)
        if item.status not in {"found", "entity_not_found"}:
            return reply("CONFLICT" if item.status == "ambiguous" else "REJECTED", "ITEM_" + item.status.upper())
        if item.status == "entity_not_found" and request.operation != "create":
            return reply("NOT_FOUND", "ITEM_NOT_FOUND")

        location_id = None
        if not isinstance(request, NamedDeleteItem):
            location = await self._resolve(request.location_name, None, context)
            if location.status != "found":
                status = "CONFLICT" if location.status == "ambiguous" else "NOT_FOUND" if location.status == "entity_not_found" else "REJECTED"
                return reply(status, "LOCATION_" + location.status.upper())
            location_id = location.value["id"]
            # The canonical writer validates destination endpoint types. A
            # container name never implies an arbitrary hosted subspace.

        source = {"type": "conversation"}
        if context.conversation_id:
            source["session_id"] = context.conversation_id
        common = {"mode": request.mode, "source": source}
        if isinstance(request, NamedCreateItem):
            if item.status == "found":
                preview = await self.writing.mutate({
                    "operation": "update_location", "item_id": item.value["id"],
                    "location_id": location_id, "mode": "preview", "source": source,
                })
                if preview.status == "NO_CHANGE":
                    return reply("NO_CHANGE", "ITEM_ALREADY_RECORDED_AT_LOCATION")
                return reply("CONFLICT", "ITEM_ALREADY_EXISTS")
            key = hashlib.sha256((str(context.household_id) + "\0" + normalize_entity_alias(request.item_name)).encode()).hexdigest()
            kind = self.writing.catalog.entities["item"].property_types.get("name")
            name = [request.item_name] if kind == "collection" else request.item_name if kind == "string" else {"und": request.item_name}
            result = await self.writing.mutate({
                "operation": "create", "item": {"id": "item:recorded_" + key,
                    "type": "item", "properties": {"name": name}},
                "location_id": location_id, **common,
            })
        else:
            result = await self.writing.mutate({
                "operation": request.operation, "item_id": item.value["id"],
                **({"location_id": location_id} if location_id else {}), **common,
            })
        return reply(result.status, result.reason)


def render_mutation_result(request: NamedWriteRequest, response: dict[str, Any], language: str) -> str:
    """Acknowledge only tool-confirmed outcomes, without another model turn."""
    zh = language.startswith('zh')
    result = response.get('result', {}) if response.get('ok') is True else {}
    status = result.get('status')
    item = request.item_name
    location = getattr(request, 'location_name', '')
    if status in {'APPLIED', 'NO_CHANGE'}:
        if request.operation == 'delete':
            return f'已删除{item}的物品记录。' if zh else f'Deleted the item record for {item}.'
        if request.operation == 'update_location':
            return f'已将{item}的位置更新为{location}。' if zh else f'Recorded {item} at {location}.'
        return f'已记住：{location}有{item}。' if zh else f'Recorded {item} in {location}.'
    if status == 'PROPOSED':
        if request.operation == 'delete':
            return f'预览：删除{item}的物品记录；尚未执行。' if zh else f'Preview: delete {item}. No change has been saved.'
        action = '记录' if request.operation == 'create' else '更新位置'
        return f'预览：{item}位于{location}（{action}）；尚未保存。' if zh else f'Preview: record {item} at {location}. No change has been saved.'
    if status == 'NOT_FOUND':
        name = location if str(result.get('reason', '')).startswith('LOCATION_') else item
        return f'没有找到“{name}”，未执行修改。请确认完整名称。' if zh else f'Could not find {name}; no change was made. Please confirm the full name.'
    if status == 'CONFLICT':
        if result.get('reason') == 'ITEM_ALREADY_EXISTS':
            return f'已有名为“{item}”的物品记录，未新增或移动。若要移动它，请明确指定。' if zh else f'{item} already has a record. It was not created or moved; please explicitly request a move if intended.'
        return '物品或位置存在冲突，未执行修改。请提供更明确的名称。' if zh else 'The item or location is ambiguous or conflicting; no change was made. Please clarify the name.'
    return '未能确认这次修改已完成，请查询当前记录后再试。' if zh else 'The change could not be confirmed. Check the current record before retrying.'
