"""Resolve model-facing item names before invoking canonical mutations."""
from __future__ import annotations

from typing import Any

from .semantic_facts import AgentRequestContext, HouseholdFactEngine, SemanticFactRequest, FactResult
from .writing import ItemWritingService
from .semantic_ontology import SemanticOntology
from .mutation_ir import NamedCreateItem, NamedMoveItem, NamedUpdateAttributes, NamedWriteRequest


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

        attributes = dict(getattr(request, "attributes", {}))
        if isinstance(request, NamedCreateItem):
            attributes.setdefault("item_type", "unknown")
        properties = {}
        for name, value in attributes.items():
            prop = self.writing.ontology.properties.get(name)
            if prop is None or not prop.item_writable or not prop.fields:
                return reply("REJECTED", "ATTRIBUTE_NOT_WRITABLE")
            if not self.writing._matches_kind(value, prop.item_writable):
                return reply("REJECTED", "INVALID_ITEM_PROPERTY_TYPE")
            if isinstance(value, str) and (not value.strip() or len(value) > 4096):
                return reply("REJECTED", "INVALID_ATTRIBUTE_VALUE")
            properties[prop.fields[0]] = value

        location_id = None
        if isinstance(request, (NamedCreateItem, NamedMoveItem)):
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
                    stored = await self.writing._record(item.value["id"])
                    # Repeated creation never claims mismatching explicit attributes were saved.
                    if any(stored.get(self.writing.ontology.properties[key].fields[0]) != value
                           for key, value in request.attributes.items()):
                        return reply("CONFLICT", "ITEM_ALREADY_EXISTS")
                    return reply("NO_CHANGE", "ITEM_ALREADY_RECORDED_AT_LOCATION")
                return reply("CONFLICT", "ITEM_ALREADY_EXISTS")
            record_id = f"item:{request.item_key}"
            name = {"en": request.name_en, "zh": request.name_zh}
            result = await self.writing.mutate({
                "operation": "create", "item": {"id": record_id,
                    "type": "item", "properties": {"name": name, **properties}},
                "location_id": location_id, **common,
            })
            if result.reason == "ENTITY_ALREADY_EXISTS":
                return reply("CONFLICT", "ITEM_ALREADY_EXISTS")
        else:
            result = await self.writing.mutate({
                "operation": request.operation, "item_id": item.value["id"],
                **({"location_id": location_id} if location_id else {}),
                **({"properties": properties} if isinstance(request, NamedUpdateAttributes) else {}), **common,
            })
        response = reply(result.status, result.reason)
        if attributes and result.status in {"APPLIED", "PROPOSED", "NO_CHANGE"}:
            response["attributes"] = attributes
        return response


def render_mutation_result(
    request: NamedWriteRequest, response: dict[str, Any], language: str,
    ontology: SemanticOntology,
) -> str:
    """Acknowledge only tool-confirmed outcomes, without another model turn."""
    zh = language.startswith('zh')
    result = response.get('result', {}) if response.get('ok') is True else {}
    status = result.get('status')
    item = request.item_name
    location = getattr(request, 'location_name', '')
    if request.operation == 'update_attributes' and status in {'APPLIED', 'NO_CHANGE', 'PROPOSED'}:
        from .semantic_display import SemanticDisplay
        display = SemanticDisplay(ontology, language)
        details = '；'.join(f'{display.property(key)}：{display.literal(key, value)}'
                           for key, value in result.get('attributes', {}).items())
        if status == 'PROPOSED':
            return f'预览：{item}的属性更新为 {details}；尚未保存。' if zh else f'Preview: {item} attributes {details}; not saved.'
        return f'已记录{item}的属性：{details}。' if zh else f'Recorded attributes for {item}: {details}.'
    if status == 'REJECTED' and result.get('reason') in {'ATTRIBUTE_NOT_WRITABLE', 'INVALID_ITEM_PROPERTY_TYPE', 'INVALID_ATTRIBUTE_VALUE', 'ITEM_TYPE_REQUIRED'}:
        return '属性名称或值不符合声明的 schema，未保存。请提供支持的属性及正确类型的值。' if zh else 'Attribute name or value does not match the declared schema; nothing was saved.'
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
            return f'已有名为“{item}”的物品记录，未覆盖。若要修改属性或位置，请明确说明。' if zh else f'{item} already has a record and was not overwritten. Please explicitly request an attribute or location update.'
        return '物品或位置存在冲突，未执行修改。请提供更明确的名称。' if zh else 'The item or location is ambiguous or conflicting; no change was made. Please clarify the name.'
    return '未能确认这次修改已完成，请查询当前记录后再试。' if zh else 'The change could not be confirmed. Check the current record before retrying.'
