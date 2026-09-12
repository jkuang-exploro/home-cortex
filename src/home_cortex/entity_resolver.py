"""Ground SemanticReference values to canonical household entities."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from dataclasses import dataclass
from typing import Any

from .request_tracing import stage
from .operator_registry import (
    OPERATORS,
    OperatorExecutionError,
    OperatorInput,
    OperatorValidationError,
    evaluate_predicate,
    execute_operator,
    infer_field_kind,
)
from .semantic_ir import (
    AgentRequestContext,
    FactContentGroup,
    FactEvidence,
    FactRelationshipEvidence,
    FactResult,
    SemanticFilter,
    SemanticReference,
    _FactFailure,
    _entity_type,
    _string_or_none,
    _unique_entities,
)
from .semantic_schema import SemanticSchemaRegistry

@dataclass(frozen=True)
class ResolvedEntities:
    """Successful grounding, including the edges required for later computation.

    Failures use FactResult directly, just like execution failures.
    """
    entities: tuple[Mapping[str, Any], ...] = ()
    entity_ids: tuple[str, ...] = ()
    evidence: FactEvidence = FactEvidence()
    relationship_records: tuple[Mapping[str, Any], ...] = ()
    content_groups: tuple[FactContentGroup, ...] = ()


class EntityResolver:
    """Authoritative resolver from semantic references to canonical entities."""

    def __init__(
        self,
        schema: SemanticSchemaRegistry,
        *,
        max_records: int = 25,
    ) -> None:
        self.schema = schema
        self.max_records = max_records

    @stage("resolver.total")
    async def resolve(
        self,
        reference: SemanticReference,
        context: AgentRequestContext,
        execution: "_FactExecution",
        *,
        allow_empty_collection: bool = False,
        expect_many: bool = False,
    ) -> ResolvedEntities | FactResult:
        try:
            entities, relationship_records, content_groups = await self._resolve(
                reference,
                context,
                execution,
                allow_empty_collection=allow_empty_collection,
            )
            if len(entities) > 1 and not expect_many:
                candidates = tuple(
                    [await execution.load_if_unnamed(item) for item in entities]
                )
                return FactResult("ambiguous", candidates=candidates)
            entity_ids = tuple(
                str(item["id"])
                for item in entities
                if isinstance(item.get("id"), str)
            )
            return ResolvedEntities(
                tuple(entities),
                entity_ids,
                FactEvidence(
                    entity_ids=entity_ids,
                    relationship=(
                        reference.path[-1].relation if reference.path else None
                    ),
                    relationships=tuple(execution.relationship_evidence),
                ),
                relationship_records=tuple(relationship_records),
                content_groups=tuple(content_groups),
            )
        except _FactFailure as error:
            return FactResult(
                error.status,
                evidence=error.evidence,
                candidates=error.candidates,
                missing_requirements=error.missing,
            )

    async def _resolve(
        self,
        reference: SemanticReference,
        context: AgentRequestContext,
        execution: "_FactExecution",
        *,
        allow_empty_collection: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[FactContentGroup]]:
        if reference.kind == "unresolved":
            raise _FactFailure("ambiguous", missing=("semantic_reference",))
        elif reference.kind == "discourse":
            discourse = context.discourse
            if (discourse is None or context.conversation_id is None
                or (discourse.conversation_id, discourse.caller_entity_id, discourse.household_id, discourse.assistant_id)
                != (context.conversation_id, context.caller_entity_id, context.household_id, context.assistant_id)
                or reference.turn_offset is None or reference.turn_offset > len(discourse.turns)
                or not discourse.turns[-reference.turn_offset]):
                raise _FactFailure("discourse_context_missing", missing=("discourse_antecedent",))
            ids = discourse.turns[-reference.turn_offset]
            entities = [await execution.load({"id": entity_id}) for entity_id in ids]
            entities = [
                entity
                for entity in entities
                if _entity_type(entity) == reference.entity_type
            ]
            if not entities:
                raise _FactFailure(
                    "discourse_context_missing", missing=("discourse_antecedent",)
                )
            if reference.cardinality == "single" and len(entities) != 1:
                raise _FactFailure("ambiguous", candidates=tuple(entities), missing=("discourse_antecedent",))
        elif reference.kind == "assistant":
            entities = [
                {
                    "id": context.assistant_id,
                    "display_name": context.assistant_display_name,
                }
            ]
        elif reference.kind == "self":
            if context.caller_entity_id is None:
                raise _FactFailure("caller_context_missing")
            entities = [{"id": context.caller_entity_id}]
        elif reference.kind == "current_household":
            if context.household_id is None:
                raise _FactFailure("entity_not_found")
            entities = [{"id": context.household_id}]
        elif reference.kind == "entity_id":
            entities = [{"id": reference.value}]
        elif reference.kind == "named_entity":
            records = await execution.records(
                "resolve_entity_alias",
                {
                    "text": reference.value,
                    "entity_type": reference.entity_type,
                    "limit": self.max_records,
                    "speaker_id": context.caller_entity_id,
                    "household_id": context.household_id,
                },
            )
            records = await self._scope_named_records(
                records,
                context.household_id,
                execution,
            )
            if not records:
                raise _FactFailure("entity_not_found")
            if len(records) > 1:
                raise _FactFailure(
                    "ambiguous",
                    evidence=FactEvidence(
                        entity_ids=tuple(
                            str(item["id"])
                            for item in records
                            if isinstance(item.get("id"), str)
                        )
                    ),
                    candidates=tuple(records),
                )
            entities = records
        else:
            raise _FactFailure("computation_impossible")

        anchors = list(entities)
        last_relation: str | None = None
        last_relationship_records: list[dict[str, Any]] = []
        last_content_groups: list[FactContentGroup] = []
        for step in reference.path:
            resolved = self.schema.physical_relation(step.relation)
            if resolved is None:
                raise _FactFailure(
                    "relationship_not_found",
                    evidence=FactEvidence(relationship=step.relation),
                )
            relation, direction = resolved
            last_relation = step.relation
            related: list[dict[str, Any]] = []
            step_relationship_records: list[dict[str, Any]] = []
            grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
            collapsed = False
            for entity in entities:
                entity_id = entity.get("id")
                if not isinstance(entity_id, str):
                    continue
                sources = [entity]
                if step.relation == "contents":
                    stored = await execution.load(entity)
                    if stored.get("collapse") is True:
                        collapsed = True
                        sources.extend(await self._hosted_descendants(stored, execution))
                for source in sources:
                    edges = await self._relation_edges(
                        str(source["id"]), relation, direction, execution,
                    )
                    for edge in edges:
                        if not self._edge_matches(edge, relation, step.filters):
                            continue
                        execution.remember_relationship(step.relation, edge)
                        candidate = edge.get("related_entity")
                        if isinstance(candidate, Mapping):
                            related.append(dict(candidate))
                            step_relationship_records.append(dict(edge))
                            if step.relation == "contents":
                                group = grouped.setdefault(str(source["id"]), (dict(source), []))
                                group[1].append(dict(candidate))
            entities = _unique_entities(related)
            entities = await self._filter_entities(
                entities,
                step.filters,
                execution,
                anchors,
                context,
            )
            if not entities:
                if allow_empty_collection:
                    return [], [], []
                raise _FactFailure(
                    "relationship_not_found",
                    evidence=FactEvidence(relationship=last_relation),
                )
            entity_ids = {
                str(entity["id"])
                for entity in entities
                if isinstance(entity.get("id"), str)
            }
            last_relationship_records = [
                edge
                for edge in step_relationship_records
                if isinstance(edge.get("related_entity"), Mapping)
                and edge["related_entity"].get("id") in entity_ids
            ]
            last_content_groups = [
                FactContentGroup(space, tuple(_unique_entities([
                    item for item in items if item.get("id") in entity_ids
                ])))
                for space, items in grouped.values()
            ] if collapsed else []
        return entities, last_relationship_records, last_content_groups

    async def _relation_edges(
        self, entity_id: str, relation: str, direction: str | None,
        execution: "_FactExecution",
    ) -> list[dict[str, Any]]:
        arguments: dict[str, Any] = {
            "entity_id": entity_id, "relation": relation,
            "include_ended": False, "limit": self.max_records + 1,
        }
        if direction is not None:
            arguments["direction"] = direction
        edges = await execution.records("get_relationships", arguments)
        if len(edges) > self.max_records:
            raise _FactFailure("collection_incomplete", missing=(relation,))
        return edges

    async def _hosted_descendants(
        self, parent: Mapping[str, Any], execution: "_FactExecution",
    ) -> list[dict[str, Any]]:
        """Expand a query view, retaining only authoritative graph edges."""
        hosting = self.schema.physical_relation("hosted_space")
        if hosting is None:
            return []
        descendants = []
        visited: set[str] = set()
        active: set[str] = set()
        stack = [(dict(parent), False)]
        while stack:
            entity, leaving = stack.pop()
            entity_id = str(entity["id"])
            if leaving:
                active.remove(entity_id)
                continue
            if entity_id in active:
                raise _FactFailure("computation_impossible", missing=("hosting_cycle",))
            if entity_id in visited:
                continue
            visited.add(entity_id)
            active.add(entity_id)
            stack.append((entity, True))
            edges = await self._relation_edges(entity_id, *hosting, execution)
            for edge in edges:
                child = edge.get("related_entity")
                if not isinstance(child, Mapping) or _entity_type(child) != "space":
                    continue
                execution.remember_relationship("hosted_space", edge)
                descendants.append(dict(child))
                stack.append((dict(child), False))
        return _unique_entities(descendants)

    async def _scope_named_records(
        self,
        records: Sequence[Mapping[str, Any]],
        household_id: str | None,
        execution: "_FactExecution",
    ) -> list[dict[str, Any]]:
        """Apply schema-declared containment scope to named records.

        Entity types without a declared scope-parent edge retain their existing
        resolution behavior. Types with such an edge must have a recorded path
        to the request's household. This keeps scoping generic across items,
        spaces, and future contained entity types.
        """
        if household_id is None:
            return [dict(record) for record in records]
        scoped: list[dict[str, Any]] = []
        for record in records:
            entity_id = record.get("id")
            entity_type = _entity_type(record)
            parents = self.schema.edge_registry.scope_parent_relations(entity_type)
            if not parents:
                scoped.append(dict(record))
                continue
            if isinstance(entity_id, str) and await self._belongs_to_household(
                entity_id,
                household_id,
                execution,
                visited=frozenset(),
            ):
                scoped.append(dict(record))
        return scoped

    async def _belongs_to_household(
        self,
        entity_id: str,
        household_id: str,
        execution: "_FactExecution",
        *,
        visited: frozenset[str],
    ) -> bool:
        if entity_id == household_id:
            return True
        if entity_id in visited or len(visited) >= 12:
            return False
        parents = self.schema.edge_registry.scope_parent_relations(
            entity_id.partition(":")[0]
        )
        if not parents:
            return False
        next_visited = visited | {entity_id}
        for relation in parents:
            edges = await execution.records(
                "get_relationships",
                {
                    "entity_id": entity_id,
                    "relation": relation,
                    "direction": "out",
                    "include_ended": False,
                    "limit": self.max_records + 1,
                },
            )
            if len(edges) > self.max_records:
                raise _FactFailure(
                    "collection_incomplete",
                    missing=(relation,),
                )
            for edge in edges:
                parent = edge.get("related_entity")
                parent_id = parent.get("id") if isinstance(parent, Mapping) else None
                if isinstance(parent_id, str) and await self._belongs_to_household(
                    parent_id,
                    household_id,
                    execution,
                    visited=next_visited,
                ):
                    return True
        return False

    def _edge_matches(
        self,
        edge: Mapping[str, Any],
        relation: str,
        filters: Sequence[SemanticFilter],
    ) -> bool:
        for item in filters:
            if item.source != "relation":
                continue
            physical = self.schema.relation_property(relation, item.property)
            if physical is None:
                raise _FactFailure(
                    "property_unavailable",
                    evidence=FactEvidence(semantic_property=item.property),
                    missing=(item.property,),
                )
            self.schema.validate_filter_value(item.property, edge.get(physical))
            if not evaluate_predicate(item.operator, edge.get(physical), item.value):
                return False
        return True

    async def _filter_entities(
        self,
        entities: list[dict[str, Any]],
        filters: Sequence[SemanticFilter],
        execution: "_FactExecution",
        anchors: Sequence[Mapping[str, Any]],
        context: AgentRequestContext,
    ) -> list[dict[str, Any]]:
        entity_filters = [item for item in filters if item.source == "entity"]
        if not entity_filters:
            return entities
        matched: list[dict[str, Any]] = []
        for entity in entities:
            entity_type = _entity_type(entity)
            mapped = [
                (item, self.schema.physical_property(entity_type, item.property))
                for item in entity_filters
            ]
            unavailable = next(
                (item.property for item, physical in mapped if physical is None),
                None,
            )
            if unavailable is not None:
                raise _FactFailure(
                    "property_unavailable",
                    evidence=FactEvidence(semantic_property=unavailable),
                    missing=(unavailable,),
                )
            needs_load = any(
                physical not in entity for _, physical in mapped if physical is not None
            )
            record = (
                await execution.load(entity) if needs_load else dict(entity)
            )
            predicates: list[bool] = []
            for item, physical in mapped:
                self.schema.validate_filter_value(item.property, record.get(physical))
                if item.transform:
                    predicates.append(
                        _derived_date_matches(
                            record.get(physical),
                            context.current_time,
                            transform=item.transform,
                            mode=item.mode,
                            operator=item.operator,
                            compare_value=item.value,
                            require_past=True,
                            missing=(item.property,),
                        )
                    )
                    continue
                expected = item.value
                if item.value_from == "anchor":
                    expected = await self._anchor_property(
                        anchors,
                        item.value_property or item.property,
                        execution,
                    )
                predicates.append(
                    evaluate_predicate(item.operator, record.get(physical), expected)
                )
            if all(predicates):
                matched.append(record)
        return matched

    async def _anchor_property(
        self,
        anchors: Sequence[Mapping[str, Any]],
        semantic_property: str,
        execution: "_FactExecution",
    ) -> Any:
        if len(anchors) != 1:
            raise _FactFailure("ambiguous", candidates=tuple(anchors))
        anchor = await execution.load(anchors[0])
        physical = self.schema.physical_property(
            _entity_type(anchor),
            semantic_property,
        )
        if physical is None or anchor.get(physical) is None:
            raise _FactFailure(
                "property_unavailable",
                evidence=FactEvidence(semantic_property=semantic_property),
                missing=(semantic_property,),
            )
        self.schema.validate_filter_value(semantic_property, anchor[physical])
        return anchor[physical]


class _FactExecution:
    def __init__(self, dispatcher: Any, caller_entity_id: str | None) -> None:
        self.dispatcher = dispatcher
        self.caller_entity_id = caller_entity_id
        self.query_count = 0
        self.entity_cache: dict[str, dict[str, Any]] = {}
        self.relationship_evidence: list[FactRelationshipEvidence] = []

    def remember_relationship(
        self,
        semantic_relation: str,
        edge: Mapping[str, Any],
    ) -> None:
        evidence = FactRelationshipEvidence(
            relation=semantic_relation,
            source_id=_string_or_none(edge.get("in") or edge.get("from")),
            target_id=_string_or_none(edge.get("out") or edge.get("to")),
            start=edge.get("start"),
            end=edge.get("end"),
        )
        if evidence not in self.relationship_evidence:
            self.relationship_evidence.append(evidence)

    @stage("graph.records")
    async def records(self, tool: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        entity_id = arguments.get("entity_id")
        if tool == "get_entity" and isinstance(entity_id, str):
            cached = self.entity_cache.get(entity_id)
            if cached is not None:
                return [dict(cached)]
        self.query_count += 1
        try:
            response = await self.dispatcher.dispatch_internal(
                tool,
                arguments,
                caller_entity_id=self.caller_entity_id,
            )
        except Exception:
            raise _FactFailure("computation_impossible") from None
        if not isinstance(response, Mapping) or response.get("ok") is not True:
            raise _FactFailure("computation_impossible")
        value = response.get("result")
        if not isinstance(value, list):
            raise _FactFailure("computation_impossible")
        records = [dict(item) for item in value if isinstance(item, Mapping)]
        if tool == "get_entity" and isinstance(entity_id, str) and records:
            self.entity_cache[entity_id] = dict(records[0])
        return records

    async def load(self, entity: Mapping[str, Any]) -> dict[str, Any]:
        entity_id = entity.get("id")
        if not isinstance(entity_id, str):
            raise _FactFailure("entity_not_found")
        if ":" not in entity_id:
            return dict(entity)
        records = await self.records("get_entity", {"entity_id": entity_id})
        if not records:
            raise _FactFailure("entity_not_found")
        return records[0]

    async def load_if_unnamed(self, entity: Mapping[str, Any]) -> dict[str, Any]:
        if any(entity.get(field) for field in ("display_name", "name", "full_name")):
            return dict(entity)
        return await self.load(entity)



def _derived_date_matches(
    raw_value: Any,
    now: datetime,
    *,
    transform: str,
    mode: str | None,
    operator: str,
    compare_value: Any,
    require_past: bool,
    missing: tuple[str, ...],
) -> bool:
    """Compare a date property after a declared transform against Household now."""
    normalized = {"value": raw_value}
    try:
        if require_past and execute_operator(
            "date_difference",
            OperatorInput(
                records=[normalized],
                field="value",
                mode="seconds",
                now=now,
            ),
        ) < 0:
            raise OperatorExecutionError("derived date filter requires a past date")
        OPERATORS[transform].validate(
            field="value",
            field_kind=infer_field_kind([raw_value]),
            parameters={"reference": "household_now", "mode": mode},
        )
        derived = execute_operator(
            transform,
            OperatorInput(
                records=[normalized],
                field="value",
                reference="household_now",
                now=now,
                mode=mode,
            ),
        )
    except (OperatorValidationError, OperatorExecutionError, TypeError, ValueError, KeyError):
        raise _FactFailure("filter_input_missing", missing=missing) from None
    return evaluate_predicate(operator, derived, compare_value)
