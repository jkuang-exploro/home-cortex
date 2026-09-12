"""Deterministic household-fact execution over resolved entities."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from time import perf_counter
from typing import Any

from .profiling import stage
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
    FactResult,
    FactRow,
    FactStatus,
    SemanticFactRequest,
    SemanticFilter,
    _FactFailure,
    _entity_type,
    _last_relation,
    _related_entity_id,
)
from .semantic_schema import SemanticSchemaRegistry
from .entity_resolver import EntityResolver, ResolvedEntities, _FactExecution, _derived_date_matches


class HouseholdFactEngine:
    def __init__(
        self,
        dispatcher: Any,
        schema: SemanticSchemaRegistry,
        *,
        max_records: int = 25,
    ) -> None:
        self.dispatcher = dispatcher
        self.schema = schema
        if not 1 <= max_records < 100:
            raise ValueError("max_records must be between 1 and 99 for completeness checks")
        self.resolver = EntityResolver(
            schema,
            max_records=max_records,
        )

    @stage("executor.total")
    async def execute(
        self,
        request: SemanticFactRequest,
        context: AgentRequestContext,
    ) -> tuple[FactResult, int, float, float]:
        if not self.schema.validates(request):
            return FactResult("semantic_plan_unsupported"), 0, 0, 0
        execution = _FactExecution(self.dispatcher, context.caller_entity_id)
        resolution_started = perf_counter()
        allow_empty_collection = request.operation in {"count", "select"} or request.projection == "each"
        operation = OPERATORS[request.operation]
        expect_many = request.other is None and (
            request.projection == "each" or operation.input_shape == "collection"
            or (request.operation == "select" and request.property is None)
            or bool(request.filters)
        )
        resolution = await self.resolver.resolve(
            request.subject,
            context,
            execution,
            allow_empty_collection=allow_empty_collection,
            expect_many=expect_many,
        )
        other_resolution = (
            await self.resolver.resolve(
                request.other,
                context,
                execution,
                allow_empty_collection=False,
                expect_many=False,
            )
            if request.other is not None
            else ResolvedEntities()
        )
        exclusions = [await self.resolver.resolve(
            reference, context, execution, expect_many=reference.cardinality == "collection"
        ) for reference in request.exclude]
        failed = next(
            (
                item
                for item in (resolution, other_resolution, *exclusions)
                if isinstance(item, FactResult)
            ),
            None,
        )
        if failed is not None:
            return (
                failed,
                execution.query_count,
                (perf_counter() - resolution_started) * 1000,
                0,
            )
        entities = [dict(item) for item in resolution.entities]
        other_entities = [dict(item) for item in other_resolution.entities]
        relationship_records = [dict(item) for item in resolution.relationship_records]
        entity_resolution_ms = (perf_counter() - resolution_started) * 1000
        excluded_ids = {entity_id for item in exclusions for entity_id in item.entity_ids}
        entities = [item for item in entities if item.get("id") not in excluded_ids]
        relationship_records = [edge for edge in relationship_records if _related_entity_id(edge) not in excluded_ids]
        computation_started = perf_counter()
        try:
            filter_failures: list[FactRow] = []
            if request.filters:
                entities, relationship_records, filter_failures = await self._filter_collection(
                    request,
                    entities,
                    relationship_records,
                    context,
                    execution,
                )
            if request.projection == "each":
                result = await self._project_each(
                    request, entities, other_entities, relationship_records,
                    context, execution, initial_rows=filter_failures,
                )
            else:
                result = await self._operate(
                    request, entities, other_entities, relationship_records,
                    context, execution,
                )
        except _FactFailure as error:
            result = FactResult(
                error.status,
                evidence=error.evidence,
                missing_requirements=error.missing,
                candidates=error.candidates,
            )
        if result.status == "found":
            if result.shape == "entities" and resolution.content_groups:
                visible = {item["id"]: item for item in result.value}
                groups = tuple(
                    FactContentGroup(group.space, tuple(
                        visible[item["id"]] for item in group.entities if item["id"] in visible
                    ))
                    for group in resolution.content_groups
                    if any(item["id"] in visible for item in group.entities)
                )
                result = replace(result, content_groups=groups)
            focus = tuple(str(item["id"]) for item in (*entities, *other_entities) if item.get("id"))
            if (request.other is None and request.property_source == "entity"
                and isinstance(result.value, Mapping) and result.value.get("id")):
                focus = (str(result.value["id"]),)
            final_types = {_entity_type(item) for item in entities}
            if (
                request.subject.kind == "named_entity"
                and request.subject.path
                and request.subject.entity_type not in final_types
            ):
                anchor = next(
                    (
                        entity_id
                        for edge in resolution.evidence.relationships
                        for entity_id in (edge.source_id, edge.target_id)
                        if entity_id is not None
                        and entity_id.partition(":")[0]
                        == request.subject.entity_type
                    ),
                    None,
                )
                if anchor is not None:
                    focus = (anchor, *focus)
            result = replace(result, focus_entity_ids=tuple(dict.fromkeys(focus)))
        computation_ms = (perf_counter() - computation_started) * 1000
        return result, execution.query_count, entity_resolution_ms, computation_ms

    async def _project_each(
        self,
        request: SemanticFactRequest,
        entities: list[dict[str, Any]],
        other_entities: list[dict[str, Any]],
        relationship_records: list[dict[str, Any]],
        context: AgentRequestContext,
        execution: "_FactExecution",
        *,
        initial_rows: Sequence[FactRow] = (),
    ) -> FactResult:
        rows: list[FactRow] = list(initial_rows)
        scalar = request.model_copy(update={"projection": "scalar", "exclude": ()})
        relation = _last_relation(request.subject)
        for entity in entities:
            edges = [
                edge for edge in relationship_records
                if _related_entity_id(edge) == entity.get("id")
            ]
            groups = (
                [[edge] for edge in edges]
                if request.property_source == "relationship" else [edges]
            )
            for group in groups or [[]]:
                # Share loaded records, but scope relationship evidence to this row.
                row_execution = _FactExecution(self.dispatcher, context.caller_entity_id)
                row_execution.entity_cache = execution.entity_cache
                for edge in group:
                    assert relation is not None
                    row_execution.remember_relationship(relation, edge)
                evidence = FactEvidence(
                    (str(entity["id"]),), relation, request.property,
                    tuple(row_execution.relationship_evidence),
                )
                try:
                    visible = await row_execution.load_if_unnamed(entity)
                    value = await self._operate(
                        scalar, [entity], [], group, context, row_execution,
                    )
                except _FactFailure as error:
                    visible = entity
                    value = FactResult(error.status, missing_requirements=error.missing)
                execution.query_count += row_execution.query_count
                rows.append(FactRow(
                    visible, value.status, value.value, _result_unit(request),
                    evidence, value.missing_requirements,
                ))
        return FactResult("found", shape="rows", rows=tuple(rows))

    async def _operate(
        self,
        request: SemanticFactRequest,
        entities: list[dict[str, Any]],
        other_entities: list[dict[str, Any]],
        relationship_records: list[dict[str, Any]],
        context: AgentRequestContext,
        execution: "_FactExecution",
    ) -> FactResult:
        evidence = FactEvidence(
            tuple(
                str(item.get("id"))
                for item in (*entities, *other_entities)
                if item.get("id")
            ),
            request.subject.path[-1].relation if request.subject.path else None,
            request.property,
            tuple(execution.relationship_evidence),
        )
        if request.operation == "count":
            value = execute_operator("count", OperatorInput(records=entities))
            return FactResult("found", value, evidence)
        if request.operation == "select" and request.property is None:
            visible = [await execution.load_if_unnamed(item) for item in entities]
            return FactResult("found", visible, evidence, shape="entities")
        if request.operation == "inspect":
            singular = await self._singular(entities, request, execution)
            if isinstance(singular, FactResult):
                return singular
            entity_type = str(singular.get("id", "")).split(":", 1)[0]
            values = {
                name: singular.get(self.schema.physical_property(entity_type, name))
                for name in sorted(self.schema.semantic_properties(entity_type) & self.schema.ontology.properties.keys())
            }
            return FactResult("found", values, evidence)
        if request.operation == "resolve_reference":
            singular = await self._singular(
                entities,
                request,
                execution,
                load_full=False,
            )
            if isinstance(singular, FactResult):
                return singular
            return FactResult("found", singular, evidence, shape="entity")
        if request.operation == "same_entity":
            left = await self._singular(
                entities, request, execution, load_full=False
            )
            right = await self._singular(
                other_entities, request, execution, load_full=False
            )
            if isinstance(left, FactResult):
                return left
            if isinstance(right, FactResult):
                return right
            left_id = left.get("id")
            right_id = right.get("id")
            if not isinstance(left_id, str) or not isinstance(right_id, str):
                return FactResult("computation_impossible", evidence=evidence)
            return FactResult(
                "found", left_id == right_id, evidence, shape="scalar"
            )
        if request.operation == "select":
            if request.property_source == "relationship":
                relationship = self._singular_relationship(
                    relationship_records,
                    evidence,
                )
                if isinstance(relationship, FactResult):
                    return relationship
                property_result = self._relationship_property(
                    relationship,
                    request,
                    evidence,
                )
                if isinstance(property_result, FactResult):
                    return property_result
                return FactResult("found", property_result, evidence)
            singular = await self._singular(entities, request, execution)
            if isinstance(singular, FactResult):
                return singular
            property_result = self._property(singular, request.property or "")
            if isinstance(property_result, FactResult):
                return property_result
            return FactResult("found", property_result, evidence)
        definition = OPERATORS[request.operation]
        records = (
            list(relationship_records)
            if request.property_source == "relationship"
            else list(entities)
        )
        if request.other is not None:
            left = await self._singular(entities, request, execution)
            right = await self._singular(other_entities, request, execution)
            if isinstance(left, FactResult):
                return left
            if isinstance(right, FactResult):
                return right
            records = [left, right]
        elif definition.input_shape == "scalar":
            singular = (
                self._singular_relationship(relationship_records, evidence)
                if request.property_source == "relationship"
                else await self._singular(entities, request, execution)
            )
            if isinstance(singular, FactResult):
                return singular
            records = [dict(singular)]
        else:
            records = (
                records
                if request.property_source == "relationship"
                else [await execution.load(item) for item in records]
            )
        normalized: list[dict[str, Any]] = []
        for record in records:
            value = (
                self._relationship_property(record, request, evidence)
                if request.property_source == "relationship"
                else self._property(record, request.property)
                if request.property is not None
                else record
            )
            if isinstance(value, FactResult):
                return FactResult(
                    (
                        "computation_input_missing"
                        if value.status == "property_unavailable"
                        else value.status
                    ),
                    evidence=evidence,
                    missing_requirements=value.missing_requirements,
                    candidates=value.candidates,
                )
            normalized.append({"value": value, "entity": record})
        if not normalized:
            return FactResult("computation_impossible", evidence=evidence)
        if request.operation in {"latest", "earliest"}:
            normalized.sort(
                key=lambda item: item["value"],
                reverse=request.operation == "latest",
            )
        field = "value" if request.property is not None else None
        parameters = {
            "reference": "household_now",
            "mode": request.mode,
            "amount": request.amount,
            "from_unit": request.from_unit,
            "to_unit": request.to_unit,
        }
        try:
            definition.validate(
                field=field,
                field_kind=(
                    infer_field_kind([item["value"] for item in normalized])
                    if field is not None
                    else "unknown"
                ),
                order_by=(field if request.operation in {"latest", "earliest"} else None),
                order_by_kind=(
                    infer_field_kind([item["value"] for item in normalized])
                    if field is not None
                    else "unknown"
                ),
                parameters=parameters,
            )
            value = execute_operator(
                request.operation,
                OperatorInput(
                    records=normalized,
                    field=field,
                    order_by=field if request.operation in {"latest", "earliest"} else None,
                    mode=request.mode,
                    amount=request.amount,
                    reference="household_now",
                    from_unit=request.from_unit,
                    to_unit=request.to_unit,
                    now=context.current_time,
                ),
            )
        except (OperatorValidationError, OperatorExecutionError, TypeError, ValueError):
            return FactResult(
                "computation_impossible",
                evidence=evidence,
                missing_requirements=((request.property,) if request.property else ()),
            )
        if isinstance(value, Mapping) and isinstance(value.get("entity"), Mapping):
            selected = dict(value["entity"])
            if request.other is not None:
                selected_id = selected.get("id")
                other = next(
                    (
                        record
                        for record in records
                        if record.get("id") != selected_id
                    ),
                    records[0],
                )
                selected_value = self._property(selected, request.property or "")
                other_value = self._property(other, request.property or "")
                return FactResult(
                    "found",
                    {
                        "selected": selected,
                        "other": other,
                        "equal": selected_value == other_value,
                    },
                    evidence,
                )
            value = selected
        shape = (
            "entity" if request.property_source == "entity"
            and isinstance(value, Mapping) and value.get("id") else "scalar"
        )
        return FactResult("found", value, evidence, unit=_result_unit(request), shape=shape)

    async def _filter_collection(
        self,
        request: SemanticFactRequest,
        entities: list[dict[str, Any]],
        relationship_records: list[dict[str, Any]],
        context: AgentRequestContext,
        execution: "_FactExecution",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[FactRow]]:
        # Relationship projections have edge rows: all edge conditions bind the
        # same final edge, rather than admitting every edge of a matching entity.
        edge_filters = tuple(item for item in request.filters if item.source == "relation")
        if request.projection == "each" and request.property_source == "relationship" and edge_filters:
            relationship_records = [
                edge for edge in relationship_records
                if all(self._relation_filter_matches(request, item, [edge]) for item in edge_filters)
            ]
            retained = {_related_entity_id(edge) for edge in relationship_records}
            entities = [entity for entity in entities if entity.get("id") in retained]
            request = request.model_copy(update={
                "filters": tuple(item for item in request.filters if item.source != "relation")
            })
        matched: list[dict[str, Any]] = []
        failures: list[FactRow] = []
        for entity in entities:
            entity_id = entity.get("id")
            edges = [
                edge
                for edge in relationship_records
                if _related_entity_id(edge) == entity_id
            ]
            include = True
            try:
                for item in request.filters:
                    if item.predicate is not None:
                        include = await self._semantic_predicate_matches(
                            request,
                            item.predicate,
                            entity,
                            edges,
                            context,
                            execution,
                        )
                    elif item.source == "relation":
                        include = self._relation_filter_matches(request, item, edges)
                    else:
                        include = await self._entity_filter_matches(
                            item,
                            entity,
                            execution,
                            context,
                        )
                    if not include:
                        break
            except _FactFailure as error:
                if request.projection != "each":
                    raise
                visible = await execution.load_if_unnamed(entity)
                failures.append(
                    FactRow(
                        visible,
                        error.status,
                        evidence=FactEvidence(
                            entity_ids=(str(entity_id),) if isinstance(entity_id, str) else (),
                            relationship=_last_relation(request.subject),
                            semantic_property=request.property,
                        ),
                        missing_requirements=error.missing,
                        unit=_result_unit(request),
                    )
                )
                continue
            if include:
                matched.append(entity)
        matched_ids = {
            entity.get("id")
            for entity in matched
            if isinstance(entity.get("id"), str)
        }
        return matched, [
            edge
            for edge in relationship_records
            if _related_entity_id(edge) in matched_ids
        ], failures

    async def _entity_filter_matches(
        self,
        item: SemanticFilter,
        entity: Mapping[str, Any],
        execution: "_FactExecution",
        context: AgentRequestContext,
    ) -> bool:
        assert item.property is not None
        physical = self.schema.physical_property(_entity_type(entity), item.property)
        if physical is None:
            raise _FactFailure("filter_unsupported", missing=(item.property,))
        record = (
            dict(entity)
            if physical in entity
            else await execution.load(entity)
        )
        if physical not in record or record.get(physical) is None:
            raise _FactFailure("filter_input_missing", missing=(item.property,))
        self.schema.validate_filter_value(item.property, record.get(physical))
        if item.transform:
            return _derived_date_matches(
                record.get(physical),
                context.current_time,
                transform=item.transform,
                mode=item.mode,
                operator=item.operator,
                compare_value=item.value,
                require_past=True,
                missing=(item.property,),
            )
        return evaluate_predicate(item.operator, record.get(physical), item.value)

    def _relation_filter_matches(
        self,
        request: SemanticFactRequest,
        item: SemanticFilter,
        edges: Sequence[Mapping[str, Any]],
    ) -> bool:
        assert request.subject.path and item.property is not None
        resolved = self.schema.physical_relation(request.subject.path[-1].relation)
        if resolved is None:
            raise _FactFailure("filter_unsupported", missing=(item.property,))
        physical = self.schema.relation_property(resolved[0], item.property)
        if physical is None:
            raise _FactFailure("filter_unsupported", missing=(item.property,))
        values = [edge.get(physical) for edge in edges if edge.get(physical) is not None]
        if not values:
            raise _FactFailure("filter_input_missing", missing=(item.property,))
        for value in values:
            self.schema.validate_filter_value(item.property, value)
        return any(evaluate_predicate(item.operator, value, item.value) for value in values)

    async def _semantic_predicate_matches(
        self,
        request: SemanticFactRequest,
        predicate: str,
        entity: Mapping[str, Any],
        edges: Sequence[Mapping[str, Any]],
        context: AgentRequestContext,
        execution: "_FactExecution",
    ) -> bool:
        definition = self.schema.ontology.collection_predicates.get(predicate)
        if definition is None:
            raise _FactFailure("filter_unsupported", missing=(predicate,))
        relation_name = (
            (self.schema.physical_relation(request.subject.path[-1].relation)
             if request.subject.path else None) or (None, None)
        )[0]
        role_property = (
            self.schema.relation_property(
                relation_name,
                definition.relation_property,
            )
            if relation_name is not None
            else None
        )
        recognized_roles = {
            edge.get(role_property)
            for edge in edges
            if role_property is not None
            and edge.get(role_property) in definition.recognized_values
        }
        if self.schema.contracts is not None and role_property is not None:
            for edge in edges:
                if edge.get(role_property) is not None:
                    self.schema.validate_filter_value(definition.relation_property, edge[role_property])
        if recognized_roles:
            decisions = {
                role in definition.matching_values for role in recognized_roles
            }
            if len(decisions) != 1:
                raise _FactFailure("filter_input_missing", missing=(predicate,))
            return decisions.pop()

        fallback = definition.fallback
        physical = self.schema.physical_property(
            _entity_type(entity), fallback.property
        )
        if physical is None:
            raise _FactFailure("filter_unsupported", missing=(predicate,))
        record = (
            dict(entity)
            if physical in entity
            else await execution.load(entity)
        )
        raw_value = record.get(physical)
        if raw_value is None:
            raise _FactFailure(
                "filter_input_missing",
                missing=(predicate, fallback.property),
            )
        self.schema.validate_filter_value(fallback.property, raw_value)
        return _derived_date_matches(
            raw_value,
            context.current_time,
            transform=fallback.transform,
            mode=fallback.mode,
            operator=fallback.operator,
            compare_value=self.schema.ontology.policy_values[fallback.value_from_policy],
            require_past=fallback.require_past,
            missing=(predicate, fallback.property),
        )

    @staticmethod
    def _singular_relationship(
        records: Sequence[Mapping[str, Any]],
        evidence: FactEvidence,
    ) -> Mapping[str, Any] | FactResult:
        if not records:
            return FactResult("relationship_not_found", evidence=evidence)
        if len(records) > 1:
            return FactResult("ambiguous", evidence=evidence)
        return records[0]

    def _relationship_property(
        self,
        relationship: Mapping[str, Any],
        request: SemanticFactRequest,
        evidence: FactEvidence,
    ) -> Any | FactResult:
        assert request.subject.path and request.property is not None
        semantic_relation = request.subject.path[-1].relation
        resolved = self.schema.physical_relation(semantic_relation)
        physical = (
            self.schema.relation_property(resolved[0], request.property)
            if resolved is not None
            else None
        )
        if (
            physical is None
            or physical not in relationship
            or relationship.get(physical) is None
            or not self.schema.value_valid(request.property, relationship.get(physical))
        ):
            return FactResult(
                "relation_property_unavailable",
                evidence=evidence,
                missing_requirements=(request.property,),
            )
        return relationship[physical]

    async def _singular(
        self,
        entities: list[dict[str, Any]],
        request: SemanticFactRequest,
        execution: "_FactExecution",
        *,
        load_full: bool = True,
    ) -> dict[str, Any] | FactResult:
        if not entities:
            status: FactStatus = (
                "relationship_not_found" if request.subject.path else "entity_not_found"
            )
            return FactResult(status)
        if len(entities) > 1:
            loaded = tuple(
                [
                    await execution.load_if_unnamed(item)
                    for item in entities
                ]
            )
            return FactResult("ambiguous", candidates=loaded)
        record = (
            await execution.load(entities[0])
            if load_full
            else await execution.load_if_unnamed(entities[0])
        )
        return record

    def _property(self, entity: Mapping[str, Any], semantic: str) -> Any | FactResult:
        entity_type = _entity_type(entity)
        physical = self.schema.physical_property(entity_type, semantic)
        if (physical is None or physical not in entity or entity.get(physical) is None
            or not self.schema.value_valid(semantic, entity.get(physical))):
            return FactResult(
                "property_unavailable",
                evidence=FactEvidence(
                    (str(entity.get("id")),) if entity.get("id") else (),
                    semantic_property=semantic,
                ),
                missing_requirements=(semantic,),
            )
        return entity[physical]



def _result_unit(request: SemanticFactRequest) -> str | None:
    if request.operation == "completed_years":
        return "years"
    if request.operation in {"date_difference", "duration", "annual_occurrence"}:
        return request.mode
    if request.operation == "unit_conversion":
        return request.to_unit
    return None
