import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.grounding import AgentRequestContext
from home_cortex.schema_catalog import RuntimeSchemaCatalog
from home_cortex.semantic_facts import (
    HouseholdFactEngine,
    SemanticFactRequest,
    SemanticFactService,
    SemanticFactPlanner,
    SemanticFilter,
    SemanticReference,
    SemanticRelationStep,
    SemanticSchemaRegistry,
    TierZeroSemanticParser,
)

DATA_DIR = Path(__file__).parents[1] / "data"


class FixtureGraphDispatcher:
    def __init__(self) -> None:
        self.registry = EdgeSchemaRegistry.load_default(DATA_DIR)
        self.entities = {
            record["id"]: record
            for path in (DATA_DIR / "nodes").glob("*.json")
            for record in json.loads(path.read_text(encoding="utf-8"))
        }
        self.edges = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in (DATA_DIR / "edges").glob("*.json")
        }
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def dispatch(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        **_: Any,
    ) -> dict[str, Any]:
        self.calls.append((tool_name, arguments))
        if tool_name == "get_entity":
            record = self.entities.get(arguments["entity_id"])
            records = [record] if record else []
        elif tool_name == "search_entities":
            query = arguments["text"].casefold()
            expected = arguments.get("entity_type")
            records = [
                self._summary(record)
                for record in self.entities.values()
                if (expected is None or record["id"].startswith(f"{expected}:"))
                and any(query == alias.casefold() for alias in self._aliases(record))
            ][: arguments.get("limit", 25)]
        elif tool_name == "get_relationships":
            records = self._relationships(arguments)
        else:
            raise AssertionError(f"unexpected tool: {tool_name}")
        return {"ok": True, "tool": tool_name, "result": records}

    def _relationships(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        entity_id = arguments["entity_id"]
        resolved = self.registry.resolve(arguments["relation"])
        schema = resolved.schema
        requested = arguments.get("direction")
        if resolved.inverse and requested in {"in", "out"}:
            requested = "out" if requested == "in" else "in"
        records: list[dict[str, Any]] = []
        for raw in self.edges.get(schema.id, []):
            if not arguments.get("include_ended") and raw.get("end") is not None:
                continue
            is_out = raw.get("from") == entity_id
            is_in = raw.get("to") == entity_id
            if schema.symmetric:
                matches = is_out or is_in
            elif requested == "out":
                matches = is_out
            elif requested == "in":
                matches = is_in
            else:
                matches = is_out or is_in
            if not matches:
                continue
            related_id = raw["to"] if is_out else raw["from"]
            edge = dict(raw)
            edge["relation"] = schema.id
            edge["related_entity"] = self._summary(self.entities[related_id])
            records.append(edge)
        return records[: arguments.get("limit", 25)]

    @staticmethod
    def _aliases(record: dict[str, Any]) -> list[str]:
        value = record.get("name")
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str)]
        if isinstance(value, dict):
            return [item for item in value.values() if isinstance(item, str)]
        return []

    @staticmethod
    def _summary(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in record.items()
            if key in {"id", "name", "display_name", "gender"}
        }


@pytest.fixture
def dispatcher() -> FixtureGraphDispatcher:
    return FixtureGraphDispatcher()


@pytest.fixture
def service(dispatcher: FixtureGraphDispatcher) -> SemanticFactService:
    catalog = RuntimeSchemaCatalog.from_data_dir(DATA_DIR, dispatcher.registry)
    schema = SemanticSchemaRegistry(catalog)
    return SemanticFactService(HouseholdFactEngine(dispatcher, schema))


@pytest.fixture
def context() -> AgentRequestContext:
    return AgentRequestContext(
        caller_entity_id="person:jian_kuang",
        assistant_id="steward",
        assistant_display_name="老管家",
        household_id="address:fort_cerritos",
        current_time=datetime.fromisoformat("2026-09-02T12:00:00-07:00"),
        locale="zh",
    )


async def _ask(
    service: SemanticFactService,
    context: AgentRequestContext,
    question: str,
):
    answer = await service.try_answer(
        [{"role": "user", "content": question}],
        context=context,
    )
    assert answer is not None
    return answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "expected"),
    (
        ("我是谁", "匡健"),
        ("你是谁", "老管家"),
        ("家里都有谁", "匡健"),
        ("家里有几个人", "五个人"),
        ("我老婆是谁", "巴璞"),
        ("我生日是哪天", "1988-11-11"),
        ("我有几个孩子", "两个孩子"),
        ("我儿子是谁", "匡德伦"),
        ("我女儿是谁", "匡悠然"),
        ("我儿子几岁了", "9岁"),
        ("谁最年长", "巴志刚"),
        ("我和我老婆谁年龄大", "巴璞年龄比匡健大"),
        ("我老婆的生日是哪天", "1988-02-26"),
        ("我儿子的生日是哪天", "2016-10-30"),
        ("我老婆和我谁年龄大", "巴璞年龄比匡健大"),
        ("家里有几个孩子", "两个孩子"),
        ("我老婆的爸爸是谁", "您妻子的父亲是巴志刚"),
    ),
)
async def test_canonical_and_composed_facts_are_deterministic(
    service: SemanticFactService,
    context: AgentRequestContext,
    question: str,
    expected: str,
) -> None:
    answer = await _ask(service, context, question)

    assert answer.result.status == "found"
    assert expected in answer.text
    assert answer.timings.tier == 0
    assert answer.timings.llm_call_count == 0


@pytest.mark.asyncio
async def test_household_list_is_current_and_uses_one_graph_query(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: FixtureGraphDispatcher,
) -> None:
    answer = await _ask(service, context, "能告诉我一下家里的成员名单吗？")

    assert all(name in answer.text for name in ("匡健", "巴璞", "匡德伦", "匡悠然", "巴志刚"))
    assert "张玉梅" not in answer.text
    assert answer.timings.db_query_count == 1
    assert [name for name, _ in dispatcher.calls] == ["get_relationships"]


def test_birth_date_plan_never_contains_physical_alias() -> None:
    request = TierZeroSemanticParser().parse("我生日是哪天")

    assert isinstance(request, SemanticFactRequest)
    assert request.property == "birth_date"
    assert "dob" not in request.model_dump_json()


@pytest.mark.asyncio
async def test_birth_date_resolves_when_storage_uses_birthday(
    dispatcher: FixtureGraphDispatcher,
    context: AgentRequestContext,
) -> None:
    person = dispatcher.entities["person:jian_kuang"]
    person["birthday"] = person.pop("dob")
    catalog = RuntimeSchemaCatalog.from_data_dir(DATA_DIR, dispatcher.registry)
    person_schema = catalog.entities["person"]
    replacement = type(person_schema)(
        "person",
        tuple(field if field != "dob" else "birthday" for field in person_schema.properties),
    )
    catalog = RuntimeSchemaCatalog(
        {**catalog.entities, "person": replacement},
        catalog.relations,
    )
    semantic = SemanticFactService(
        HouseholdFactEngine(dispatcher, SemanticSchemaRegistry(catalog))
    )

    answer = await _ask(semantic, context, "我生日是哪天")

    assert answer.result.status == "found"
    assert "1988-11-11" in answer.text


@pytest.mark.asyncio
async def test_missing_birth_date_is_semantic_and_does_not_hallucinate(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: FixtureGraphDispatcher,
) -> None:
    dispatcher.entities["person:jian_kuang"].pop("dob")

    answer = await _ask(service, context, "我生日是哪天")

    assert answer.result.status == "property_unavailable"
    assert "出生日期" in answer.text
    assert "dob" not in answer.text
    assert "1988" not in answer.text


@pytest.mark.asyncio
async def test_absent_relationship_is_not_reported_as_missing_entity(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: FixtureGraphDispatcher,
) -> None:
    dispatcher.edges["spouse_of"] = []

    answer = await _ask(service, context, "我老婆是谁")

    assert answer.result.status == "relationship_not_found"
    assert "配偶关系" in answer.text


@pytest.mark.asyncio
async def test_named_entity_with_absent_spouse_preserves_relationship_status(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: FixtureGraphDispatcher,
) -> None:
    dispatcher.edges["spouse_of"] = []
    request = SemanticFactRequest(
        operation="resolve_entity",
        subject=SemanticReference(
            kind="named_entity",
            value="匡健",
            entity_type="person",
            path=(
                SemanticRelationStep(
                    relation="spouse",
                    filters=(SemanticFilter(property="gender", value="female"),),
                ),
            ),
        ),
    )

    result, _, _, _ = await service.engine.execute(request, context)

    assert result.status == "relationship_not_found"
    assert result.evidence.relationship == "spouse"


@pytest.mark.asyncio
async def test_multiple_matching_children_are_ambiguous(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: FixtureGraphDispatcher,
) -> None:
    dispatcher.entities["person:second_son"] = {
        "id": "person:second_son",
        "name": ["Second Son", "次子"],
        "gender": "male",
        "dob": "2021-01-01",
    }
    dispatcher.edges["parent_of"].append(
        {"from": "person:jian_kuang", "to": "person:second_son", "type": "father"}
    )

    answer = await _ask(service, context, "我儿子是谁")

    assert answer.result.status == "ambiguous"
    assert "匡德伦" in answer.text
    assert "次子" in answer.text


@pytest.mark.asyncio
async def test_assistant_identity_never_queries_household_graph(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: FixtureGraphDispatcher,
) -> None:
    answer = await _ask(service, context, "你是谁")

    assert answer.text == "我是老管家。"
    assert dispatcher.calls == []
    assert answer.timings.db_query_count == 0


@pytest.mark.asyncio
async def test_relationship_evidence_preserves_temporal_metadata(
    service: SemanticFactService,
    context: AgentRequestContext,
) -> None:
    answer = await _ask(service, context, "我老婆是谁")

    assert answer.result.evidence.relationships
    relationship = answer.result.evidence.relationships[0]
    assert relationship.relation == "spouse"
    assert relationship.start == "2014-05-04"
    assert relationship.end is None


def test_capabilities_are_semantic_and_new_fields_require_no_handler(
    dispatcher: FixtureGraphDispatcher,
) -> None:
    catalog = RuntimeSchemaCatalog.from_data_dir(DATA_DIR, dispatcher.registry)
    person = catalog.entities["person"]
    augmented = type(person)(
        "person",
        (*person.properties, "favorite_color"),
        {**person.property_types, "favorite_color": "string"},
    )
    schema = SemanticSchemaRegistry(
        RuntimeSchemaCatalog({**catalog.entities, "person": augmented}, catalog.relations)
    )

    assert schema.physical_property("person", "favorite_color") == "favorite_color"
    assert "favorite_color" in schema.capability_payload()["semantic_properties"]["person"]
    assert "birth_date" in schema.capability_payload()["semantic_properties"]["person"]
    assert "dob" not in schema.capability_payload()["semantic_properties"]["person"]


def test_semantic_validator_rejects_type_incompatible_extreme(
    dispatcher: FixtureGraphDispatcher,
) -> None:
    catalog = RuntimeSchemaCatalog.from_data_dir(DATA_DIR, dispatcher.registry)
    schema = SemanticSchemaRegistry(catalog)
    parsed = TierZeroSemanticParser().parse("谁最年长")
    assert parsed is not None
    request = SemanticFactRequest(
        operation="argmin",
        subject=parsed.subject,
        property="display_name",
    )

    assert schema.validates(request) is False


@pytest.mark.asyncio
async def test_empty_child_collection_counts_as_zero_not_missing_person(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: FixtureGraphDispatcher,
) -> None:
    dispatcher.edges["parent_of"] = [
        edge
        for edge in dispatcher.edges["parent_of"]
        if edge["from"] != "person:jian_kuang"
    ]

    answer = await _ask(service, context, "我有几个孩子")

    assert answer.result.status == "found"
    assert answer.result.value == 0
    assert "零个孩子" in answer.text


@pytest.mark.asyncio
async def test_new_numeric_field_immediately_supports_generic_argmax(
    dispatcher: FixtureGraphDispatcher,
    context: AgentRequestContext,
) -> None:
    household_ids = {
        "person:jian_kuang",
        "person:pu_ba",
        "person:dylan_kuang",
        "person:evelyn_kuang",
        "person:zhigang_ba",
    }
    for index, entity_id in enumerate(sorted(household_ids), start=1):
        dispatcher.entities[entity_id]["fixture_score"] = index
    catalog = RuntimeSchemaCatalog.from_data_dir(DATA_DIR, dispatcher.registry)
    person = catalog.entities["person"]
    augmented = type(person)(
        "person",
        (*person.properties, "fixture_score"),
        {**person.property_types, "fixture_score": "integer"},
    )
    schema = SemanticSchemaRegistry(
        RuntimeSchemaCatalog({**catalog.entities, "person": augmented}, catalog.relations)
    )
    engine = HouseholdFactEngine(dispatcher, schema)
    parsed = TierZeroSemanticParser().parse("谁最年长")
    assert parsed is not None
    request = SemanticFactRequest(
        operation="argmax",
        subject=parsed.subject,
        property="fixture_score",
    )

    result, _, _, _ = await engine.execute(request, context)

    assert schema.validates(request) is True
    assert result.status == "found"
    assert result.value["id"] == max(household_ids)


@pytest.mark.asyncio
async def test_tier_one_uses_one_semantic_call_then_deterministic_execution(
    dispatcher: FixtureGraphDispatcher,
    context: AgentRequestContext,
) -> None:
    catalog = RuntimeSchemaCatalog.from_data_dir(DATA_DIR, dispatcher.registry)
    schema = SemanticSchemaRegistry(catalog)

    class Interpreter:
        def __init__(self) -> None:
            self.calls = 0
            self.capabilities: dict[str, Any] | None = None

        async def plan_semantic_fact(
            self,
            _messages: Any,
            capabilities: dict[str, Any],
            _output_schema: Any,
            **_: Any,
        ) -> dict[str, Any]:
            self.calls += 1
            self.capabilities = capabilities
            return {
                "requires_fact": True,
                "request": {
                    "operation": "argmin",
                    "subject": {
                        "kind": "current_household",
                        "entity_type": "address",
                        "path": [{"relation": "member", "filters": []}],
                    },
                    "property": "birth_date",
                },
            }

    interpreter = Interpreter()
    semantic = SemanticFactService(
        HouseholdFactEngine(dispatcher, schema),
        planner=SemanticFactPlanner(interpreter, schema),
    )

    answer = await _ask(semantic, context, "家里哪位成员出生最早")

    assert "巴志刚" in answer.text
    assert answer.timings.tier == 1
    assert answer.timings.llm_call_count == 1
    assert interpreter.calls == 1
    assert interpreter.capabilities is not None
    assert "dob" not in json.dumps(interpreter.capabilities)


@pytest.mark.asyncio
async def test_tier_one_rejects_unadvertised_semantic_property(
    dispatcher: FixtureGraphDispatcher,
    context: AgentRequestContext,
) -> None:
    catalog = RuntimeSchemaCatalog.from_data_dir(DATA_DIR, dispatcher.registry)
    schema = SemanticSchemaRegistry(catalog)

    class Interpreter:
        async def plan_semantic_fact(self, *_: Any, **__: Any) -> dict[str, Any]:
            return {
                "requires_fact": True,
                "request": {
                    "operation": "get_property",
                    "subject": {"kind": "self", "entity_type": "person"},
                    "property": "invented_private_fact",
                },
            }

    semantic = SemanticFactService(
        HouseholdFactEngine(dispatcher, schema),
        planner=SemanticFactPlanner(Interpreter(), schema),
    )

    answer = await _ask(semantic, context, "我有什么秘密家庭属性")

    assert answer.result.status == "computation_impossible"
    assert answer.timings.llm_call_count == 1
    assert dispatcher.calls == []
