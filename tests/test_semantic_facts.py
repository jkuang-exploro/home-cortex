import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from home_cortex.agent_service import AgentService
from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.grounding import AgentRequestContext
from home_cortex.operator_registry import OPERATORS
from home_cortex.schema_catalog import (
    RuntimeSchemaCatalog,
    matches_scoped_appellation,
    normalize_entity_alias,
    record_aliases,
)
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
from home_cortex.tools import get_tool_definitions

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
        elif tool_name in {"search_entities", "resolve_entity_alias"}:
            query = arguments["text"].casefold()
            expected = arguments.get("entity_type")
            candidates = [
                record
                for record in self.entities.values()
                if expected is None or record["id"].startswith(f"{expected}:")
            ]
            aliases = [
                self._summary(record)
                for record in candidates
                if any(
                    normalize_entity_alias(query) == normalize_entity_alias(alias)
                    for alias in record_aliases(record)
                )
            ]
            appellations = [
                self._summary(record)
                for record in candidates
                if matches_scoped_appellation(
                    record,
                    arguments["text"],
                    speaker_id=arguments.get("speaker_id"),
                    household_id=arguments.get("household_id"),
                )
            ]
            records = (aliases or appellations)[: arguments.get("limit", 25)]
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
        ("家里都有哪些人", "匡健"),
        ("家里有几个人", "五个人"),
        ("我老婆是谁", "巴璞"),
        ("我生日是哪天", "1988-11-11"),
        ("我有几个孩子", "两个孩子"),
        ("我儿子是谁", "匡德伦"),
        ("我女儿是谁", "匡悠然"),
        ("我儿子几岁了", "9岁"),
        ("我儿子几岁", "9岁"),
        ("谁最年长", "巴志刚"),
        ("我和我老婆谁年龄大", "巴璞年龄比匡健大"),
        ("我老婆的生日是哪天", "1988-02-26"),
        ("我儿子的生日是哪天", "2016-10-30"),
        ("我老婆和我谁年龄大", "巴璞年龄比匡健大"),
        ("家里有几个孩子", "两个孩子"),
        ("我老婆的爸爸是谁", "您妻子的父亲是巴志刚"),
        ("我老婆的父亲是谁", "您妻子的父亲是巴志刚"),
        ("巴璞的父亲是谁", "巴璞的父亲是巴志刚"),
        ("巴志刚的女儿是谁", "巴志刚的女儿是巴璞"),
        ("我岳父是谁", "您配偶的父亲是巴志刚"),
        ("我岳母是谁", "您配偶的母亲是张玉梅"),
        ("匡德伦的生日是哪天", "匡德伦的出生日期是2016-10-30"),
        ("匡德伦哪天出生", "匡德伦的出生日期是2016-10-30"),
        ("匡德伦的出生日期是什么", "匡德伦的出生日期是2016-10-30"),
        ("when was Dylan Kuang born", "2016-10-30"),
        ("what is my date of birth", "1988-11-11"),
        ("巴璞哪天过生日", "巴璞的下次生日是2027-02-26"),
        ("我儿子哪天过生日", "您儿子的下次生日是2026-10-30"),
        ("我儿子的生日还有多少天", "您儿子的生日还有58天"),
        ("距离我儿子生日还有几天", "您儿子的生日还有58天"),
        ("我老婆生日还有多少天", "您妻子的生日还有177天"),
        ("我家住哪里", "12745 Droxford St, Cerritos, CA 90703"),
        ("请问我的具体住址是什么？", "12745 Droxford St, Cerritos, CA 90703"),
        ("街道地址", "12745 Droxford St, Cerritos, CA 90703"),
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


@pytest.mark.asyncio
async def test_all_static_and_relational_references_converge_on_canonical_person(
    service: SemanticFactService,
    context: AgentRequestContext,
) -> None:
    questions = (
        "匡德伦是谁",
        "德伦是谁",
        "Dylan是谁",
        "Dylan Kuang是谁",
        "我儿子是谁",
        "巴璞的儿子是谁",
    )

    answers = [await _ask(service, context, question) for question in questions]

    assert all(answer.result.status == "found" for answer in answers)
    assert all(
        answer.result.evidence.entity_ids == ("person:dylan_kuang",)
        for answer in answers
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("speaker_id", "question"),
    (
        ("person:jian_kuang", "我儿子是谁"),
        ("person:pu_ba", "我儿子是谁"),
        ("person:guiqiu_wang", "我孙子是谁"),
        ("person:zhigang_ba", "我外孙是谁"),
        ("person:evelyn_kuang", "我哥哥是谁"),
    ),
)
async def test_speaker_relative_kinship_converges_on_dylan(
    service: SemanticFactService,
    context: AgentRequestContext,
    speaker_id: str,
    question: str,
) -> None:
    answer = await _ask(
        service,
        replace(context, caller_entity_id=speaker_id),
        question,
    )

    assert answer.result.status == "found"
    assert answer.result.evidence.entity_ids == ("person:dylan_kuang",)


@pytest.mark.asyncio
async def test_same_self_relation_resolves_from_each_active_speaker(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: FixtureGraphDispatcher,
) -> None:
    dispatcher.entities.update(
        {
            "person:other_parent": {
                "id": "person:other_parent",
                "name": ["Other Parent"],
                "gender": "female",
                "dob": "1985-01-01",
            },
            "person:other_son": {
                "id": "person:other_son",
                "name": ["Other Son"],
                "gender": "male",
                "dob": "2018-01-01",
            },
        }
    )
    dispatcher.edges["parent_of"].append(
        {"from": "person:other_parent", "to": "person:other_son", "type": "mother"}
    )

    jian = await _ask(service, context, "我儿子是谁")
    other = await _ask(
        service,
        replace(context, caller_entity_id="person:other_parent"),
        "我儿子是谁",
    )

    assert jian.result.evidence.entity_ids == ("person:dylan_kuang",)
    assert other.result.evidence.entity_ids == ("person:other_son",)


@pytest.mark.asyncio
async def test_multi_match_grandson_is_ambiguous(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: FixtureGraphDispatcher,
) -> None:
    dispatcher.entities["person:second_grandson"] = {
        "id": "person:second_grandson",
        "name": ["次孙"],
        "gender": "male",
        "dob": "2022-01-01",
    }
    dispatcher.edges["parent_of"].append(
        {
            "from": "person:jian_kuang",
            "to": "person:second_grandson",
            "type": "father",
        }
    )

    answer = await _ask(
        service,
        replace(context, caller_entity_id="person:guiqiu_wang"),
        "我孙子是谁",
    )

    assert answer.result.status == "ambiguous"
    assert {item["id"] for item in answer.result.candidates} == {
        "person:dylan_kuang",
        "person:second_grandson",
    }


@pytest.mark.asyncio
async def test_scoped_appellation_is_grounded_by_resolver_context(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: FixtureGraphDispatcher,
) -> None:
    dispatcher.entities["person:dylan_kuang"]["appellations"] = [
        {
            "value": "大宝",
            "household_id": "address:fort_cerritos",
            "speaker_ids": ["person:jian_kuang"],
        }
    ]

    resolved = await _ask(service, context, "大宝是谁")
    unscoped = await _ask(
        service,
        replace(context, caller_entity_id="person:pu_ba"),
        "大宝是谁",
    )

    assert resolved.result.evidence.entity_ids == ("person:dylan_kuang",)
    assert unscoped.result.status == "entity_not_found"


@pytest.mark.asyncio
async def test_empty_household_list_has_a_clear_response(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: FixtureGraphDispatcher,
) -> None:
    dispatcher.edges["lives_in"] = []

    answer = await _ask(service, context, "家里都有哪些人")

    assert answer.result.status == "found"
    assert answer.result.value == []
    assert answer.text == "家庭资料中目前没有记录当前家庭成员。"


def test_birth_date_plan_never_contains_physical_alias() -> None:
    request = TierZeroSemanticParser().parse("我生日是哪天")

    assert isinstance(request, SemanticFactRequest)
    assert request.property == "birth_date"
    assert "dob" not in request.model_dump_json()


def test_birthday_intents_have_distinct_generic_plans() -> None:
    parser = TierZeroSemanticParser()

    birth_date = parser.parse("匡德伦的生日是哪天")
    age = parser.parse("我儿子几岁了")
    next_birthday = parser.parse("我儿子哪天过生日")
    days_until = parser.parse("我儿子的生日还有多少天")

    assert birth_date is not None and birth_date.operation == "select"
    assert age is not None and age.operation == "completed_years"
    assert next_birthday is not None
    assert next_birthday.operation == "annual_occurrence"
    assert next_birthday.mode is None
    assert days_until is not None
    assert days_until.operation == "annual_occurrence"
    assert days_until.mode == "days"
    assert all(
        request.property == "birth_date"
        for request in (birth_date, age, next_birthday, days_until)
    )


def test_in_law_is_composed_from_existing_generic_relations() -> None:
    request = TierZeroSemanticParser().parse("我岳父是谁")

    assert request is not None
    assert [step.relation for step in request.subject.path] == ["spouse", "parent"]
    assert request.subject.path[-1].filters[0].property == "gender"
    assert request.subject.path[-1].filters[0].value == "male"


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
async def test_missing_birth_date_is_computation_input_missing_for_transform(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: FixtureGraphDispatcher,
) -> None:
    dispatcher.entities["person:dylan_kuang"].pop("dob")

    answer = await _ask(service, context, "我儿子的生日还有多少天")

    assert answer.result.status == "computation_input_missing"
    assert answer.result.missing_requirements == ("birth_date",)
    assert "出生日期" in answer.text


@pytest.mark.asyncio
async def test_named_person_missing_birth_date_is_property_unavailable_not_computation(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: FixtureGraphDispatcher,
) -> None:
    dispatcher.entities["person:dylan_kuang"].pop("dob")

    answer = await _ask(service, context, "匡德伦哪天出生")

    assert answer.result.status == "property_unavailable"
    assert answer.result.missing_requirements == ("birth_date",)
    assert "匡德伦的出生日期" in answer.text
    assert "计算" not in answer.text


@pytest.mark.asyncio
async def test_invalid_birth_date_is_computation_impossible(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: FixtureGraphDispatcher,
) -> None:
    dispatcher.entities["person:dylan_kuang"]["dob"] = "not-a-date"

    answer = await _ask(service, context, "我儿子的生日还有多少天")

    assert answer.result.status == "computation_impossible"
    assert "家庭资料不足以完成" in answer.text


@pytest.mark.asyncio
async def test_missing_named_person_is_entity_not_found(
    service: SemanticFactService,
    context: AgentRequestContext,
) -> None:
    answer = await _ask(service, context, "不存在的人哪天出生")

    assert answer.result.status == "entity_not_found"
    assert "没有找到对应的人或实体" in answer.text


@pytest.mark.asyncio
async def test_duplicate_exact_name_is_ambiguous(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: FixtureGraphDispatcher,
) -> None:
    dispatcher.entities["person:other_dylan"] = {
        "id": "person:other_dylan",
        "name": ["Other Dylan", "匡德伦"],
        "gender": "male",
        "dob": "2001-01-01",
    }

    answer = await _ask(service, context, "匡德伦哪天出生")

    assert answer.result.status == "ambiguous"
    assert "找到多个符合条件" in answer.text


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
async def test_absent_in_law_path_is_relationship_not_found(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: FixtureGraphDispatcher,
) -> None:
    dispatcher.edges["spouse_of"] = []

    answer = await _ask(service, context, "我岳父是谁")

    assert answer.result.status == "relationship_not_found"
    assert answer.result.evidence.relationship == "spouse"


@pytest.mark.asyncio
async def test_named_entity_with_absent_spouse_preserves_relationship_status(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: FixtureGraphDispatcher,
) -> None:
    dispatcher.edges["spouse_of"] = []
    request = SemanticFactRequest(
        operation="resolve_reference",
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
    assert "first_name" not in schema.capability_payload()["semantic_properties"]["person"]
    assert "last_name" not in schema.capability_payload()["semantic_properties"]["person"]
    assert "aliases" not in schema.capability_payload()["semantic_properties"]["person"]
    assert "appellations" not in schema.capability_payload()["semantic_properties"]["person"]
    assert "parent_of" not in json.dumps(schema.capability_payload())
    assert schema.physical_property("person", "dob") is None
    assert schema.physical_relation("parent_of") is None
    contracts = schema.capability_payload()["operator_contracts"]
    assert contracts["average"]["field_types"] == ["integer", "number"]
    assert contracts["completed_years"]["output"] == "integer"
    assert set(schema.capability_payload()["operations"]).issubset(OPERATORS)


@pytest.mark.asyncio
async def test_tier_one_can_select_a_new_schema_field_without_a_fact_handler(
    dispatcher: FixtureGraphDispatcher,
    context: AgentRequestContext,
) -> None:
    dispatcher.entities["person:jian_kuang"]["favorite_color"] = "green"
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

    class Interpreter:
        async def plan_semantic_fact(self, *_: Any, **__: Any) -> dict[str, Any]:
            return {
                "requires_fact": True,
                "request": {
                    "operation": "select",
                    "subject": {"kind": "self", "entity_type": "person"},
                    "property": "favorite_color",
                },
            }

    semantic = SemanticFactService(
        HouseholdFactEngine(dispatcher, schema),
        planner=SemanticFactPlanner(Interpreter(), schema),
    )

    answer = await _ask(semantic, context, "我的偏爱颜色是什么？")

    assert answer.result.status == "found"
    assert answer.result.value == "green"
    assert answer.timings.llm_call_count == 1


def test_semantic_ir_rejects_non_allowlisted_operation() -> None:
    with pytest.raises(ValidationError):
        SemanticFactRequest.model_validate(
            {
                "operation": "GET_AGE",
                "subject": {"kind": "self", "entity_type": "person"},
            }
        )


@pytest.mark.asyncio
async def test_agent_service_reported_queries_never_reach_legacy_planner(
    dispatcher: FixtureGraphDispatcher,
) -> None:
    catalog = RuntimeSchemaCatalog.from_data_dir(DATA_DIR, dispatcher.registry)

    class NoLegacyPlanner:
        async def plan_grounding(self, *_: Any, **__: Any) -> dict[str, Any]:
            raise AssertionError("legacy physical-field planner was invoked")

    agent = AgentService(
        NoLegacyPlanner(),  # type: ignore[arg-type]
        dispatcher,  # type: ignore[arg-type]
        system_prompt="You are the household steward.",
        tools=get_tool_definitions(("get_entity",)),
        schema_catalog=catalog,
        localized_identity={"zh": "老管家"},
        assistant_id="steward",
        home_entity_id="address:fort_cerritos",
        clock=lambda: datetime.fromisoformat("2026-09-03T12:00:00-07:00"),
    )
    identity = {"id": "person:jian_kuang", "name": ["Jian Kuang", "匡健"]}

    answers = {
        question: (
            await agent.answer(question, user_entity=identity)
        ).answer
        for question in (
            "我是谁",
            "家里都有哪些人",
            "我家住哪里",
            "请问我的具体住址是什么？",
            "街道地址",
        )
    }

    assert "匡健" in answers["我是谁"]
    assert "巴璞" in answers["家里都有哪些人"]
    assert all(
        "12745 Droxford St" in answers[question]
        for question in ("我家住哪里", "请问我的具体住址是什么？", "街道地址")
    )


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

    invalid_birthday = SemanticFactRequest(
        operation="annual_occurrence",
        subject=SemanticReference(kind="self", entity_type="person"),
        property="display_name",
    )
    assert schema.validates(invalid_birthday) is False


def test_semantic_validator_rejects_context_reference_type_spoofing(
    dispatcher: FixtureGraphDispatcher,
) -> None:
    schema = SemanticSchemaRegistry(
        RuntimeSchemaCatalog.from_data_dir(DATA_DIR, dispatcher.registry)
    )

    assert schema.validates(
        SemanticFactRequest(
            operation="resolve_reference",
            subject=SemanticReference(kind="self", entity_type="address"),
        )
    ) is False
    assert schema.validates(
        SemanticFactRequest(
            operation="resolve_reference",
            subject=SemanticReference(
                kind="entity_id",
                value="person:jian_kuang",
                entity_type="address",
            ),
        )
    ) is False

    assert schema.validates(
        SemanticFactRequest(
            operation="count",
            subject=SemanticReference(kind="self", entity_type="person"),
        )
    ) is False
    assert schema.validates(
        SemanticFactRequest(
            operation="select",
            subject=SemanticReference(
                kind="current_household",
                entity_type="address",
            ),
            property="birth_date",
        )
    ) is False
    assert schema.validates(
        SemanticFactRequest(
            operation="select",
            subject=SemanticReference(kind="self", entity_type="person"),
            property="dob",
        )
    ) is False
    assert schema.validates(
        SemanticFactRequest(
            operation="resolve_reference",
            subject=SemanticReference(
                kind="self",
                entity_type="person",
                path=(SemanticRelationStep(relation="parent_of"),),
            ),
        )
    ) is False


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
async def test_registered_date_range_predicate_filters_relationships(
    service: SemanticFactService,
    context: AgentRequestContext,
) -> None:
    request = SemanticFactRequest(
        operation="count",
        subject=SemanticReference(
            kind="current_household",
            entity_type="address",
            path=(
                SemanticRelationStep(
                    relation="member",
                    filters=(
                        SemanticFilter(
                            property="start",
                            operator="date_range",
                            value=("2026-06-01", "2026-07-01"),
                            source="relation",
                        ),
                    ),
                ),
            ),
        ),
    )

    result, _, _, _ = await service.engine.execute(request, context)

    assert result.status == "found"
    assert result.value == 3


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

    total_request = SemanticFactRequest(
        operation="sum",
        subject=parsed.subject,
        property="fixture_score",
    )
    total, _, _, _ = await engine.execute(total_request, context)
    assert schema.validates(total_request) is True
    assert total.status == "found"
    assert total.value == 15


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
                    "operation": "select",
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


@pytest.mark.asyncio
async def test_tier_zero_disabled_preserves_semantic_correctness(
    dispatcher: FixtureGraphDispatcher,
    context: AgentRequestContext,
) -> None:
    catalog = RuntimeSchemaCatalog.from_data_dir(DATA_DIR, dispatcher.registry)
    schema = SemanticSchemaRegistry(catalog)

    def reference(
        kind: str,
        *,
        value: str | None = None,
        path: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "kind": kind,
            "value": value,
            "entity_type": "address" if kind == "current_household" else "person",
            "path": path or [],
        }

    son = [
        {
            "relation": "child",
            "filters": [{"property": "gender", "value": "male"}],
        }
    ]
    wife = [
        {
            "relation": "spouse",
            "filters": [{"property": "gender", "value": "female"}],
        }
    ]
    father_in_law = [
        {"relation": "spouse", "filters": []},
        {
            "relation": "parent",
            "filters": [{"property": "gender", "value": "male"}],
        },
    ]
    members = [{"relation": "member", "filters": []}]
    plans: dict[str, dict[str, Any]] = {
        "我是谁": {"operation": "resolve_reference", "subject": reference("self")},
        "匡德伦是谁": {
            "operation": "resolve_reference",
            "subject": reference("named_entity", value="匡德伦"),
        },
        "德伦是谁": {
            "operation": "resolve_reference",
            "subject": reference("named_entity", value="德伦"),
        },
        "Dylan是谁": {
            "operation": "resolve_reference",
            "subject": reference("named_entity", value="Dylan"),
        },
        "我儿子是谁": {
            "operation": "resolve_reference",
            "subject": reference("self", path=son),
        },
        "巴璞的儿子是谁": {
            "operation": "resolve_reference",
            "subject": reference("named_entity", value="巴璞", path=son),
        },
        "我老婆是谁": {
            "operation": "resolve_reference",
            "subject": reference("self", path=wife),
        },
        "我岳父是谁": {
            "operation": "resolve_reference",
            "subject": reference("self", path=father_in_law),
        },
        "我儿子哪天出生": {
            "operation": "select",
            "subject": reference("self", path=son),
            "property": "birth_date",
        },
        "我儿子的生日还有多少天": {
            "operation": "annual_occurrence",
            "subject": reference("self", path=son),
            "property": "birth_date",
            "mode": "days",
        },
        "家里有几个人": {
            "operation": "count",
            "subject": reference("current_household", path=members),
        },
        "家里谁最年长": {
            "operation": "argmin",
            "subject": reference("current_household", path=members),
            "property": "birth_date",
        },
        "我和我老婆谁年龄大": {
            "operation": "argmin",
            "subject": reference("self"),
            "other": reference("self", path=wife),
            "property": "birth_date",
        },
    }

    class Interpreter:
        async def plan_semantic_fact(
            self,
            messages: list[dict[str, Any]],
            *_: Any,
            **__: Any,
        ) -> dict[str, Any]:
            question = messages[-1]["content"]
            return {"requires_fact": True, "request": plans[question]}

    class DisabledParser:
        def parse(self, _text: str) -> None:
            raise AssertionError("Tier 0 must be bypassed")

    semantic = SemanticFactService(
        HouseholdFactEngine(dispatcher, schema),
        parser=DisabledParser(),  # type: ignore[arg-type]
        planner=SemanticFactPlanner(Interpreter(), schema),
        tier_zero_enabled=False,
    )

    answers = {
        question: await _ask(semantic, context, question) for question in plans
    }

    dylan_queries = (
        "匡德伦是谁",
        "德伦是谁",
        "Dylan是谁",
        "我儿子是谁",
        "巴璞的儿子是谁",
        "我儿子哪天出生",
        "我儿子的生日还有多少天",
    )
    assert all(
        answers[question].result.evidence.entity_ids == ("person:dylan_kuang",)
        for question in dylan_queries
    )
    assert answers["我是谁"].result.evidence.entity_ids == ("person:jian_kuang",)
    assert answers["我老婆是谁"].result.evidence.entity_ids == ("person:pu_ba",)
    assert answers["我岳父是谁"].result.evidence.entity_ids == (
        "person:zhigang_ba",
    )
    assert answers["家里有几个人"].result.value == 5
    assert all(answer.timings.tier == 1 for answer in answers.values())
    assert all(answer.timings.llm_call_count == 1 for answer in answers.values())


@pytest.mark.asyncio
async def test_tier_zero_disabled_preserves_multi_speaker_kinship(
    dispatcher: FixtureGraphDispatcher,
    context: AgentRequestContext,
) -> None:
    schema = SemanticSchemaRegistry(
        RuntimeSchemaCatalog.from_data_dir(DATA_DIR, dispatcher.registry)
    )
    paths = {
        "我儿子是谁": [
            {
                "relation": "child",
                "filters": [{"property": "gender", "value": "male"}],
            }
        ],
        "我孙子是谁": [
            {
                "relation": "child",
                "filters": [{"property": "gender", "value": "male"}],
            },
            {
                "relation": "child",
                "filters": [{"property": "gender", "value": "male"}],
            },
        ],
        "我外孙是谁": [
            {
                "relation": "child",
                "filters": [{"property": "gender", "value": "female"}],
            },
            {
                "relation": "child",
                "filters": [{"property": "gender", "value": "male"}],
            },
        ],
        "我哥哥是谁": [
            {"relation": "parent", "filters": []},
            {
                "relation": "child",
                "filters": [
                    {"property": "gender", "value": "male"},
                    {
                        "property": "birth_date",
                        "operator": "lt",
                        "value_from": "anchor",
                    },
                ],
            },
        ],
    }

    class Interpreter:
        async def plan_semantic_fact(
            self,
            messages: list[dict[str, Any]],
            *_: Any,
            **__: Any,
        ) -> dict[str, Any]:
            return {
                "requires_fact": True,
                "request": {
                    "operation": "resolve_reference",
                    "subject": {
                        "kind": "self",
                        "entity_type": "person",
                        "path": paths[messages[-1]["content"]],
                    },
                },
            }

    semantic = SemanticFactService(
        HouseholdFactEngine(dispatcher, schema),
        planner=SemanticFactPlanner(Interpreter(), schema),
        tier_zero_enabled=False,
    )
    cases = (
        ("person:jian_kuang", "我儿子是谁"),
        ("person:pu_ba", "我儿子是谁"),
        ("person:guiqiu_wang", "我孙子是谁"),
        ("person:zhigang_ba", "我外孙是谁"),
        ("person:evelyn_kuang", "我哥哥是谁"),
    )

    answers = [
        await _ask(
            semantic,
            replace(context, caller_entity_id=speaker_id),
            question,
        )
        for speaker_id, question in cases
    ]

    assert all(
        answer.result.evidence.entity_ids == ("person:dylan_kuang",)
        for answer in answers
    )
    assert all(answer.timings.tier == 1 for answer in answers)


@pytest.mark.asyncio
async def test_tier_one_open_world_paraphrases_use_resolver_not_entity_ids(
    dispatcher: FixtureGraphDispatcher,
    context: AgentRequestContext,
) -> None:
    catalog = RuntimeSchemaCatalog.from_data_dir(DATA_DIR, dispatcher.registry)
    schema = SemanticSchemaRegistry(catalog)
    outputs = {
        "Dylan是哪位？": {
            "kind": "named_entity",
            "value": "Dylan",
            "entity_type": "person",
        },
        "德伦是哪一个人？": {
            "kind": "named_entity",
            "value": "德伦",
            "entity_type": "person",
        },
        "巴璞她儿子是谁？": {
            "kind": "named_entity",
            "value": "巴璞",
            "entity_type": "person",
            "path": [
                {
                    "relation": "child",
                    "filters": [{"property": "gender", "value": "male"}],
                }
            ],
        },
        "我家那个叫Dylan的孩子是谁？": {
            "kind": "named_entity",
            "value": "Dylan",
            "entity_type": "person",
        },
        "我妻子的父亲是谁？": {
            "kind": "self",
            "entity_type": "person",
            "path": [
                {
                    "relation": "spouse",
                    "filters": [{"property": "gender", "value": "female"}],
                },
                {
                    "relation": "parent",
                    "filters": [{"property": "gender", "value": "male"}],
                },
            ],
        },
    }

    class Interpreter:
        async def plan_semantic_fact(
            self,
            messages: list[dict[str, Any]],
            *_: Any,
            **__: Any,
        ) -> dict[str, Any]:
            return {
                "requires_fact": True,
                "request": {
                    "operation": "resolve_reference",
                    "subject": outputs[messages[-1]["content"]],
                },
            }

    semantic = SemanticFactService(
        HouseholdFactEngine(dispatcher, schema),
        planner=SemanticFactPlanner(Interpreter(), schema),
        tier_zero_enabled=False,
    )

    answers = [await _ask(semantic, context, question) for question in outputs]

    assert all(answer.result.status == "found" for answer in answers)
    assert [answer.result.evidence.entity_ids for answer in answers] == [
        ("person:dylan_kuang",),
        ("person:dylan_kuang",),
        ("person:dylan_kuang",),
        ("person:dylan_kuang",),
        ("person:zhigang_ba",),
    ]


@pytest.mark.asyncio
async def test_semantic_planner_rejects_model_originated_entity_id(
    dispatcher: FixtureGraphDispatcher,
    context: AgentRequestContext,
) -> None:
    schema = SemanticSchemaRegistry(
        RuntimeSchemaCatalog.from_data_dir(DATA_DIR, dispatcher.registry)
    )

    class Interpreter:
        async def plan_semantic_fact(self, *_: Any, **__: Any) -> dict[str, Any]:
            return {
                "requires_fact": True,
                "request": {
                    "operation": "resolve_reference",
                    "subject": {
                        "kind": "entity_id",
                        "value": "person:dylan_kuang",
                        "entity_type": "person",
                    },
                },
            }

    with pytest.raises(ValueError, match="cannot originate entity IDs"):
        await SemanticFactPlanner(Interpreter(), schema).plan(
            [{"role": "user", "content": "Dylan是谁"}],
            context,
        )
