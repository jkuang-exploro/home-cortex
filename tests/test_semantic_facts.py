import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pytest
from ollama import ChatResponse
from pydantic import ValidationError
from surrealdb import AsyncSurreal

from home_cortex.agent_service import AgentService
from home_cortex.agents import get_agent
from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.fact_benchmark import _JsonGraphDispatcher
from home_cortex.ingestion import ingest_directory
from home_cortex.operator_registry import OPERATORS
from home_cortex.retrieval import RetrievalService
from home_cortex.schema_catalog import RuntimeSchemaCatalog
from home_cortex.semantic_facts import (
    AgentRequestContext,
    HouseholdFactEngine,
    SemanticFactPlanner,
    SemanticFactRequest,
    SemanticFactService,
    SemanticFilter,
    SemanticPlannerFailure,
    SemanticReference,
    SemanticRelationStep,
    SemanticSchemaRegistry,
    TierZeroSemanticParser,
)
from home_cortex.tools import ToolDispatcher, get_tool_definitions

ROOT = Path(__file__).parents[1]
DATA_DIR = ROOT / "data"
STATIC_TEST_DATA = Path(__file__).parent / "static_test_data"
STEWARD = get_agent("steward")


class _Interpreter:
    def __init__(self, payload: Any, *, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.calls = 0
        self.messages: list[list[dict[str, Any]]] = []
        self.capabilities: dict[str, Any] | None = None
        self.output_schema: dict[str, Any] | None = None

    async def plan_semantic_fact(
        self,
        messages: list[dict[str, Any]],
        capabilities: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        **_: Any,
    ) -> Any:
        self.calls += 1
        self.messages.append(messages)
        self.capabilities = capabilities
        self.output_schema = output_schema
        if self.error is not None:
            raise self.error
        payload = self.payload
        if callable(payload):
            payload = payload(self.calls, messages)
        if isinstance(payload, SemanticFactRequest):
            return {
                "requires_fact": True,
                "request": payload.model_dump(mode="json"),
            }
        return payload


class _FailingDispatcher:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        response: dict[str, Any] | None = None,
    ) -> None:
        self.error = error
        self.response = response or {
            "ok": False,
            "error": {
                "code": "tool_execution_failed",
                "message": "database password should not be returned",
            },
        }
        self.calls: list[tuple[str, Any]] = []

    async def dispatch_internal(
        self, tool_name: str, arguments: Any, **_: Any
    ) -> dict[str, Any]:
        self.calls.append((tool_name, arguments))
        if self.error is not None:
            raise self.error
        return self.response

    dispatch = dispatch_internal


class _MemoryDatabase:
    def __init__(self) -> None:
        self.client = AsyncSurreal("mem://")

    async def connect(self) -> None:
        await self.client.connect()
        await self.client.use("test", "test")

    async def close(self) -> None:
        await self.client.close()

    async def query(
        self,
        statement: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        return await self.client.query(statement, variables or {})

    async def upsert(self, record: Any, data: dict[str, Any]) -> Any:
        return await self.client.upsert(record, data)


class _ChatOllama:
    def __init__(self, payload: Any, *, error: Exception | None = None) -> None:
        self.interpreter = _Interpreter(payload, error=error)
        self.calls: list[list[dict[str, Any]]] = []

    async def plan_semantic_fact(self, *args: Any, **kwargs: Any) -> Any:
        return await self.interpreter.plan_semantic_fact(*args, **kwargs)

    async def chat_with_tools(
        self, messages: list[dict[str, Any]], tools: Any
    ) -> ChatResponse:
        self.calls.append([dict(message) for message in messages])
        return ChatResponse.model_validate(
            {
                "model": "qwen3:8b",
                "created_at": "2026-08-31T00:00:00Z",
                "done": True,
                "message": {"role": "assistant", "content": "casual conversation"},
            }
        )


def _schema(data_dir: Path) -> SemanticSchemaRegistry:
    registry = EdgeSchemaRegistry.load_default(data_dir)
    return SemanticSchemaRegistry(RuntimeSchemaCatalog.from_data_dir(data_dir, registry))


def _context(
    *,
    caller_entity_id: str | None = "person:jian_kuang",
    household_id: str | None = "address:fort_cerritos",
    locale: str = "zh",
    current_time: str = "2026-09-02T12:00:00-07:00",
) -> AgentRequestContext:
    return AgentRequestContext(
        caller_entity_id=caller_entity_id,
        assistant_id="steward",
        assistant_display_name="老管家",
        household_id=household_id,
        current_time=datetime.fromisoformat(current_time),
        locale=locale,
    )


def _static_context(
    caller_entity_id: str | None = "person:alex_example",
    **overrides: Any,
) -> AgentRequestContext:
    return _context(
        caller_entity_id=caller_entity_id,
        household_id="address:test_house",
        **overrides,
    )


def _self(*steps: SemanticRelationStep) -> SemanticReference:
    return SemanticReference(kind="self", entity_type="person", path=steps)


def _named(value: str, *steps: SemanticRelationStep) -> SemanticReference:
    return SemanticReference(
        kind="named_entity",
        value=value,
        entity_type="person",
        path=steps,
    )


def _members() -> SemanticReference:
    return SemanticReference(
        kind="current_household",
        entity_type="address",
        path=(SemanticRelationStep(relation="member"),),
    )


def _spouse() -> SemanticReference:
    return _self(SemanticRelationStep(relation="spouse"))


def _son() -> SemanticRelationStep:
    return SemanticRelationStep(
        relation="child",
        filters=(SemanticFilter(property="gender", value="male"),),
    )


def _daughter() -> SemanticRelationStep:
    return SemanticRelationStep(
        relation="child",
        filters=(SemanticFilter(property="gender", value="female"),),
    )


def _parent(gender: str) -> SemanticRelationStep:
    return SemanticRelationStep(
        relation="parent",
        filters=(SemanticFilter(property="gender", value=gender),),
    )


def _resolve(subject: SemanticReference) -> SemanticFactRequest:
    return SemanticFactRequest(operation="resolve_reference", subject=subject)


def _select(
    subject: SemanticReference,
    property_name: str,
    *,
    property_source: str = "entity",
) -> SemanticFactRequest:
    return SemanticFactRequest(
        operation="select",
        subject=subject,
        property=property_name,
        property_source=property_source,  # type: ignore[arg-type]
    )


def _candidate_ids(result: Any) -> set[str]:
    return {
        str(item["id"])
        for item in result.candidates
        if isinstance(item, Mapping) and item.get("id")
    }


def _leaked(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "password",
            "boom",
            "runtimeerror",
            "traceback",
            "select *",
            "surreal",
        )
    )


def _service(
    dispatcher: Any,
    payload: Any = None,
    *,
    data_dir: Path = DATA_DIR,
    schema: SemanticSchemaRegistry | None = None,
    tier_zero_enabled: bool = True,
    parser: Any = None,
    error: Exception | None = None,
) -> SemanticFactService:
    schema = schema or _schema(data_dir)
    return SemanticFactService(
        HouseholdFactEngine(dispatcher, schema),
        planner=SemanticFactPlanner(
            _Interpreter(
                payload
                if payload is not None
                else {"requires_fact": False, "request": None},
                error=error,
            ),
            schema,
        ),
        parser=parser,
        tier_zero_enabled=tier_zero_enabled,
    )


def _agent(
    ollama: Any,
    dispatcher: Any,
    *,
    data_dir: Path = STATIC_TEST_DATA,
    **settings: Any,
) -> AgentService:
    registry = EdgeSchemaRegistry.load_default(data_dir)
    catalog = RuntimeSchemaCatalog.from_data_dir(data_dir, registry)
    return AgentService(
        ollama,
        dispatcher,
        system_prompt=STEWARD.prompt,
        tools=STEWARD.tool_definitions or get_tool_definitions(("calculate",)),
        schema_catalog=catalog,
        localized_identity=STEWARD.settings["localized_identity"],
        assistant_id="steward",
        home_entity_id=settings.pop("home_entity_id", "address:test_house"),
        **settings,
    )


@pytest.fixture
def dispatcher() -> _JsonGraphDispatcher:
    return _JsonGraphDispatcher(DATA_DIR, EdgeSchemaRegistry.load_default(DATA_DIR))


@pytest.fixture
def service(dispatcher: _JsonGraphDispatcher) -> SemanticFactService:
    return _service(dispatcher)


@pytest.fixture
def context() -> AgentRequestContext:
    return _context()


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


async def _execute(
    service: SemanticFactService,
    request: SemanticFactRequest,
    context: AgentRequestContext,
):
    result, queries, _, _ = await service.engine.execute(request, context)
    return result, queries


# --- P1: graph/dispatcher failure -------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing",
    (
        _FailingDispatcher(),
        _FailingDispatcher(error=RuntimeError("database password should not be returned")),
    ),
    ids=("ok_false", "raises"),
)
async def test_graph_dispatcher_failure_is_computation_impossible(
    failing: _FailingDispatcher,
) -> None:
    service = _service(failing, data_dir=STATIC_TEST_DATA)
    answer = await _ask(service, _static_context(), "我是谁")

    assert answer.result.status == "computation_impossible"
    assert failing.calls
    assert not _leaked(answer.text)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing",
    (
        _FailingDispatcher(),
        _FailingDispatcher(error=RuntimeError("database password should not be returned")),
    ),
    ids=("ok_false", "raises"),
)
async def test_graph_dispatcher_failure_does_not_fall_through_to_chat(
    failing: _FailingDispatcher,
) -> None:
    ollama = _ChatOllama({"requires_fact": False, "request": None})
    result = await _agent(ollama, failing).answer(
        "我是谁",
        user_entity={"id": "person:alex_example", "name": ["Alex Example"]},
    )

    assert ollama.calls == []
    assert "casual conversation" not in result.answer
    assert not _leaked(result.answer)


# --- P2: exhausted MALFORMED_OUTPUT ----------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", ("not-a-plan", None, []))
async def test_exhausted_malformed_planner_output_fails_closed(
    dispatcher: _JsonGraphDispatcher,
    payload: Any,
) -> None:
    interpreter = _Interpreter(payload)
    schema = _schema(DATA_DIR)
    with pytest.raises(SemanticPlannerFailure) as captured:
        await SemanticFactPlanner(interpreter, schema).plan(
            [{"role": "user", "content": "今天天气怎么样"}],
            _context(),
        )

    assert captured.value.diagnostics.validation_result == "MALFORMED_OUTPUT"
    assert captured.value.diagnostics.attempt_count == 2
    assert interpreter.calls == 2
    assert dispatcher.calls == []


@pytest.mark.asyncio
async def test_malformed_planner_output_is_unsupported_and_skips_graph_and_chat(
    dispatcher: _JsonGraphDispatcher,
) -> None:
    service = _service(dispatcher, "not-a-plan")
    answer = await _ask(service, _context(), "今天天气怎么样")

    assert answer.result.status == "semantic_plan_unsupported"
    assert answer.planner_diagnostics is not None
    assert answer.planner_diagnostics.validation_result == "MALFORMED_OUTPUT"
    assert dispatcher.calls == []

    ollama = _ChatOllama("not-a-plan")
    failing_unused = _FailingDispatcher()
    result = await _agent(ollama, failing_unused, disable_tier0=True).answer(
        "今天天气怎么样",
        user_entity={"id": "person:alex_example", "name": ["Alex Example"]},
    )
    assert ollama.calls == []
    assert failing_unused.calls == []
    assert "casual conversation" not in result.answer
    assert not _leaked(result.answer)


# --- P3: INVALID_PLAN ------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_classifies_invalid_plan_without_retry_or_graph(
    dispatcher: _JsonGraphDispatcher,
) -> None:
    request = SemanticFactRequest(
        operation="argmin",
        subject=_members(),
        property="display_name",
    )
    interpreter = _Interpreter(request)
    schema = _schema(DATA_DIR)

    with pytest.raises(SemanticPlannerFailure) as captured:
        await SemanticFactPlanner(interpreter, schema).plan(
            [{"role": "user", "content": "家里谁名字排第一"}],
            _context(),
        )

    assert captured.value.diagnostics.validation_result == "INVALID_PLAN"
    assert captured.value.diagnostics.attempt_count == 1
    assert interpreter.calls == 1
    assert dispatcher.calls == []

    service = _service(dispatcher, request)
    answer = await _ask(service, _context(), "家里谁名字排第一")
    assert answer.result.status == "semantic_plan_unsupported"
    assert answer.planner_diagnostics is not None
    assert answer.planner_diagnostics.validation_result == "INVALID_PLAN"
    assert dispatcher.calls == []


# --- P4: NOT_A_FACT --------------------------------------------------------------


@pytest.mark.asyncio
async def test_not_a_fact_is_the_conversation_fallback(
    dispatcher: _JsonGraphDispatcher,
) -> None:
    payload = {"requires_fact": False, "request": None}
    schema = _schema(DATA_DIR)
    interpreter = _Interpreter(payload)
    outcome = await SemanticFactPlanner(interpreter, schema).plan(
        [{"role": "user", "content": "今天天气怎么样"}],
        _context(),
    )
    service = _service(dispatcher, payload)

    answer = await service.try_answer(
        [{"role": "user", "content": "今天天气怎么样"}],
        context=_context(),
    )

    assert outcome.diagnostics.validation_result == "NOT_A_FACT"
    assert answer is None
    assert dispatcher.calls == []

    ollama = _ChatOllama(payload)
    result = await _agent(ollama, dispatcher, data_dir=DATA_DIR).answer(
        "今天天气怎么样",
        user_entity={"id": "person:jian_kuang", "name": ["Jian"]},
    )
    assert len(ollama.calls) == 1
    assert result.answer == "casual conversation"


# --- P5: speaker-relative self ---------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("tier_zero_enabled", (True, False))
@pytest.mark.parametrize(
    "speaker_id",
    ("person:alex_example", "person:blair_example"),
)
async def test_speaker_relative_self_uses_canonical_ids(
    tier_zero_enabled: bool,
    speaker_id: str,
) -> None:
    dispatcher = _JsonGraphDispatcher(
        STATIC_TEST_DATA, EdgeSchemaRegistry.load_default(STATIC_TEST_DATA)
    )
    service = _service(
        dispatcher,
        _resolve(_self()),
        data_dir=STATIC_TEST_DATA,
        tier_zero_enabled=tier_zero_enabled,
    )

    answer = await _ask(service, _static_context(speaker_id), "我是谁")

    assert answer.result.status == "found"
    assert answer.result.evidence.entity_ids == (speaker_id,)
    assert answer.timings.tier == (0 if tier_zero_enabled else 1)


@pytest.mark.asyncio
async def test_missing_caller_context_is_reported_by_the_engine() -> None:
    dispatcher = _JsonGraphDispatcher(
        STATIC_TEST_DATA, EdgeSchemaRegistry.load_default(STATIC_TEST_DATA)
    )
    service = _service(dispatcher, data_dir=STATIC_TEST_DATA)
    result, _ = await _execute(
        service,
        _resolve(_self()),
        _static_context(None),
    )

    assert result.status == "caller_context_missing"


# --- P6: RetrievalService integration --------------------------------------------


@pytest.mark.asyncio
async def test_semantic_facts_use_retrieval_service_for_alias_and_kinship() -> None:
    database = _MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, STATIC_TEST_DATA)  # type: ignore[arg-type]
        registry = EdgeSchemaRegistry.load_default(STATIC_TEST_DATA)
        retrieval = RetrievalService(
            database,  # type: ignore[arg-type]
            limit=25,
            data_dir=STATIC_TEST_DATA,
            edge_registry=registry,
        )
        engine = HouseholdFactEngine(
            ToolDispatcher(retrieval),
            SemanticSchemaRegistry(
                RuntimeSchemaCatalog.from_data_dir(STATIC_TEST_DATA, registry)
            ),
        )
        context = _static_context()
        alias, _, _, _ = await engine.execute(_resolve(_named("艾力克斯")), context)
        child, _, _, _ = await engine.execute(
            _resolve(_named("艾力克斯", SemanticRelationStep(relation="child"))),
            context,
        )
    finally:
        await database.close()

    assert alias.status == "found"
    assert alias.evidence.entity_ids == ("person:alex_example",)
    assert child.status == "found"
    assert child.evidence.entity_ids == ("person:casey_example",)


# --- P7: ended relationships -----------------------------------------------------


@pytest.mark.asyncio
async def test_ended_spouse_is_invisible_to_current_fact_queries() -> None:
    dispatcher = _JsonGraphDispatcher(
        STATIC_TEST_DATA, EdgeSchemaRegistry.load_default(STATIC_TEST_DATA)
    )
    dispatcher.edges["spouse_of"][0]["end"] = "2020-01-01"
    service = _service(dispatcher, data_dir=STATIC_TEST_DATA)
    result, _ = await _execute(service, _resolve(_spouse()), _static_context())

    assert result.status == "relationship_not_found"
    assert any(
        call[0] == "get_relationships" and call[1].get("include_ended") is False
        for call in dispatcher.calls
    )


# --- P8: planner transport failure -----------------------------------------------


@pytest.mark.asyncio
async def test_planner_transport_failure_is_unsupported_without_graph_or_chat(
    dispatcher: _JsonGraphDispatcher,
) -> None:
    service = _service(
        dispatcher,
        error=RuntimeError("boom"),
    )
    answer = await _ask(service, _context(), "今天天气怎么样")

    assert answer.result.status == "semantic_plan_unsupported"
    assert dispatcher.calls == []
    assert not _leaked(answer.text)

    ollama = _ChatOllama(None, error=RuntimeError("boom"))
    unused = _FailingDispatcher()
    result = await _agent(ollama, unused, disable_tier0=True).answer(
        "今天天气怎么样",
        user_entity={"id": "person:alex_example", "name": ["Alex Example"]},
    )
    assert ollama.calls == []
    assert unused.calls == []
    assert "casual conversation" not in result.answer
    assert not _leaked(result.answer)


# --- Executor invariants on the household graph ----------------------------------


@pytest.mark.asyncio
async def test_household_list_is_current_and_uses_one_graph_query(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    result, queries = await _execute(
        service,
        SemanticFactRequest(operation="select", subject=_members()),
        context,
    )
    member_ids = {item["id"] for item in result.value}

    assert result.status == "found"
    assert member_ids >= {
        "person:jian_kuang",
        "person:pu_ba",
        "person:dylan_kuang",
        "person:evelyn_kuang",
        "person:zhigang_ba",
    }
    assert "person:yumei_zhang" not in member_ids
    assert queries == 1
    assert [name for name, _ in dispatcher.calls] == ["get_relationships"]


@pytest.mark.asyncio
async def test_all_static_and_relational_references_converge_on_canonical_person(
    service: SemanticFactService,
    context: AgentRequestContext,
) -> None:
    requests = (
        _resolve(_named("匡德伦")),
        _resolve(_named("德伦")),
        _resolve(_named("Dylan")),
        _resolve(_named("Dylan Kuang")),
        _resolve(_self(_son())),
        _resolve(_named("巴璞", _son())),
    )
    results = [await _execute(service, request, context) for request in requests]

    assert all(result.status == "found" for result, _ in results)
    assert all(result.evidence.entity_ids == ("person:dylan_kuang",) for result, _ in results)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("speaker_id", "path"),
    (
        ("person:jian_kuang", (_son(),)),
        ("person:pu_ba", (_son(),)),
        (
            "person:guiqiu_wang",
            (SemanticRelationStep(relation="child"), _son()),
        ),
        ("person:zhigang_ba", (_daughter(), _son())),
        (
            "person:evelyn_kuang",
            (
                SemanticRelationStep(relation="parent"),
                SemanticRelationStep(
                    relation="child",
                    filters=(
                        SemanticFilter(property="gender", value="male"),
                        SemanticFilter(
                            property="birth_date",
                            operator="lt",
                            value_from="anchor",
                        ),
                    ),
                ),
            ),
        ),
    ),
)
async def test_speaker_relative_kinship_converges_on_dylan(
    service: SemanticFactService,
    context: AgentRequestContext,
    speaker_id: str,
    path: tuple[SemanticRelationStep, ...],
) -> None:
    result, _ = await _execute(
        service,
        _resolve(_self(*path)),
        replace(context, caller_entity_id=speaker_id),
    )

    assert result.status == "found"
    assert result.evidence.entity_ids == ("person:dylan_kuang",)


@pytest.mark.asyncio
async def test_same_self_relation_resolves_from_each_active_speaker(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
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
    request = _resolve(_self(_son()))

    jian, _ = await _execute(service, request, context)
    other, _ = await _execute(
        service,
        request,
        replace(context, caller_entity_id="person:other_parent"),
    )

    assert jian.evidence.entity_ids == ("person:dylan_kuang",)
    assert other.evidence.entity_ids == ("person:other_son",)


@pytest.mark.asyncio
async def test_multi_match_grandson_is_ambiguous(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
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
    result, _ = await _execute(
        service,
        _resolve(_self(SemanticRelationStep(relation="child"), _son())),
        replace(context, caller_entity_id="person:guiqiu_wang"),
    )

    assert result.status == "ambiguous"
    assert _candidate_ids(result) == {
        "person:dylan_kuang",
        "person:second_grandson",
    }


@pytest.mark.asyncio
async def test_scoped_appellation_is_grounded_by_resolver_context(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    dispatcher.entities["person:dylan_kuang"]["appellations"] = [
        {
            "value": "大宝",
            "household_id": "address:fort_cerritos",
            "speaker_ids": ["person:jian_kuang"],
        }
    ]
    request = _resolve(_named("大宝"))

    resolved, _ = await _execute(service, request, context)
    unscoped, _ = await _execute(
        service,
        request,
        replace(context, caller_entity_id="person:pu_ba"),
    )

    assert resolved.evidence.entity_ids == ("person:dylan_kuang",)
    assert unscoped.status == "entity_not_found"


@pytest.mark.asyncio
async def test_empty_household_list_has_a_clear_response(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    dispatcher.edges["lives_in"] = []
    request = SemanticFactRequest(operation="select", subject=_members())
    result, _ = await _execute(service, request, context)

    assert result.status == "found"
    assert result.value == []
    assert service.renderer.render(request, result, context) == (
        "家庭资料中目前没有记录当前家庭成员。"
    )


@pytest.mark.asyncio
async def test_birth_date_resolves_when_storage_uses_birthday(
    dispatcher: _JsonGraphDispatcher,
    context: AgentRequestContext,
) -> None:
    person = dispatcher.entities["person:jian_kuang"]
    person["birthday"] = person.pop("dob")
    catalog = RuntimeSchemaCatalog.from_data_dir(DATA_DIR, dispatcher.registry)
    person_schema = catalog.entities["person"]
    replacement = type(person_schema)(
        "person",
        tuple(
            field if field != "dob" else "birthday" for field in person_schema.properties
        ),
    )
    schema = SemanticSchemaRegistry(
        RuntimeSchemaCatalog(
            {**catalog.entities, "person": replacement},
            catalog.relations,
            catalog.edge_registry,
        )
    )
    service = _service(dispatcher, schema=schema)
    result, _ = await _execute(service, _select(_self(), "birth_date"), context)

    assert result.status == "found"
    assert result.value == "1988-11-11"
    assert result.evidence.entity_ids == ("person:jian_kuang",)


@pytest.mark.asyncio
async def test_missing_birth_date_is_semantic_and_does_not_hallucinate(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    dispatcher.entities["person:jian_kuang"].pop("dob")
    result, _ = await _execute(service, _select(_self(), "birth_date"), context)

    assert result.status == "property_unavailable"
    assert result.missing_requirements == ("birth_date",)
    rendered = service.renderer.render(_select(_self(), "birth_date"), result, context)
    assert "出生日期" in rendered
    assert "dob" not in rendered
    assert "1988" not in rendered


@pytest.mark.asyncio
async def test_missing_birth_date_is_computation_input_missing_for_transform(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    dispatcher.entities["person:dylan_kuang"].pop("dob")
    request = SemanticFactRequest(
        operation="annual_occurrence",
        subject=_self(_son()),
        property="birth_date",
        mode="days",
    )
    result, _ = await _execute(service, request, context)

    assert result.status == "computation_input_missing"
    assert result.missing_requirements == ("birth_date",)


@pytest.mark.asyncio
async def test_named_person_missing_birth_date_is_property_unavailable_not_computation(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    dispatcher.entities["person:dylan_kuang"].pop("dob")
    result, _ = await _execute(service, _select(_named("匡德伦"), "birth_date"), context)

    assert result.status == "property_unavailable"
    assert result.missing_requirements == ("birth_date",)
    assert result.evidence.entity_ids == ("person:dylan_kuang",)


@pytest.mark.asyncio
async def test_invalid_birth_date_is_computation_impossible(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    dispatcher.entities["person:dylan_kuang"]["dob"] = "not-a-date"
    request = SemanticFactRequest(
        operation="annual_occurrence",
        subject=_self(_son()),
        property="birth_date",
        mode="days",
    )
    result, _ = await _execute(service, request, context)

    assert result.status == "computation_impossible"


@pytest.mark.asyncio
async def test_missing_named_person_is_entity_not_found(
    service: SemanticFactService,
    context: AgentRequestContext,
) -> None:
    result, _ = await _execute(
        service, _select(_named("不存在的人"), "birth_date"), context
    )

    assert result.status == "entity_not_found"


@pytest.mark.asyncio
async def test_duplicate_exact_name_is_ambiguous(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    dispatcher.entities["person:other_dylan"] = {
        "id": "person:other_dylan",
        "name": ["Other Dylan", "匡德伦"],
        "gender": "male",
        "dob": "2001-01-01",
    }
    result, _ = await _execute(service, _select(_named("匡德伦"), "birth_date"), context)

    assert result.status == "ambiguous"
    assert _candidate_ids(result) == {"person:dylan_kuang", "person:other_dylan"}


@pytest.mark.asyncio
async def test_absent_relationship_is_not_reported_as_missing_entity(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    dispatcher.edges["spouse_of"] = []
    result, _ = await _execute(
        service,
        _resolve(
            _self(
                SemanticRelationStep(
                    relation="spouse",
                    filters=(SemanticFilter(property="gender", value="female"),),
                )
            )
        ),
        context,
    )

    assert result.status == "relationship_not_found"


@pytest.mark.asyncio
async def test_absent_in_law_path_is_relationship_not_found(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    dispatcher.edges["spouse_of"] = []
    result, _ = await _execute(
        service,
        _resolve(_self(SemanticRelationStep(relation="spouse"), _parent("male"))),
        context,
    )

    assert result.status == "relationship_not_found"
    assert result.evidence.relationship == "spouse"


@pytest.mark.asyncio
async def test_named_entity_with_absent_spouse_preserves_relationship_status(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    dispatcher.edges["spouse_of"] = []
    result, _ = await _execute(
        service,
        _resolve(
            _named(
                "匡健",
                SemanticRelationStep(
                    relation="spouse",
                    filters=(SemanticFilter(property="gender", value="female"),),
                ),
            )
        ),
        context,
    )

    assert result.status == "relationship_not_found"
    assert result.evidence.relationship == "spouse"


@pytest.mark.asyncio
async def test_multiple_matching_children_are_ambiguous(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
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
    result, _ = await _execute(service, _resolve(_self(_son())), context)

    assert result.status == "ambiguous"
    assert _candidate_ids(result) == {"person:dylan_kuang", "person:second_son"}


@pytest.mark.asyncio
async def test_assistant_identity_never_queries_household_graph(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    answer = await _ask(service, context, "你是谁")

    assert answer.result.status == "found"
    assert answer.result.evidence.entity_ids == ("steward",)
    assert dispatcher.calls == []
    assert answer.timings.db_query_count == 0
    assert answer.timings.tier == 0


@pytest.mark.asyncio
async def test_relationship_evidence_preserves_temporal_metadata(
    service: SemanticFactService,
    context: AgentRequestContext,
) -> None:
    result, _ = await _execute(
        service,
        _resolve(
            _self(
                SemanticRelationStep(
                    relation="spouse",
                    filters=(SemanticFilter(property="gender", value="female"),),
                )
            )
        ),
        context,
    )

    assert result.evidence.relationships
    relationship = result.evidence.relationships[0]
    assert relationship.relation == "spouse"
    assert relationship.start == "2014-05-04"
    assert relationship.end is None
    assert result.evidence.entity_ids == ("person:pu_ba",)


def test_capabilities_are_semantic_and_allowlisted(
    dispatcher: _JsonGraphDispatcher,
) -> None:
    schema = _schema(DATA_DIR)
    catalog = RuntimeSchemaCatalog.from_data_dir(DATA_DIR, dispatcher.registry)
    person = catalog.entities["person"]
    augmented = type(person)(
        "person",
        (*person.properties, "favorite_color"),
        {**person.property_types, "favorite_color": "string"},
    )
    schema = SemanticSchemaRegistry(
        RuntimeSchemaCatalog(
            {**catalog.entities, "person": augmented},
            catalog.relations,
            catalog.edge_registry,
        )
    )
    capabilities = schema.capability_payload()

    assert schema.physical_property("person", "favorite_color") == "favorite_color"
    assert "favorite_color" in capabilities["semantic_properties"]["person"]
    assert "birth_date" in capabilities["semantic_properties"]["person"]
    assert "dob" not in capabilities["semantic_properties"]["person"]
    assert "parent_of" not in json.dumps(capabilities)
    assert schema.physical_property("person", "dob") is None
    assert schema.physical_relation("parent_of") is None
    assert "filter" in capabilities["operations"]
    assert capabilities["collection_predicates"] == ["adult", "minor"]
    assert capabilities["property_sources"] == ["entity", "relationship"]
    assert capabilities["semantic_relation_properties"]["spouse"] == [
        "end_date",
        "start_date",
    ]
    assert set(capabilities["operations"]).issubset(set(OPERATORS) | {"filter", "traverse"})
    invalid = SemanticFactRequest(
        operation="count",
        subject=_members(),
        filters=(SemanticFilter(predicate="invented_status"),),
    )
    assert schema.validates(invalid) is False


@pytest.mark.asyncio
async def test_tier_one_can_select_a_new_schema_field_without_a_fact_handler(
    dispatcher: _JsonGraphDispatcher,
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
        RuntimeSchemaCatalog(
            {**catalog.entities, "person": augmented},
            catalog.relations,
            catalog.edge_registry,
        )
    )
    request = _select(_self(), "favorite_color")
    service = _service(dispatcher, request, schema=schema)
    answer = await _ask(service, context, "我的偏爱颜色是什么？")

    assert answer.result.status == "found"
    assert answer.result.value == "green"
    assert answer.result.evidence.entity_ids == ("person:jian_kuang",)
    assert answer.timings.llm_call_count == 1
    assert answer.timings.tier == 1


def test_semantic_ir_rejects_non_allowlisted_operation() -> None:
    with pytest.raises(ValidationError):
        SemanticFactRequest.model_validate(
            {
                "operation": "GET_AGE",
                "subject": {"kind": "self", "entity_type": "person"},
            }
        )


def test_completed_years_does_not_inject_an_age_specific_property() -> None:
    request = SemanticFactRequest(
        operation="completed_years",
        subject=_self(),
    )

    assert request.property is None
    assert _schema(DATA_DIR).validates(request) is False


def test_semantic_validator_rejects_type_incompatible_extreme() -> None:
    schema = _schema(DATA_DIR)
    request = SemanticFactRequest(
        operation="argmin",
        subject=_members(),
        property="display_name",
    )

    assert schema.validates(request) is False
    assert schema.validation_code(request) == "INVALID_PLAN"
    assert schema.validates(
        SemanticFactRequest(
            operation="annual_occurrence",
            subject=_self(),
            property="display_name",
        )
    ) is False


def test_semantic_validator_rejects_context_reference_type_spoofing() -> None:
    schema = _schema(DATA_DIR)

    assert schema.validates(
        SemanticFactRequest(
            operation="resolve_reference",
            subject=SemanticReference(kind="self", entity_type="address"),
        )
    ) is False
    assert schema.validates(
        SemanticFactRequest(
            operation="count",
            subject=_self(),
        )
    ) is False
    assert schema.validates(
        SemanticFactRequest(
            operation="select",
            subject=_self(),
            property="dob",
        )
    ) is False
    assert schema.validates(
        _resolve(_self(SemanticRelationStep(relation="parent_of")))
    ) is False


@pytest.mark.asyncio
async def test_empty_child_collection_counts_as_zero_not_missing_person(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    dispatcher.edges["parent_of"] = [
        edge
        for edge in dispatcher.edges["parent_of"]
        if edge["from"] != "person:jian_kuang"
    ]
    result, _ = await _execute(
        service,
        SemanticFactRequest(
            operation="count",
            subject=_self(SemanticRelationStep(relation="child")),
        ),
        context,
    )

    assert result.status == "found"
    assert result.value == 0


@pytest.mark.asyncio
async def test_registered_date_range_predicate_filters_relationships(
    service: SemanticFactService,
    context: AgentRequestContext,
) -> None:
    result, _ = await _execute(
        service,
        SemanticFactRequest(
            operation="count",
            subject=SemanticReference(
                kind="current_household",
                entity_type="address",
                path=(
                    SemanticRelationStep(
                        relation="member",
                        filters=(
                            SemanticFilter(
                                property="start_date",
                                operator="date_range",
                                value=("2026-06-01", "2026-07-01"),
                                source="relation",
                            ),
                        ),
                    ),
                ),
            ),
        ),
        context,
    )

    assert result.status == "found"
    assert result.value == 3


@pytest.mark.asyncio
async def test_argmin_and_argmax_share_the_household_extrema_path(
    service: SemanticFactService,
    context: AgentRequestContext,
) -> None:
    oldest = SemanticFactRequest(
        operation="argmin",
        subject=_members(),
        property="birth_date",
    )
    youngest = oldest.model_copy(update={"operation": "argmax"})
    oldest_result, _ = await _execute(service, oldest, context)
    youngest_result, _ = await _execute(service, youngest, context)

    assert oldest_result.status == youngest_result.status == "found"
    assert oldest_result.value["id"] == "person:zhigang_ba"
    assert youngest_result.value["id"] == "person:evelyn_kuang"
    assert TierZeroSemanticParser().parse("谁最年幼") is None


@pytest.mark.asyncio
async def test_youngest_reports_partial_birth_date_evidence(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    dispatcher.entities["person:evelyn_kuang"].pop("dob")
    result, _ = await _execute(
        service,
        SemanticFactRequest(
            operation="argmax",
            subject=_members(),
            property="birth_date",
        ),
        context,
    )

    assert result.status == "computation_input_missing"
    assert result.missing_requirements == ("birth_date",)


@pytest.mark.asyncio
async def test_count_composes_with_generic_property_and_status_filters(
    service: SemanticFactService,
    context: AgentRequestContext,
) -> None:
    adult = SemanticFactRequest(
        operation="count",
        subject=_members(),
        filters=(SemanticFilter(predicate="adult"),),
    )
    minor = adult.model_copy(update={"filters": (SemanticFilter(predicate="minor"),)})
    female = adult.model_copy(
        update={"filters": (SemanticFilter(property="gender", value="female"),)}
    )
    adult_result, _ = await _execute(service, adult, context)
    minor_result, _ = await _execute(service, minor, context)
    female_result, _ = await _execute(service, female, context)

    assert adult_result.status == minor_result.status == female_result.status == "found"
    assert adult_result.value == 3
    assert minor_result.value == 2
    assert female_result.value == 2


@pytest.mark.asyncio
async def test_minor_status_is_distinct_from_a_persons_child_relation(
    service: SemanticFactService,
    context: AgentRequestContext,
) -> None:
    minors, _ = await _execute(
        service,
        SemanticFactRequest(
            operation="count",
            subject=_members(),
            filters=(SemanticFilter(predicate="minor"),),
        ),
        context,
    )
    children, _ = await _execute(
        service,
        SemanticFactRequest(
            operation="count",
            subject=_self(SemanticRelationStep(relation="child")),
        ),
        context,
    )

    assert minors.value == 2
    assert children.value == 2


@pytest.mark.asyncio
async def test_status_predicate_prefers_authoritative_role_then_falls_back_to_age(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    request = SemanticFactRequest(
        operation="count",
        subject=_members(),
        filters=(SemanticFilter(predicate="adult"),),
    )
    dylan_edge = next(
        edge
        for edge in dispatcher.edges["lives_in"]
        if edge["from"] == "person:dylan_kuang"
    )
    dylan_edge["household_role"] = "owner"
    role_result, _ = await _execute(service, request, context)
    assert role_result.value == 4

    dylan_edge.pop("household_role")
    fallback_result, _ = await _execute(service, request, context)
    assert fallback_result.value == 3

    dispatcher.entities["person:dylan_kuang"].pop("dob")
    missing_result, _ = await _execute(service, request, context)
    assert missing_result.status == "filter_input_missing"
    assert missing_result.missing_requirements[:1] == ("adult",)


@pytest.mark.asyncio
async def test_relationship_properties_and_duration_use_the_spouse_edge(
    service: SemanticFactService,
    context: AgentRequestContext,
) -> None:
    marriage_date = _select(_spouse(), "start_date", property_source="relationship")
    duration = marriage_date.model_copy(update={"operation": "duration", "mode": "days"})
    date_result, _ = await _execute(service, marriage_date, context)
    duration_result, _ = await _execute(service, duration, context)

    assert date_result.status == "found"
    assert date_result.value == "2014-05-04"
    assert date_result.evidence.entity_ids == ("person:pu_ba",)
    assert duration_result.status == "found"
    assert duration_result.value == 4504


@pytest.mark.asyncio
async def test_missing_relationship_start_has_specific_failure(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    dispatcher.edges["spouse_of"][0].pop("start")
    request = _select(_spouse(), "start_date", property_source="relationship")
    result, _ = await _execute(service, request, context)

    assert result.status == "relation_property_unavailable"
    assert result.evidence.relationship == "spouse"
    assert result.missing_requirements == ("start_date",)


@pytest.mark.asyncio
async def test_relationship_property_resolution_is_speaker_relative(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    dispatcher.entities.update(
        {
            "person:alex_fixture": {"id": "person:alex_fixture", "name": ["Alex"]},
            "person:sam_fixture": {"id": "person:sam_fixture", "name": ["Sam"]},
        }
    )
    dispatcher.edges["spouse_of"].append(
        {
            "from": "person:alex_fixture",
            "to": "person:sam_fixture",
            "start": "2020-02-20",
            "end": None,
        }
    )
    result, _ = await _execute(
        service,
        _select(_spouse(), "start_date", property_source="relationship"),
        replace(context, caller_entity_id="person:alex_fixture"),
    )

    assert result.status == "found"
    assert result.value == "2020-02-20"
    assert result.evidence.entity_ids == ("person:sam_fixture",)


@pytest.mark.asyncio
async def test_planner_retries_once_for_structural_failure(
    context: AgentRequestContext,
) -> None:
    schema = _schema(DATA_DIR)
    interpreter = _Interpreter(
        lambda calls, _messages: (
            {"requires_fact": True, "request": {"operation": "select"}}
            if calls == 1
            else {
                "requires_fact": True,
                "request": {
                    "operation": "resolve_reference",
                    "subject": {"kind": "self", "entity_type": "person"},
                },
            }
        )
    )
    outcome = await SemanticFactPlanner(interpreter, schema).plan(
        [{"role": "user", "content": "我是谁"}], context
    )

    assert outcome.plan.request is not None
    assert outcome.diagnostics.validation_result == "VALID"
    assert outcome.diagnostics.attempt_count == 2
    assert interpreter.calls == 2
    assert "strict structural validation" in interpreter.messages[1][-1]["content"]


@pytest.mark.asyncio
async def test_planner_classifies_unsupported_operation_after_one_retry(
    context: AgentRequestContext,
) -> None:
    interpreter = _Interpreter(
        {
            "requires_fact": True,
            "request": {
                "operation": "invented_operation",
                "subject": {"kind": "self", "entity_type": "person"},
            },
        }
    )
    with pytest.raises(SemanticPlannerFailure) as captured:
        await SemanticFactPlanner(interpreter, _schema(DATA_DIR)).plan([], context)

    assert captured.value.diagnostics.validation_result == "UNSUPPORTED_OPERATION"
    assert captured.value.diagnostics.attempt_count == 2
    assert interpreter.calls == 2


@pytest.mark.asyncio
async def test_semantic_validation_failure_is_classified_without_retry(
    dispatcher: _JsonGraphDispatcher,
    context: AgentRequestContext,
) -> None:
    service = _service(
        dispatcher,
        {
            "requires_fact": True,
            "request": {
                "operation": "select",
                "subject": {"kind": "self", "entity_type": "person"},
                "property": "raw_private_field",
            },
        },
        tier_zero_enabled=False,
    )
    answer = await _ask(service, context, "读取一个不存在的字段")

    assert answer.result.status == "semantic_plan_unsupported"
    assert answer.planner_diagnostics is not None
    assert answer.planner_diagnostics.validation_result == "UNKNOWN_PROPERTY"
    assert answer.planner_diagnostics.attempt_count == 1
    assert dispatcher.calls == []


def test_semantic_validation_classifies_unknown_relation() -> None:
    request = _resolve(_self(SemanticRelationStep(relation="invented_relation")))

    assert _schema(DATA_DIR).validation_code(request) == "UNKNOWN_RELATION"


@pytest.mark.asyncio
async def test_planner_schema_excludes_entity_ids_and_normalizes_status_filter_shape(
    context: AgentRequestContext,
) -> None:
    schema = _schema(DATA_DIR)
    interpreter = _Interpreter(
        {
            "requires_fact": True,
            "request": {
                "operation": "count",
                "subject": {
                    "kind": "current_household",
                    "entity_type": "person",
                    "path": [
                        {
                            "relation": "child",
                            "filters": [{"property": "adult"}],
                        }
                    ],
                },
            },
        }
    )
    outcome = await SemanticFactPlanner(interpreter, schema).plan([], context)
    plan = outcome.plan

    assert interpreter.output_schema is not None
    kinds = interpreter.output_schema["$defs"]["SemanticReference"]["properties"][
        "kind"
    ]["enum"]
    assert "entity_id" not in kinds
    assert plan.request is not None
    assert plan.request.subject.entity_type == "address"
    assert plan.request.subject.path[-1].relation == "member"
    assert plan.request.filters == (SemanticFilter(predicate="adult"),)
    assert schema.validates(plan.request) is True


@pytest.mark.asyncio
async def test_new_numeric_field_immediately_supports_generic_argmax(
    dispatcher: _JsonGraphDispatcher,
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
        RuntimeSchemaCatalog(
            {**catalog.entities, "person": augmented},
            catalog.relations,
            catalog.edge_registry,
        )
    )
    engine = HouseholdFactEngine(dispatcher, schema)
    request = SemanticFactRequest(
        operation="argmax",
        subject=_members(),
        property="fixture_score",
    )
    result, _, _, _ = await engine.execute(request, context)

    assert schema.validates(request) is True
    assert result.status == "found"
    assert result.value["id"] == max(household_ids)


@pytest.mark.asyncio
async def test_tier_one_uses_one_semantic_call_then_deterministic_execution(
    dispatcher: _JsonGraphDispatcher,
    context: AgentRequestContext,
) -> None:
    request = SemanticFactRequest(
        operation="argmin",
        subject=_members(),
        property="birth_date",
    )
    interpreter = _Interpreter(request)
    schema = _schema(DATA_DIR)
    service = SemanticFactService(
        HouseholdFactEngine(dispatcher, schema),
        planner=SemanticFactPlanner(interpreter, schema),
    )
    answer = await _ask(service, context, "家里哪位成员出生最早")

    assert answer.result.status == "found"
    assert answer.result.value["id"] == "person:zhigang_ba"
    assert answer.timings.tier == 1
    assert answer.timings.llm_call_count == 1
    assert interpreter.calls == 1
    assert interpreter.capabilities is not None
    assert "dob" not in json.dumps(interpreter.capabilities)


@pytest.mark.asyncio
async def test_tier_one_rejects_unadvertised_semantic_property(
    dispatcher: _JsonGraphDispatcher,
    context: AgentRequestContext,
) -> None:
    service = _service(
        dispatcher,
        _select(_self(), "invented_private_fact"),
    )
    answer = await _ask(service, context, "我有什么秘密家庭属性")

    assert answer.result.status == "semantic_plan_unsupported"
    assert answer.timings.llm_call_count == 1
    assert dispatcher.calls == []


@pytest.mark.asyncio
async def test_tier_zero_disabled_bypasses_parser_for_core_plans(
    dispatcher: _JsonGraphDispatcher,
    context: AgentRequestContext,
) -> None:
    plans = {
        "我是谁": _resolve(_self()),
        "德伦是谁": _resolve(_named("德伦")),
        "我岳父是谁": _resolve(
            _self(SemanticRelationStep(relation="spouse"), _parent("male"))
        ),
        "家里有几个人": SemanticFactRequest(operation="count", subject=_members()),
    }

    class DisabledParser:
        def parse(self, _text: str) -> None:
            raise AssertionError("Tier 0 must be bypassed")

    service = _service(
        dispatcher,
        lambda _calls, messages: plans[messages[-1]["content"]],
        parser=DisabledParser(),
        tier_zero_enabled=False,
    )
    answers = {
        question: await _ask(service, context, question) for question in plans
    }

    assert answers["我是谁"].result.evidence.entity_ids == ("person:jian_kuang",)
    assert answers["德伦是谁"].result.evidence.entity_ids == ("person:dylan_kuang",)
    assert answers["我岳父是谁"].result.evidence.entity_ids == ("person:zhigang_ba",)
    assert answers["家里有几个人"].result.value == 5
    assert all(answer.timings.tier == 1 for answer in answers.values())
    assert all(answer.timings.llm_call_count == 1 for answer in answers.values())


@pytest.mark.asyncio
async def test_tier_one_open_world_paraphrases_use_resolver_not_entity_ids(
    dispatcher: _JsonGraphDispatcher,
    context: AgentRequestContext,
) -> None:
    outputs = {
        "Dylan是哪位？": _named("Dylan"),
        "德伦是哪一个人？": _named("德伦"),
        "巴璞她儿子是谁？": _named("巴璞", _son()),
        "我妻子的父亲是谁？": _self(
            SemanticRelationStep(
                relation="spouse",
                filters=(SemanticFilter(property="gender", value="female"),),
            ),
            _parent("male"),
        ),
    }
    service = _service(
        dispatcher,
        lambda _calls, messages: _resolve(outputs[messages[-1]["content"]]),
        tier_zero_enabled=False,
    )
    answers = [await _ask(service, context, question) for question in outputs]

    assert all(answer.result.status == "found" for answer in answers)
    assert [answer.result.evidence.entity_ids for answer in answers] == [
        ("person:dylan_kuang",),
        ("person:dylan_kuang",),
        ("person:dylan_kuang",),
        ("person:zhigang_ba",),
    ]
    assert all(answer.timings.tier == 1 for answer in answers)


@pytest.mark.asyncio
async def test_semantic_planner_rejects_model_originated_entity_id(
    context: AgentRequestContext,
) -> None:
    interpreter = _Interpreter(
        {
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
    )
    with pytest.raises(SemanticPlannerFailure) as captured:
        await SemanticFactPlanner(interpreter, _schema(DATA_DIR)).plan(
            [{"role": "user", "content": "Dylan是谁"}],
            context,
        )

    assert captured.value.diagnostics.validation_result == "MODEL_ORIGINATED_ENTITY_ID"
    assert captured.value.diagnostics.attempt_count == 1


# --- Optional high-value cases ---------------------------------------------------


@pytest.mark.asyncio
async def test_equal_age_comparison_reports_equality(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    dispatcher.entities["person:pu_ba"]["dob"] = dispatcher.entities["person:jian_kuang"][
        "dob"
    ]
    result, _ = await _execute(
        service,
        SemanticFactRequest(
            operation="argmin",
            subject=_self(),
            other=_self(
                SemanticRelationStep(
                    relation="spouse",
                    filters=(SemanticFilter(property="gender", value="female"),),
                )
            ),
            property="birth_date",
        ),
        context,
    )

    assert result.status == "found"
    assert result.value["equal"] is True


@pytest.mark.asyncio
async def test_birthday_today_is_zero_days(
    service: SemanticFactService,
    context: AgentRequestContext,
) -> None:
    result, _ = await _execute(
        service,
        SemanticFactRequest(
            operation="annual_occurrence",
            subject=_self(_son()),
            property="birth_date",
            mode="days",
        ),
        replace(
            context,
            current_time=datetime.fromisoformat("2026-10-30T00:01:00-07:00"),
        ),
    )

    assert result.status == "found"
    assert result.value == 0
    assert result.evidence.entity_ids == ("person:dylan_kuang",)


@pytest.mark.asyncio
async def test_english_success_and_missing_property_use_english_copy(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    english = replace(context, locale="en")
    found, _ = await _execute(service, _resolve(_self()), english)
    dispatcher.entities["person:jian_kuang"].pop("dob")
    missing, _ = await _execute(service, _select(_self(), "birth_date"), english)

    assert found.status == "found"
    assert found.evidence.entity_ids == ("person:jian_kuang",)
    assert "resolved person" in service.renderer.render(
        _resolve(_self()), found, english
    ).lower()
    assert missing.status == "property_unavailable"
    rendered = service.renderer.render(_select(_self(), "birth_date"), missing, english)
    assert "unavailable" in rendered.lower()
    assert "出生日期" not in rendered


@pytest.mark.asyncio
async def test_missing_household_id_does_not_fabricate_members(
    service: SemanticFactService,
    context: AgentRequestContext,
) -> None:
    result, _ = await _execute(
        service,
        SemanticFactRequest(operation="count", subject=_members()),
        replace(context, household_id=None),
    )

    assert result.status == "entity_not_found"


@pytest.mark.asyncio
async def test_older_sibling_without_anchor_birth_date_fails_closed(
    service: SemanticFactService,
    context: AgentRequestContext,
    dispatcher: _JsonGraphDispatcher,
) -> None:
    dispatcher.entities["person:evelyn_kuang"].pop("dob")
    result, _ = await _execute(
        service,
        _resolve(
            _self(
                SemanticRelationStep(relation="parent"),
                SemanticRelationStep(
                    relation="child",
                    filters=(
                        SemanticFilter(property="gender", value="male"),
                        SemanticFilter(
                            property="birth_date",
                            operator="lt",
                            value_from="anchor",
                        ),
                    ),
                ),
            )
        ),
        replace(context, caller_entity_id="person:evelyn_kuang"),
    )

    assert result.status in {"property_unavailable", "computation_impossible"}
