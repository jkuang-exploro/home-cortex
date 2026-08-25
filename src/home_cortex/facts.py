"""Structured, deterministic household-fact resolution.

This module contains household semantics rather than sentence-specific answer
handlers. Natural-language input is reduced to a small ``FactRequest``; graph
relationships are then traversed from trusted roots and rendered without asking
the model to invent or restate facts.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from time import perf_counter
from typing import Any, Literal

from .display import (
    INTERNAL_ID_PATTERN,
    resolve_display_name,
    resolve_person_reference,
)
from .memorable_dates import (
    MemorableDateRegistry,
    MemorableDateSchema,
    default_memorable_date_registry,
)

FactField = Literal[
    "address",
    "identity",
    "items",
    "relationship_to_speaker",
    "relationship_exists",
    "count",
    "memorable_date",
    "location",
    "residents",
    "spaces",
]
SubjectKind = Literal["speaker", "home", "item", "named", "relative", "space"]
DateQuery = Literal["stored", "next"]
StopReason = Literal["answer", "tool_error", "timeout"]

logger = logging.getLogger("uvicorn.error.home_cortex.facts")


@dataclass(frozen=True)
class SubjectReference:
    kind: SubjectKind
    value: str | tuple[str, ...] | None = None


@dataclass(frozen=True)
class FactRequest:
    subject: SubjectReference
    field: FactField
    cardinality: Literal["one", "all"] = "one"
    memorable_date: str | None = None
    date_query: DateQuery = "stored"
    relation: str | None = None
    target: SubjectReference | None = None
    space_type: str | None = None


@dataclass(frozen=True)
class RelationshipStep:
    relation: str
    direction: Literal["out", "in", "both"] | None = None
    gender: Literal["female", "male"] | None = None


@dataclass(frozen=True)
class RelativeDefinition:
    steps: tuple[RelationshipStep, ...]


@dataclass(frozen=True)
class FactAnswer:
    text: str
    tool_calls: int
    stop_reason: StopReason = "answer"


@dataclass(frozen=True)
class FactContext:
    """Trusted graph context supplied to an open-ended model response."""

    text: str
    tool_calls: int


@dataclass(frozen=True)
class _Traversal:
    entities: tuple[Mapping[str, Any], ...]
    edges: tuple[Mapping[str, Any], ...]


RELATIVES: Mapping[str, RelativeDefinition] = {
    "spouse": RelativeDefinition((RelationshipStep("spouse_of"),)),
    "wife": RelativeDefinition(
        (RelationshipStep("spouse_of", gender="female"),)
    ),
    "husband": RelativeDefinition(
        (RelationshipStep("spouse_of", gender="male"),)
    ),
    "children": RelativeDefinition(
        (RelationshipStep("parent_of", direction="out"),)
    ),
    "son": RelativeDefinition(
        (RelationshipStep("parent_of", direction="out", gender="male"),)
    ),
    "daughter": RelativeDefinition(
        (RelationshipStep("parent_of", direction="out", gender="female"),)
    ),
    "parents": RelativeDefinition(
        (RelationshipStep("parent_of", direction="in"),)
    ),
    "father": RelativeDefinition(
        (RelationshipStep("parent_of", direction="in", gender="male"),)
    ),
    "mother": RelativeDefinition(
        (RelationshipStep("parent_of", direction="in", gender="female"),)
    ),
    "father_in_law": RelativeDefinition(
        (
            RelationshipStep("spouse_of", gender="female"),
            RelationshipStep("parent_of", direction="in", gender="male"),
        )
    ),
    "mother_in_law": RelativeDefinition(
        (
            RelationshipStep("spouse_of", gender="female"),
            RelationshipStep("parent_of", direction="in", gender="female"),
        )
    ),
}

# These are vocabulary aliases, not memorized questions. The longest terms are
# checked first so a compound relationship such as father-in-law is not reduced
# to the more general father relationship.
_RELATIVE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("father_in_law", ("father-in-law", "father in law", "岳父")),
    ("mother_in_law", ("mother-in-law", "mother in law", "岳母")),
    ("daughter", ("daughter", "女儿")),
    ("son", ("son", "儿子")),
    ("children", ("children", "child", "孩子们", "孩子")),
    ("mother", ("mother", "mom", "mum", "母亲", "妈妈")),
    ("father", ("father", "dad", "父亲", "爸爸")),
    ("parents", ("parents", "父母")),
    ("wife", ("wife", "太太", "妻子", "老婆")),
    ("husband", ("husband", "丈夫", "老公")),
    ("spouse", ("spouse", "配偶")),
)

_NEXT_OCCURRENCE_PATTERNS = (
    r"\b(?:how many days|days)\s+(?:are\s+)?(?:left\s+)?until\b",
    r"\bhow long until\b",
    r"(?:还有|再过)(?:多少|几)天",
    r"(?:还有多久|多久以后)",
)
_LOOKUP_TERMS = (
    "who",
    "which",
    "list",
    "identify",
    "谁",
    "哪位",
    "哪些",
    "列出",
)
_COUNT_TERMS = ("how many", "count", "number", "几个", "几位", "多少")
_RESIDENCE_TERMS = (
    "live in",
    "lives in",
    "living in",
    "live at",
    "lives at",
    "reside in",
    "resides in",
    "住",
    "住在",
    "居住在",
)
_HOME_REFERENCE_TERMS = (
    "my home",
    "our home",
    "my house",
    "our house",
    "my household",
    "our household",
    "我家",
    "我们家",
    "家里",
    "家中",
)
_ADDRESS_TERMS = (
    "address",
    "street address",
    "地址",
    "住址",
    "位置",
    "哪里",
    "哪儿",
    "在哪",
)
_ROOM_TERMS = ("room", "rooms", "房间")
_SPACE_TERMS = _ROOM_TERMS + ("space", "spaces", "区域", "空间")
_SPACE_LOOKUP_TERMS = _COUNT_TERMS + (
    "what",
    "which",
    "list",
    "show",
    "哪些",
    "有什么",
    "列出",
    "显示",
)
_SPACE_NON_LOOKUP_TERMS = (
    "add",
    "decorate",
    "design",
    "recommend",
    "remodel",
    "should",
    "建议",
    "推荐",
    "装修",
    "设计",
    "改造",
    "增加",
)


class _FactFailure(RuntimeError):
    def __init__(self, reason: StopReason) -> None:
        super().__init__(reason)
        self.reason = reason


class _Execution:
    def __init__(
        self,
        dispatcher: Any,
        *,
        request_id: str,
        max_tool_calls: int,
        max_records: int,
        timeout_seconds: float,
    ) -> None:
        self.dispatcher = dispatcher
        self.request_id = request_id
        self.max_tool_calls = max_tool_calls
        self.max_records = max_records
        self.timeout_seconds = timeout_seconds
        self.tool_calls = 0

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.tool_calls >= self.max_tool_calls:
            raise _FactFailure("tool_error")
        self.tool_calls += 1
        started = perf_counter()
        try:
            result = await asyncio.wait_for(
                self.dispatcher.dispatch(tool_name, arguments),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as error:
            self._log(tool_name, arguments, False, 0, started, "tool_timeout")
            raise _FactFailure("timeout") from error
        except Exception as error:
            self._log(tool_name, arguments, False, 0, started, "tool_error")
            raise _FactFailure("tool_error") from error
        records = result.get("result")
        record_count = len(records) if isinstance(records, list) else 0
        error = result.get("error")
        error_code = error.get("code") if isinstance(error, Mapping) else "none"
        self._log(
            tool_name,
            arguments,
            result.get("ok") is True,
            record_count,
            started,
            str(error_code),
        )
        if result.get("ok") is not True:
            reason: StopReason = (
                "timeout" if error_code == "tool_timeout" else "tool_error"
            )
            raise _FactFailure(reason)
        return result

    def _log(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        success: bool,
        record_count: int,
        started: float,
        error_code: str,
    ) -> None:
        logger.info(
            "tool_execution request_id=%s step=0 tool=%s success=%s "
            "relation=%s direction=%s record_count=%d duration_ms=%.2f "
            "error_code=%s planned=true",
            _safe_log_token(self.request_id),
            _safe_log_token(tool_name),
            str(success).lower(),
            _safe_log_token(str(arguments.get("relation") or "none")),
            _safe_log_token(str(arguments.get("direction") or "none")),
            record_count,
            (perf_counter() - started) * 1_000,
            _safe_log_token(error_code),
        )


class FactService:
    """Resolve supported household facts through one structured pipeline."""

    def __init__(
        self,
        dispatcher: Any,
        *,
        home_entity_id: str | None,
        memorable_dates: MemorableDateRegistry | None = None,
        max_tool_calls: int = 4,
        max_records: int = 25,
        timeout_seconds: float = 5.0,
    ) -> None:
        if home_entity_id is not None and not re.fullmatch(
            r"address:[A-Za-z0-9_-]+",
            home_entity_id,
        ):
            raise ValueError("home_entity_id must be an address record ID")
        self.dispatcher = dispatcher
        self.home_entity_id = home_entity_id
        self.memorable_dates = memorable_dates or default_memorable_date_registry()
        self.max_tool_calls = max_tool_calls
        self.max_records = max_records
        self.timeout_seconds = timeout_seconds

    async def try_answer(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        identity: Mapping[str, Any] | None,
        language: str,
        request_id: str,
        current_date: date | None = None,
    ) -> FactAnswer | None:
        # A request may contain a retrievable sub-fact without being answerable
        # by that fact alone (for example, "How many rooms do we have, and is
        # that enough?"). Preserve the whole intent for fact-assisted model
        # reasoning instead of returning only the parsed sub-fact.
        if _is_home_adequacy_request(_latest_user_text(messages).casefold()):
            return None
        request = parse_fact_request(
            messages,
            identity=identity,
            memorable_dates=self.memorable_dates,
        )
        if request is None:
            return None

        execution = _Execution(
            self.dispatcher,
            request_id=request_id,
            max_tool_calls=self.max_tool_calls,
            max_records=self.max_records,
            timeout_seconds=self.timeout_seconds,
        )
        try:
            text = await self._execute(
                request,
                execution,
                identity=identity,
                language=language,
                current_date=current_date or date.today(),
            )
            reason: StopReason = "answer"
        except _FactFailure as error:
            text = _grounding_fallback(language)
            reason = error.reason

        logger.info(
            "agent_stop request_id=%s reason=%s steps=1 tool_calls=%d",
            _safe_log_token(request_id),
            reason,
            execution.tool_calls,
        )
        return FactAnswer(text=text, tool_calls=execution.tool_calls, stop_reason=reason)

    async def try_context(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        language: str,
        request_id: str,
    ) -> FactContext | None:
        """Retrieve facts that inform, but do not determine, a judgment."""
        if not _is_home_adequacy_request(_latest_user_text(messages).casefold()):
            return None
        if self.home_entity_id is None:
            return FactContext(
                text=(
                    "No configured home is available. Treat the user's question "
                    "as an open-ended judgment and ask only for information that "
                    "would materially affect the answer."
                ),
                tool_calls=0,
            )

        execution = _Execution(
            self.dispatcher,
            request_id=request_id,
            max_tool_calls=self.max_tool_calls,
            max_records=self.max_records,
            timeout_seconds=self.timeout_seconds,
        )
        residents: list[Mapping[str, Any]] | None = None
        rooms: list[Mapping[str, Any]] | None = None
        try:
            residents = await self._home_residents(execution)
        except _FactFailure:
            pass
        try:
            rooms = [
                space
                for space in await self._home_spaces(execution)
                if space.get("space_type") == "room"
            ]
        except _FactFailure:
            pass

        lines = [
            "Trusted household facts for the latest evaluative question:",
        ]
        if residents is None:
            lines.append("- The stored resident count could not be retrieved.")
        else:
            lines.append(f"- Stored current resident count: {len(residents)}.")
        if rooms is None:
            lines.append("- The stored room list could not be retrieved.")
        elif rooms:
            room_names = ", ".join(
                resolve_display_name(room, language) for room in rooms
            )
            lines.extend(
                (
                    f"- Stored room count: {len(rooms)}.",
                    f"- Stored rooms: {room_names}.",
                )
            )
        else:
            lines.append("- No rooms are recorded in the household graph.")
        lines.append(
            "Use these as factual inputs, not as a predetermined conclusion. "
            "If the user supplies a hypothetical or updated resident count, "
            "reason from that scenario rather than replacing it with the stored "
            "current count. "
            "Room count alone does not establish sleeping capacity, floor area, "
            "comfort, crowding, or legal occupancy. Give a practical, qualified "
            "answer and ask for missing preferences only when useful. Do not "
            "introduce legal, household-registration, or ownership issues unless "
            "the user asks about them."
        )
        return FactContext(text="\n".join(lines), tool_calls=execution.tool_calls)

    async def _execute(
        self,
        request: FactRequest,
        execution: _Execution,
        *,
        identity: Mapping[str, Any] | None,
        language: str,
        current_date: date,
    ) -> str:
        if request.subject.kind == "speaker":
            if identity is None:
                return _no_records_fallback(language)
            return _format_speaker_identity(identity, language)

        if request.subject.kind == "home":
            if self.home_entity_id is None:
                return _no_records_fallback(language)
            if request.field in {"address", "identity"}:
                home = await self._home_entity(execution)
                if request.field == "identity":
                    return _format_home_identity(home, identity, language)
                return _format_home_address(home, identity, language)
            if (
                request.field in {"spaces", "count"}
                and request.relation == "hosts_space"
            ):
                spaces = await self._home_spaces(execution)
                if request.space_type is not None:
                    spaces = [
                        space
                        for space in spaces
                        if space.get("space_type") == request.space_type
                    ]
                if request.field == "count":
                    return _format_home_space_count(
                        spaces,
                        request.space_type,
                        identity,
                        language,
                    )
                return _format_home_spaces(
                    spaces,
                    request.space_type,
                    identity,
                    language,
                )
            residents = await self._home_residents(execution)
            if request.field == "count" and request.relation == "lives_in":
                return _format_resident_count(residents, identity, language)
            return _format_residents(residents, identity, language)

        if request.subject.kind == "item":
            item_name = request.subject.value
            assert isinstance(item_name, str)
            return await self._answer_item_location(
                item_name,
                execution,
                identity=identity,
                language=language,
            )

        if request.subject.kind == "space":
            space_name = request.subject.value
            assert isinstance(space_name, str)
            return await self._answer_space_inventory(
                space_name,
                request,
                execution,
                identity=identity,
                language=language,
            )

        if request.subject.kind == "named":
            names = request.subject.value
            assert isinstance(names, tuple)
            if request.field == "memorable_date":
                schema = self._request_date_schema(request)
                if schema.source_kind != "node" or schema.source_type != "person":
                    return _no_records_fallback(language)
                return await self._answer_named_memorable_dates(
                    names,
                    execution,
                    schema=schema,
                    date_query=request.date_query,
                    identity=identity,
                    language=language,
                    current_date=current_date,
                )
            return await self._answer_named_relationships(
                names,
                execution,
                identity=identity,
                language=language,
            )

        relative = request.subject.value
        assert isinstance(relative, str)
        if identity is None:
            return _no_records_fallback(language)
        schema = (
            self._request_date_schema(request)
            if request.field == "memorable_date"
            else None
        )
        definition = (
            RelativeDefinition((RelationshipStep(schema.source_type),))
            if schema is not None and schema.source_kind == "edge"
            else RELATIVES[relative]
        )
        traversal = await self._traverse(
            str(identity["id"]),
            definition,
            execution,
        )
        if request.field == "memorable_date":
            assert schema is not None
            if schema.source_kind == "edge":
                return _format_edge_memorable_date(
                    traversal.edges,
                    self.memorable_dates,
                    schema,
                    request.date_query,
                    identity,
                    language,
                    current_date,
                )
            if schema.source_kind != "node" or schema.source_type != "person":
                return _no_records_fallback(language)
            people = await self._load_entities(traversal.entities, execution)
            return _format_relative_memorable_dates(
                people,
                relative,
                self.memorable_dates,
                schema,
                request.date_query,
                identity,
                language,
                current_date,
            )
        if request.field == "relationship_exists":
            if (
                request.relation is None
                or request.target is None
                or request.target.kind != "home"
                or self.home_entity_id is None
            ):
                return _no_records_fallback(language)
            matches = await self._relationship_matches(
                traversal.entities,
                request.relation,
                self.home_entity_id,
                execution,
            )
            return _format_boolean_answer(all(matches), identity, language)
        if request.field == "count":
            return _format_relative_count(
                traversal.entities,
                relative,
                identity,
                language,
            )
        return _format_relatives(
            traversal.entities,
            relative,
            identity,
            language,
        )

    async def _relationship_matches(
        self,
        subjects: Sequence[Mapping[str, Any]],
        relation: str,
        target_id: str,
        execution: _Execution,
    ) -> list[bool]:
        if not subjects:
            raise _FactFailure("tool_error")
        matches: list[bool] = []
        for subject in subjects:
            subject_id = subject.get("id")
            if not isinstance(subject_id, str):
                raise _FactFailure("tool_error")
            result = await execution.call(
                "get_relationships",
                {
                    "entity_id": subject_id,
                    "relation": relation,
                    "limit": execution.max_records,
                },
            )
            records = result.get("result")
            if not isinstance(records, list):
                raise _FactFailure("tool_error")
            matches.append(
                any(
                    isinstance(record, Mapping)
                    and isinstance(record.get("related_entity"), Mapping)
                    and record["related_entity"].get("id") == target_id
                    for record in records
                )
            )
        return matches

    async def _traverse(
        self,
        root_id: str,
        definition: RelativeDefinition,
        execution: _Execution,
    ) -> _Traversal:
        current: list[Mapping[str, Any]] = [{"id": root_id}]
        final_edges: list[Mapping[str, Any]] = []
        for step in definition.steps:
            next_entities: list[Mapping[str, Any]] = []
            final_edges = []
            for entity in current:
                entity_id = entity.get("id")
                if not isinstance(entity_id, str):
                    continue
                arguments: dict[str, Any] = {
                    "entity_id": entity_id,
                    "relation": step.relation,
                    "limit": execution.max_records,
                }
                if step.direction is not None:
                    arguments["direction"] = step.direction
                result = await execution.call("get_relationships", arguments)
                records = result.get("result")
                if not isinstance(records, list):
                    continue
                for record in records:
                    if not isinstance(record, Mapping):
                        continue
                    related = record.get("related_entity")
                    if not isinstance(related, Mapping):
                        continue
                    if step.gender is not None and related.get("gender") != step.gender:
                        continue
                    final_edges.append(record)
                    _append_unique_entity(next_entities, related)
            current = next_entities
            if not current:
                break
        return _Traversal(tuple(current), tuple(final_edges))

    async def _load_entities(
        self,
        summaries: Sequence[Mapping[str, Any]],
        execution: _Execution,
    ) -> list[Mapping[str, Any]]:
        records: list[Mapping[str, Any]] = []
        for summary in summaries:
            entity_id = summary.get("id")
            if not isinstance(entity_id, str):
                continue
            result = await execution.call("get_entity", {"entity_id": entity_id})
            values = result.get("result")
            if not isinstance(values, list):
                continue
            match = next(
                (
                    value
                    for value in values
                    if isinstance(value, Mapping) and value.get("id") == entity_id
                ),
                None,
            )
            if match is not None:
                records.append(match)
        return records

    async def _home_residents(
        self,
        execution: _Execution,
    ) -> list[Mapping[str, Any]]:
        result = await execution.call(
            "get_relationships",
            {
                "entity_id": self.home_entity_id,
                "relation": "lives_in",
                "limit": execution.max_records,
            },
        )
        residents: list[Mapping[str, Any]] = []
        records = result.get("result")
        if not isinstance(records, list):
            return residents
        for record in records:
            if not isinstance(record, Mapping):
                continue
            _append_unique_entity(residents, record.get("related_entity"))
            nested = record.get("residents")
            if isinstance(nested, list):
                for person in nested:
                    _append_unique_entity(residents, person)
        return residents

    async def _home_spaces(
        self,
        execution: _Execution,
    ) -> list[Mapping[str, Any]]:
        houses_result = await execution.call(
            "get_relationships",
            {
                "entity_id": self.home_entity_id,
                "relation": "located_in",
                "limit": execution.max_records,
            },
        )
        houses: list[Mapping[str, Any]] = []
        house_records = houses_result.get("result")
        if isinstance(house_records, list):
            for record in house_records:
                related = (
                    record.get("related_entity")
                    if isinstance(record, Mapping)
                    else None
                )
                if (
                    isinstance(related, Mapping)
                    and related.get("item_type") == "house"
                ):
                    _append_unique_entity(houses, related)

        spaces: list[Mapping[str, Any]] = []
        for house in houses:
            house_id = house.get("id")
            if not isinstance(house_id, str):
                continue
            hosted_result = await execution.call(
                "get_relationships",
                {
                    "entity_id": house_id,
                    "relation": "hosts_space",
                    "limit": execution.max_records,
                },
            )
            hosted_records = hosted_result.get("result")
            if not isinstance(hosted_records, list):
                continue
            for record in hosted_records:
                related = (
                    record.get("related_entity")
                    if isinstance(record, Mapping)
                    else None
                )
                _append_unique_entity(spaces, related)
        return spaces

    async def _home_entity(
        self,
        execution: _Execution,
    ) -> Mapping[str, Any] | None:
        result = await execution.call(
            "get_entity",
            {"entity_id": self.home_entity_id},
        )
        records = result.get("result")
        if not isinstance(records, list):
            raise _FactFailure("tool_error")
        return next(
            (
                record
                for record in records
                if isinstance(record, Mapping)
                and record.get("id") == self.home_entity_id
            ),
            None,
        )

    async def _answer_item_location(
        self,
        item_name: str,
        execution: _Execution,
        *,
        identity: Mapping[str, Any] | None,
        language: str,
    ) -> str:
        search_result = await execution.call(
            "search_entities",
            {
                "text": item_name,
                "entity_type": "item",
                "limit": execution.max_records,
            },
        )
        item = _exact_or_unique_entity(search_result, item_name, "item")
        if item is None:
            return _no_records_fallback(language)
        item_id = item.get("id")
        if not isinstance(item_id, str):
            raise _FactFailure("tool_error")

        location_result = await execution.call(
            "get_relationships",
            {
                "entity_id": item_id,
                "relation": "located_in",
                "limit": execution.max_records,
            },
        )
        locations: list[Mapping[str, Any]] = []
        records = location_result.get("result")
        if isinstance(records, list):
            for record in records:
                related = (
                    record.get("related_entity")
                    if isinstance(record, Mapping)
                    else None
                )
                _append_unique_entity(locations, related)
        return _format_item_location(item, locations, identity, language)

    async def _answer_space_inventory(
        self,
        space_name: str,
        request: FactRequest,
        execution: _Execution,
        *,
        identity: Mapping[str, Any] | None,
        language: str,
    ) -> str:
        search_result = await execution.call(
            "search_entities",
            {
                "text": space_name,
                "entity_type": "space",
                "limit": execution.max_records,
            },
        )
        space = _exact_or_unique_entity(search_result, space_name, "space")
        if space is None:
            return _no_records_fallback(language)
        space_id = space.get("id")
        if not isinstance(space_id, str):
            raise _FactFailure("tool_error")

        items_result = await execution.call(
            "get_relationships",
            {
                "entity_id": space_id,
                "relation": "located_in",
                "limit": execution.max_records,
            },
        )
        items: list[Mapping[str, Any]] = []
        item_records = items_result.get("result")
        if isinstance(item_records, list):
            for record in item_records:
                related = (
                    record.get("related_entity")
                    if isinstance(record, Mapping)
                    else None
                )
                if (
                    isinstance(related, Mapping)
                    and str(related.get("id", "")).startswith("item:")
                ):
                    _append_unique_entity(items, related)

        if request.field == "items":
            return _format_space_items(space, items, identity, language)

        hosted_spaces: list[Mapping[str, Any]] = []
        for item in items:
            item_id = item.get("id")
            if not isinstance(item_id, str):
                continue
            hosted_result = await execution.call(
                "get_relationships",
                {
                    "entity_id": item_id,
                    "relation": "hosts_space",
                    "limit": execution.max_records,
                },
            )
            hosted_records = hosted_result.get("result")
            if not isinstance(hosted_records, list):
                continue
            for record in hosted_records:
                related = (
                    record.get("related_entity")
                    if isinstance(record, Mapping)
                    else None
                )
                if not isinstance(related, Mapping):
                    continue
                if (
                    request.space_type is not None
                    and related.get("space_type") != request.space_type
                ):
                    continue
                _append_unique_entity(hosted_spaces, related)
        return _format_item_hosted_spaces(
            space,
            hosted_spaces,
            request.space_type,
            identity,
            language,
        )

    async def _answer_named_relationships(
        self,
        names: Sequence[str],
        execution: _Execution,
        *,
        identity: Mapping[str, Any] | None,
        language: str,
    ) -> str:
        people: list[Mapping[str, Any]] = []
        missing: list[str] = []
        for name in names:
            result = await execution.call(
                "search_entities",
                {
                    "text": name,
                    "entity_type": "person",
                    "limit": execution.max_records,
                },
            )
            person = _exact_named_person(result, name)
            if person is None:
                missing.append(name)
            else:
                people.append(person)

        if not people:
            return _no_records_fallback(language)
        labels: dict[str, str] = {}
        if identity is not None:
            identity_id = str(identity["id"])
            for person in people:
                if person.get("id") == identity_id:
                    labels[identity_id] = "self"

            unresolved = [
                person
                for person in people
                if str(person.get("id")) not in labels
            ]
            if unresolved:
                direct = await execution.call(
                    "get_relationships",
                    {
                        "entity_id": identity_id,
                        "limit": execution.max_records,
                    },
                )
                for person in unresolved:
                    label = _direct_relationship_label(
                        direct,
                        str(person["id"]),
                        person,
                    )
                    if label is not None:
                        labels[str(person["id"])] = label

                unresolved = [
                    person
                    for person in unresolved
                    if str(person["id"]) not in labels
                ]
                spouse_ids = _related_ids(direct, "spouse_of")
                for spouse_id in spouse_ids:
                    if not unresolved or execution.tool_calls >= execution.max_tool_calls:
                        break
                    parents = await execution.call(
                        "get_relationships",
                        {
                            "entity_id": spouse_id,
                            "relation": "parent_of",
                            "direction": "in",
                            "limit": execution.max_records,
                        },
                    )
                    parent_ids = set(_related_ids(parents, "parent_of"))
                    for person in unresolved:
                        person_id = str(person["id"])
                        if person_id in parent_ids:
                            labels[person_id] = (
                                "mother_in_law"
                                if person.get("gender") == "female"
                                else "father_in_law"
                                if person.get("gender") == "male"
                                else "parent_in_law"
                            )
                    unresolved = [
                        person
                        for person in unresolved
                        if str(person["id"]) not in labels
                    ]

        return _format_named_relationships(
            people,
            labels,
            missing,
            identity,
            language,
        )

    def _request_date_schema(self, request: FactRequest) -> MemorableDateSchema:
        if request.memorable_date is None:
            raise _FactFailure("tool_error")
        try:
            return self.memorable_dates.get(request.memorable_date)
        except LookupError as error:
            raise _FactFailure("tool_error") from error

    async def _answer_named_memorable_dates(
        self,
        names: Sequence[str],
        execution: _Execution,
        *,
        schema: MemorableDateSchema,
        date_query: DateQuery,
        identity: Mapping[str, Any] | None,
        language: str,
        current_date: date,
    ) -> str:
        summaries: list[Mapping[str, Any]] = []
        missing: list[str] = []
        for name in names:
            result = await execution.call(
                "search_entities",
                {
                    "text": name,
                    "entity_type": "person",
                    "limit": execution.max_records,
                },
            )
            person = _exact_named_person(result, name)
            if person is None:
                missing.append(name)
            else:
                summaries.append(person)
        people = await self._load_entities(summaries, execution)
        return _format_named_memorable_dates(
            people,
            missing,
            self.memorable_dates,
            schema,
            date_query,
            identity,
            language,
            current_date,
        )


def parse_fact_request(
    messages: Sequence[Mapping[str, Any]],
    *,
    identity: Mapping[str, Any] | None,
    memorable_dates: MemorableDateRegistry | None = None,
) -> FactRequest | None:
    """Reduce common fact language to semantics without memorizing sentences."""
    registry = memorable_dates or default_memorable_date_registry()
    text = _latest_user_text(messages)
    normalized = text.casefold()
    date_schema = registry.match(normalized)
    date_query: DateQuery = (
        "next" if _is_next_occurrence(normalized) else "stored"
    )

    # Classify the more specific household and kinship concepts before the
    # generic first-person identity concept.  For example, ``我女儿是谁``
    # contains both "my" and "who", but its subject is the daughter.
    space_inventory = _space_inventory_request(normalized)
    if space_inventory is not None:
        space_name, field, space_type = space_inventory
        return FactRequest(
            SubjectReference("space", space_name),
            field,
            "all",
            relation="located_in" if field == "items" else "hosts_space",
            space_type=space_type,
        )

    if _is_home_address_request(normalized) or (
        _contains_any(normalized, ("address", "地址", "住址"))
        and _previous_home_subject(messages, identity, registry)
    ):
        return FactRequest(SubjectReference("home"), "address")

    item_name = _item_location_subject(normalized)
    if item_name is not None:
        return FactRequest(
            SubjectReference("item", item_name),
            "location",
            relation="located_in",
        )

    if _is_home_identity_request(normalized):
        return FactRequest(SubjectReference("home"), "identity")

    is_home_space_request, home_space_type = _home_space_request(normalized)
    if not is_home_space_request:
        is_home_space_request, home_space_type = _elliptical_home_space_request(
            normalized
        )
    if is_home_space_request:
        field: FactField = (
            "count" if _contains_any(normalized, _COUNT_TERMS) else "spaces"
        )
        return FactRequest(
            SubjectReference("home"),
            field,
            "all",
            relation="hosts_space",
            space_type=home_space_type,
        )

    if _is_home_roster_count_request(normalized) or (
        _is_elliptical_roster_count_request(normalized)
        and _previous_home_subject(messages, identity, registry)
    ):
        return FactRequest(
            SubjectReference("home"),
            "count",
            "all",
            relation="lives_in",
        )

    if _is_roster_request(normalized, registry.aliases):
        return FactRequest(SubjectReference("home"), "residents", "all")

    relative = _relative_kind(normalized)
    if relative is not None and identity is not None:
        if date_schema is not None:
            date_relative = (
                "spouse" if date_schema.source_kind == "edge" else relative
            )
            return FactRequest(
                SubjectReference("relative", date_relative),
                "memorable_date",
                memorable_date=date_schema.id,
                date_query=date_query,
            )
        if _is_home_residence_check(normalized):
            return FactRequest(
                SubjectReference("relative", relative),
                "relationship_exists",
                relation="lives_in",
                target=SubjectReference("home"),
            )
        if _contains_any(normalized, _COUNT_TERMS):
            return FactRequest(SubjectReference("relative", relative), "count", "all")
        if _contains_any(normalized, _LOOKUP_TERMS):
            return FactRequest(
                SubjectReference("relative", relative),
                "identity",
                "all" if relative in {"children", "parents"} else "one",
            )

    if (
        identity is not None
        and date_schema is not None
        and date_schema.source_kind == "edge"
    ):
        return FactRequest(
            SubjectReference("relative", "spouse"),
            "memorable_date",
            memorable_date=date_schema.id,
            date_query=date_query,
        )

    if _is_speaker_identity(normalized, identity):
        return FactRequest(SubjectReference("speaker"), "identity")

    # A named-person fact is meaningful here only when it can be related to a
    # trusted caller identity.  Without one, let the ordinary guarded model
    # loop handle the conversation instead of guessing the user's perspective.
    names = _named_subjects(text) if identity is not None else ()
    if names:
        return FactRequest(
            SubjectReference("named", names),
            "relationship_to_speaker",
            "all" if len(names) > 1 else "one",
        )

    # Resolve an elliptical factual follow-up from the last explicit semantic
    # subject in the conversation.  This supports pronouns such as "他们" and
    # short follow-ups such as "生日呢" without asking the model to remember a
    # special sentence.
    if identity is not None and date_schema is not None:
        inherited = _previous_fact_subject(messages, identity, registry)
        if inherited is not None:
            return FactRequest(
                inherited,
                "memorable_date",
                "all",
                memorable_date=date_schema.id,
                date_query=date_query,
            )
    return None


def _latest_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
    return next(
        (
            str(message.get("content", "")).strip()
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )


def _is_next_occurrence(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in _NEXT_OCCURRENCE_PATTERNS)


def _is_home_address_request(text: str) -> bool:
    chinese = bool(
        re.fullmatch(
            r"(?:我们?家|这个家|家里|家中|^家的?)"
            r"(?:的)?(?:"
            r"(?:住在|位于|在)?(?:哪里|哪儿|哪|在哪)|"
            r"(?:位置|地址|住址)(?:是|在|位于)?"
            r"(?:哪里|哪儿|哪)?"
            r")"
            r"[?？。！!]?",
            text,
        )
    )
    english = bool(
        re.search(r"\b(?:my|our|the)\s+(?:home|house)\b", text)
        and _contains_any(text, ("address", "street address"))
    )
    return chinese or english


def _is_home_identity_request(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:这|这里|这儿|这个地方)(?:是)?(?:哪里|哪儿|哪)[?？。！!]?",
            text,
        )
    )


def _home_space_request(text: str) -> tuple[bool, str | None]:
    home_terms = _HOME_REFERENCE_TERMS + ("the home", "the house", "这个家")
    if not _contains_any(text, home_terms) or not _contains_any(text, _SPACE_TERMS):
        return False, None
    if _contains_any(text, _SPACE_NON_LOOKUP_TERMS):
        return False, None
    if not _contains_any(text, _SPACE_LOOKUP_TERMS):
        return False, None
    if _contains_any(text, _ROOM_TERMS):
        return True, "room"
    return True, None


def _elliptical_home_space_request(text: str) -> tuple[bool, str | None]:
    """Resolve a bare room/space lookup against the configured home."""
    chinese = re.fullmatch(
        r"(?:一共|总共)?(?:有)?(?:哪些|什么|多少|几(?:个|间)?)"
        r"(?:房间|空间|区域)[?？。！!]?",
        text,
    )
    if chinese is not None:
        field_type = "room" if "房间" in text else None
        return True, field_type
    english = re.fullmatch(
        r"(?:what|which|how many) (?:rooms|spaces)(?: are there)?[?]?",
        text,
    )
    if english is not None:
        return True, "room" if "room" in text else None
    return False, None


def _is_home_roster_count_request(text: str) -> bool:
    chinese_home = r"(?:家里|家中|家里面|家里边|这个家|这里)"
    chinese_count = r"(?:多少|几)(?:个|位)?(?:人|住户|居民)"
    if re.search(rf"{chinese_home}.*{chinese_count}", text):
        return True
    if re.search(
        rf"{chinese_count}.*(?:住|居住|待在).*(?:{chinese_home})",
        text,
    ):
        return True
    return bool(
        re.search(r"\bhow many\b.*\b(?:people|residents|occupants)\b", text)
        and re.search(r"\b(?:my|our|the)\s+(?:home|house|household)\b", text)
    )


def _is_elliptical_roster_count_request(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:一共|总共)?(?:有)?(?:多少|几)(?:个|位)?人[?？。！!]?",
            text,
        )
        or re.fullmatch(
            r"how many (?:people|residents|occupants)(?: are there)?[?]?",
            text,
        )
    )


def _is_home_adequacy_request(text: str) -> bool:
    """Identify home judgments that benefit from facts but need model reasoning."""
    chinese_direct = re.search(
        r"(?:够不够住|够住|住得下|住不下|住得开|住不开|住得舒服|"
        r"住着舒服|住.*(?:拥挤|宽敞|合适|适合))",
        text,
    )
    chinese_home = re.search(
        r"(?:家里|家中|我家|我们家|房子|住宅|房间|空间)",
        text,
    )
    chinese_evaluation = re.search(
        r"(?:够不够|够|合适|适合|舒服|舒适|拥挤|宽敞|"
        r"大不大|大吗|小吗)",
        text,
    )
    english = re.search(
        r"\b(?:enough|adequate|comfortable|crowded|spacious|suitable|fit)\b",
        text,
    ) and re.search(r"\b(?:home|house|room|rooms|live|living)\b", text)
    return (
        chinese_direct is not None
        or (chinese_home is not None and chinese_evaluation is not None)
        or bool(english)
    )


def _item_location_subject(text: str) -> str | None:
    chinese = re.fullmatch(
        r"(?:请问|麻烦告诉我)?\s*(?P<name>.+?)\s*"
        r"(?:现在)?(?:在|位于)(?:哪里|哪儿|什么地方|哪个房间|哪间房间?)"
        r"[?？。！!]?",
        text,
    )
    if chinese is not None:
        name = re.sub(
            r"^(?:我家(?:的)?|家里(?:的)?|我的)",
            "",
            chinese.group("name").strip(),
        )
        if name in {
            "家",
            "家里",
            "家中",
            "这个家",
            "这里",
            "这儿",
            "地址",
            "住址",
            "位置",
        }:
            return None
        if _contains_any(
            name,
            tuple(alias for _, aliases in _RELATIVE_ALIASES for alias in aliases),
        ):
            return None
        return name or None

    patterns = (
        r"(?:where is|where's)\s+(?:the\s+|my\s+|our\s+)?"
        r"(?P<name>.+?)[?]?",
        r"(?:which|what) room is\s+(?:the\s+|my\s+|our\s+)?"
        r"(?P<name>.+?)\s+in[?]?",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match is not None:
            name = match.group("name").strip()
            if name in {"home", "house", "household", "here"}:
                return None
            return name or None
    return None


def _space_inventory_request(
    text: str,
) -> tuple[str, Literal["items", "spaces"], str | None] | None:
    chinese = re.fullmatch(
        r"(?:请问)?\s*(?:家里(?:的)?)?(?P<name>.+?)"
        r"(?:里面|里|内)\s*(?:又)?(?P<query>.+?)[?？。！!]?",
        text,
    )
    if chinese is not None:
        name = chinese.group("name").strip()
        query = chinese.group("query").strip()
        if _contains_any(query, ("空间", "区域", "地方")):
            space_type = (
                "storage"
                if _contains_any(query, ("放东西", "储物", "存储", "收纳"))
                else None
            )
            return name, "spaces", space_type
        if _contains_any(query, ("物品", "东西", "设备", "家电")):
            return name, "items", None
        return None

    english = re.fullmatch(
        r"(?:what|which)\s+(?P<kind>items|things|appliances|"
        r"storage spaces|spaces)\s+(?:are\s+)?(?:in|inside)\s+"
        r"(?:the\s+)?(?P<name>.+?)[?]?",
        text,
    )
    if english is None:
        return None
    kind = english.group("kind")
    field: Literal["items", "spaces"] = (
        "spaces" if "spaces" in kind else "items"
    )
    space_type = "storage" if kind == "storage spaces" else None
    return english.group("name").strip(), field, space_type


def _previous_home_subject(
    messages: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any] | None,
    memorable_dates: MemorableDateRegistry,
) -> bool:
    latest_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        -1,
    )
    if latest_index <= 0:
        return False
    previous = parse_fact_request(
        messages[:latest_index],
        identity=identity,
        memorable_dates=memorable_dates,
    )
    return previous is not None and previous.subject.kind == "home"


def _is_home_residence_check(text: str) -> bool:
    if not (
        _contains_any(text, _RESIDENCE_TERMS)
        and _contains_any(text, _HOME_REFERENCE_TERMS)
    ):
        return False
    return bool(
        re.search(r"(?:吗[？?]?$|是否|是不是|有没有)", text)
        or re.match(r"\s*(?:do|does|is|are)\b", text)
    )


def _previous_fact_subject(
    messages: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    memorable_dates: MemorableDateRegistry,
) -> SubjectReference | None:
    latest_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        -1,
    )
    if latest_index <= 0:
        return None
    previous = parse_fact_request(
        messages[:latest_index],
        identity=identity,
        memorable_dates=memorable_dates,
    )
    if previous is None or previous.subject.kind not in {"named", "relative"}:
        return None
    return previous.subject


def _relative_kind(text: str) -> str | None:
    for kind, aliases in _RELATIVE_ALIASES:
        if _contains_any(text, aliases):
            return kind
    return None


def _is_speaker_identity(
    text: str,
    identity: Mapping[str, Any] | None,
) -> bool:
    if (
        identity is None
        or not _entity_aliases(identity)
        or not _contains_any(text, _LOOKUP_TERMS + ("身份", "名字", "叫什么"))
    ):
        return False
    references_self = bool(
        re.search(r"(?:^|\W)(?:i|me|myself)(?:$|\W)", text)
        or re.search(r"我|本人|自己", text)
    )
    if references_self:
        return True
    return any(alias.casefold() in text for alias in _entity_aliases(identity))


def _is_roster_request(text: str, date_aliases: Sequence[str]) -> bool:
    if _contains_any(
        text,
        tuple(date_aliases)
        + tuple(alias for _, aliases in _RELATIVE_ALIASES for alias in aliases),
    ):
        return False
    chinese_home = r"(?:家里|家中|家里面|家里边|这个家|这里)"
    chinese_people = r"(?:谁|哪些人|什么人|成员|住户)"
    if re.search(
        rf"(?:{chinese_home}.*{chinese_people}|"
        rf"{chinese_people}.*(?:住|居住|待在).*{chinese_home})",
        text,
    ):
        return True
    patterns = (
        r"\bwho\b.*\b(?:live|lives|living|reside|resides|stays?)\b.*"
        r"\b(?:home|house|household|here)\b",
        r"\b(?:household|home|house)\s+(?:members|residents|occupants)\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _named_subjects(text: str) -> tuple[str, ...]:
    chinese = re.fullmatch(
        r"(?:请问|麻烦告诉我)?\s*(?P<names>.+?)\s*是谁[？?]?",
        text,
    )
    if chinese is not None:
        raw = re.split(r"\s*(?:和|与|以及|、)\s*", chinese.group("names"))
    else:
        english = re.fullmatch(
            r"(?:please\s+)?(?:tell me\s+)?who (?:is|are)\s+"
            r"(?P<names>.+?)[?]?",
            text,
            re.IGNORECASE,
        )
        if english is None:
            return ()
        raw = re.split(
            r"\s+(?:and|&)\s+",
            english.group("names"),
            flags=re.IGNORECASE,
        )

    excluded = {
        "i",
        "me",
        "you",
        "he",
        "she",
        "this person",
        "that person",
        "someone",
        "我",
        "你",
        "您",
        "他",
        "她",
        "这个人",
        "那个人",
        "某人",
    }
    names: list[str] = []
    seen: set[str] = set()
    for value in raw:
        name = re.sub(
            r"(?:先生|女士|小姐|太太|夫人|mr\.?|mrs\.?|ms\.?)$",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip()
        normalized = name.casefold()
        if name and normalized not in excluded and normalized not in seen:
            seen.add(normalized)
            names.append(name)
    return tuple(names)


def _append_unique_entity(
    values: list[Mapping[str, Any]],
    candidate: Any,
) -> None:
    if not isinstance(candidate, Mapping):
        return
    record_id = candidate.get("id")
    if not isinstance(record_id, str):
        return
    if all(value.get("id") != record_id for value in values):
        values.append(candidate)


def _entity_aliases(entity: Mapping[str, Any]) -> tuple[str, ...]:
    name = entity.get("name")
    if isinstance(name, str):
        return (name.strip(),) if name.strip() else ()
    if isinstance(name, Mapping):
        return tuple(
            value.strip()
            for value in name.values()
            if isinstance(value, str) and value.strip()
        )
    if isinstance(name, Sequence) and not isinstance(
        name,
        (str, bytes, bytearray),
    ):
        return tuple(
            value.strip()
            for value in name
            if isinstance(value, str) and value.strip()
        )
    return ()


def _exact_named_person(
    result: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any] | None:
    records = result.get("result")
    if not isinstance(records, list):
        return None
    normalized = name.strip().casefold()
    matches = [
        record
        for record in records
        if isinstance(record, Mapping)
        and str(record.get("id", "")).startswith("person:")
        and (
            normalized == str(record.get("id", "")).casefold()
            or normalized in {alias.casefold() for alias in _entity_aliases(record)}
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _exact_or_unique_entity(
    result: Mapping[str, Any],
    name: str,
    entity_type: str,
) -> Mapping[str, Any] | None:
    records = result.get("result")
    if not isinstance(records, list):
        return None
    typed = [
        record
        for record in records
        if isinstance(record, Mapping)
        and str(record.get("id", "")).startswith(f"{entity_type}:")
    ]
    normalized = name.strip().casefold()
    exact = [
        record
        for record in typed
        if normalized == str(record.get("id", "")).casefold()
        or normalized in {alias.casefold() for alias in _entity_aliases(record)}
    ]
    if len(exact) == 1:
        return exact[0]
    return typed[0] if len(typed) == 1 else None


def _direct_relationship_label(
    result: Mapping[str, Any],
    target_id: str,
    target: Mapping[str, Any],
) -> str | None:
    records = result.get("result")
    if not isinstance(records, list):
        return None
    gender = str(target.get("gender", "")).casefold()
    for record in records:
        related = record.get("related_entity") if isinstance(record, Mapping) else None
        if not isinstance(related, Mapping) or related.get("id") != target_id:
            continue
        if record.get("relation") == "spouse_of":
            return {"female": "wife", "male": "husband"}.get(gender, "spouse")
        if record.get("relation") == "parent_of":
            if record.get("direction") == "outgoing":
                return {"female": "daughter", "male": "son"}.get(
                    gender,
                    "child",
                )
            if record.get("direction") == "incoming":
                return {"female": "mother", "male": "father"}.get(
                    gender,
                    "parent",
                )
    return None


def _related_ids(result: Mapping[str, Any], relation: str) -> list[str]:
    records = result.get("result")
    if not isinstance(records, list):
        return []
    values: list[str] = []
    for record in records:
        related = (
            record.get("related_entity")
            if isinstance(record, Mapping) and record.get("relation") == relation
            else None
        )
        record_id = related.get("id") if isinstance(related, Mapping) else None
        if isinstance(record_id, str) and record_id not in values:
            values.append(record_id)
    return values


def _format_speaker_identity(
    identity: Mapping[str, Any],
    language: str,
) -> str:
    name = resolve_display_name(identity, language)
    address = _speaker_address(identity, language)
    if language == "zh":
        prefix = f"{address}，" if address else ""
        return f"{prefix}您是{name}。"
    prefix = f"{address}, " if address else ""
    answer = f"{prefix}you are {name}."
    return answer if prefix else answer[0].upper() + answer[1:]


def _format_boolean_answer(
    value: bool,
    identity: Mapping[str, Any] | None,
    language: str,
) -> str:
    address = _speaker_address(identity, language)
    if language == "zh":
        prefix = f"{address}，" if address else ""
        return f"{prefix}{'是的' if value else '不是'}。"
    prefix = f"{address}, " if address else ""
    answer = f"{prefix}{'yes' if value else 'no'}."
    return answer if prefix else answer[0].upper() + answer[1:]


def _format_residents(
    residents: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any] | None,
    language: str,
) -> str:
    if not residents:
        return _no_records_fallback(language)
    names = [resolve_display_name(person, language) for person in residents]
    address = _speaker_address(identity, language)
    if language == "zh":
        prefix = f"{address}，" if address else ""
        heading = f"{prefix}目前家里的住户有："
    else:
        heading = (
            f"{address}, the current household residents are:"
            if address
            else "The current household residents are:"
        )
    return heading + "\n" + "\n".join(f"- {name}" for name in names)


def _format_resident_count(
    residents: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any] | None,
    language: str,
) -> str:
    address = _speaker_address(identity, language)
    if language == "zh":
        prefix = f"{address}，" if address else ""
        return f"{prefix}家里目前有 {len(residents)} 位住户。"
    prefix = f"{address}, " if address else ""
    label = "resident" if len(residents) == 1 else "residents"
    answer = f"{prefix}the home currently has {len(residents)} {label}."
    return answer if prefix else answer[0].upper() + answer[1:]


def _format_item_location(
    item: Mapping[str, Any],
    locations: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any] | None,
    language: str,
) -> str:
    if not locations:
        return _no_records_fallback(language)
    item_name = resolve_display_name(item, language)
    location_names = [
        resolve_display_name(location, language) for location in locations
    ]
    address = _speaker_address(identity, language)
    if language == "zh":
        prefix = f"{address}，" if address else ""
        return (
            f"{prefix}{item_name}在"
            f"{_join_localized(location_names, language)}。"
        )
    prefix = f"{address}, " if address else ""
    answer = (
        f"{prefix}{item_name} is in "
        f"{_join_localized(location_names, language)}."
    )
    return answer if prefix else answer[0].upper() + answer[1:]


def _format_space_items(
    space: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any] | None,
    language: str,
) -> str:
    space_name = resolve_display_name(space, language)
    address = _speaker_address(identity, language)
    if not items:
        if language == "zh":
            prefix = f"{address}，" if address else ""
            return f"{prefix}家庭资料中没有记录位于{space_name}的物品。"
        prefix = f"{address}, " if address else ""
        answer = f"{prefix}no items are recorded in {space_name}."
        return answer if prefix else answer[0].upper() + answer[1:]

    names = [resolve_display_name(item, language) for item in items]
    if language == "zh":
        prefix = f"{address}，" if address else ""
        return (
            f"{prefix}{space_name}里的物品有："
            f"{_join_localized(names, language)}。"
        )
    prefix = f"{address}, " if address else ""
    answer = (
        f"{prefix}the items in {space_name} are "
        f"{_join_localized(names, language)}."
    )
    return answer if prefix else answer[0].upper() + answer[1:]


def _format_item_hosted_spaces(
    parent_space: Mapping[str, Any],
    hosted_spaces: Sequence[Mapping[str, Any]],
    space_type: str | None,
    identity: Mapping[str, Any] | None,
    language: str,
) -> str:
    parent_name = resolve_display_name(parent_space, language)
    address = _speaker_address(identity, language)
    label = "储物空间" if space_type == "storage" else "空间"
    if not hosted_spaces:
        if language == "zh":
            prefix = f"{address}，" if address else ""
            return (
                f"{prefix}家庭资料中没有记录由{parent_name}内物品提供的"
                f"{label}。"
            )
        prefix = f"{address}, " if address else ""
        answer = (
            f"{prefix}no hosted spaces are recorded for items in "
            f"{parent_name}."
        )
        return answer if prefix else answer[0].upper() + answer[1:]

    names = [resolve_display_name(space, language) for space in hosted_spaces]
    if language == "zh":
        prefix = f"{address}，" if address else ""
        return (
            f"{prefix}{parent_name}内物品提供的{label}有："
            f"{_join_localized(names, language)}。"
        )
    prefix = f"{address}, " if address else ""
    answer = (
        f"{prefix}the spaces hosted by items in {parent_name} are "
        f"{_join_localized(names, language)}."
    )
    return answer if prefix else answer[0].upper() + answer[1:]


def _format_home_space_count(
    spaces: Sequence[Mapping[str, Any]],
    space_type: str | None,
    identity: Mapping[str, Any] | None,
    language: str,
) -> str:
    address = _speaker_address(identity, language)
    if language == "zh":
        prefix = f"{address}，" if address else ""
        label = "房间" if space_type == "room" else "空间"
        return f"{prefix}家里共有 {len(spaces)} 个{label}。"
    prefix = f"{address}, " if address else ""
    label = "room" if space_type == "room" else "space"
    if len(spaces) != 1:
        label += "s"
    answer = f"{prefix}the home has {len(spaces)} {label}."
    return answer if prefix else answer[0].upper() + answer[1:]


def _format_home_spaces(
    spaces: Sequence[Mapping[str, Any]],
    space_type: str | None,
    identity: Mapping[str, Any] | None,
    language: str,
) -> str:
    if not spaces:
        return _no_records_fallback(language)
    names = [resolve_display_name(space, language) for space in spaces]
    address = _speaker_address(identity, language)
    if language == "zh":
        prefix = f"{address}，" if address else ""
        label = "房间" if space_type == "room" else "空间"
        return f"{prefix}家里的{label}有：{_join_localized(names, language)}。"
    prefix = f"{address}, " if address else ""
    label = "rooms" if space_type == "room" else "spaces"
    answer = f"{prefix}the home's {label} are {_join_localized(names, language)}."
    return answer if prefix else answer[0].upper() + answer[1:]


def _format_home_address(
    home: Mapping[str, Any] | None,
    identity: Mapping[str, Any] | None,
    language: str,
) -> str:
    address = _localized_address(home.get("address"), language) if home else None
    speaker_address = _speaker_address(identity, language)
    if not address:
        if language == "zh":
            prefix = f"{speaker_address}，" if speaker_address else ""
            return f"{prefix}家庭资料中没有记录家的地址。"
        prefix = f"{speaker_address}, " if speaker_address else ""
        answer = f"{prefix}the home address is not recorded."
        return answer if prefix else answer[0].upper() + answer[1:]

    name = resolve_display_name(home, language) if home else ""
    if INTERNAL_ID_PATTERN.fullmatch(name):
        name = ""
    if language == "zh":
        prefix = f"{speaker_address}，" if speaker_address else ""
        label = f"家（{name}）" if name else "家"
        return f"{prefix}{label}的地址是 {address}。"
    prefix = f"{speaker_address}, " if speaker_address else ""
    label = f"your home ({name})" if name else "your home"
    answer = f"{prefix}{label} is at {address}."
    return answer if prefix else answer[0].upper() + answer[1:]


def _format_home_identity(
    home: Mapping[str, Any] | None,
    identity: Mapping[str, Any] | None,
    language: str,
) -> str:
    if home is None:
        return _no_records_fallback(language)
    name = resolve_display_name(home, language)
    if INTERNAL_ID_PATTERN.fullmatch(name):
        return _no_records_fallback(language)
    speaker_address = _speaker_address(identity, language)
    if language == "zh":
        prefix = f"{speaker_address}，" if speaker_address else ""
        return f"{prefix}这里是{name}。"
    prefix = f"{speaker_address}, " if speaker_address else ""
    answer = f"{prefix}this is {name}."
    return answer if prefix else answer[0].upper() + answer[1:]


def _localized_address(value: Any, language: str) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        parts = [
            str(part).strip()
            for part in value
            if part is not None and str(part).strip()
        ]
        return ", ".join(parts) or None
    if not isinstance(value, Mapping):
        return None

    structured_keys = {
        "street",
        "street2",
        "unit",
        "city",
        "state",
        "zip",
        "postal_code",
        "country",
    }
    if not structured_keys.intersection(value):
        localized = value.get(language) or value.get("en")
        if not isinstance(localized, str):
            localized = next(
                (item for item in value.values() if isinstance(item, str)),
                None,
            )
        return (
            localized.strip()
            if isinstance(localized, str) and localized.strip()
            else None
        )

    street_parts = [
        str(value[key]).strip()
        for key in ("street", "street2", "unit")
        if value.get(key) is not None and str(value[key]).strip()
    ]
    city = str(value.get("city", "")).strip()
    state = str(value.get("state", "")).strip()
    postal = str(value.get("zip") or value.get("postal_code") or "").strip()
    region = " ".join(filter(None, (state, postal)))
    locality = ", ".join(part for part in (city, region) if part)
    country = str(value.get("country", "")).strip()
    parts = [
        *street_parts,
        *([locality] if locality else []),
        *([country] if country else []),
    ]
    return ", ".join(parts) or None


def _format_relatives(
    people: Sequence[Mapping[str, Any]],
    relative: str,
    identity: Mapping[str, Any],
    language: str,
) -> str:
    if not people:
        return _no_records_fallback(language)
    names = [resolve_display_name(person, language) for person in people]
    address = _speaker_address(identity, language)
    if language == "zh":
        prefix = f"{address}，" if address else ""
        return (
            f"{prefix}您的{_relative_label(relative, language, len(people) != 1)}是"
            f"{_join_localized(names, language)}。"
        )
    prefix = f"{address}, " if address else ""
    label = _relative_label(relative, language, len(people) != 1)
    verb = "are" if len(people) != 1 else "is"
    answer = f"{prefix}your {label} {verb} {_join_localized(names, language)}."
    return answer if prefix else answer[0].upper() + answer[1:]


def _format_relative_count(
    people: Sequence[Mapping[str, Any]],
    relative: str,
    identity: Mapping[str, Any],
    language: str,
) -> str:
    address = _speaker_address(identity, language)
    names = [resolve_display_name(person, language) for person in people]
    if language == "zh":
        prefix = f"{address}，" if address else ""
        descriptions = [
            _described_relative(person, name, relative)
            for person, name in zip(people, names, strict=True)
        ]
        suffix = (
            f"：{_join_localized(descriptions, language)}" if descriptions else ""
        )
        return (
            f"{prefix}您有{_chinese_count(len(people))}"
            f"{_relative_label(relative, language, True)}{suffix}。"
        )
    prefix = f"{address}, " if address else ""
    label = _relative_label(relative, language, len(people) != 1)
    suffix = f": {_join_localized(names, language)}" if names else ""
    answer = f"{prefix}you have {len(people)} {label}{suffix}."
    return answer if prefix else answer[0].upper() + answer[1:]


def _format_relative_memorable_dates(
    people: Sequence[Mapping[str, Any]],
    relative: str,
    registry: MemorableDateRegistry,
    schema: MemorableDateSchema,
    date_query: DateQuery,
    identity: Mapping[str, Any],
    language: str,
    current_date: date,
) -> str:
    if not people:
        return _no_records_fallback(language)
    address = _speaker_address(identity, language)
    multiple = len(people) > 1
    clauses: list[str] = []
    for person in people:
        name = resolve_display_name(person, language)
        if language == "zh":
            owner = (
                name
                if multiple
                else f"您的{_relative_label(relative, language, False)}{name}"
            )
        else:
            owner = (
                name
                if multiple
                else f"your {_relative_label(relative, language, False)} {name}"
            )
        phrase = _owned_memorable_date_phrase(owner, schema, language)
        clauses.append(
            _format_memorable_date_clause(
                phrase,
                person.get(schema.source_field),
                registry,
                schema,
                date_query,
                language,
                current_date,
            )
        )
    return _finish_clauses(clauses, address, language)


def _format_named_memorable_dates(
    people: Sequence[Mapping[str, Any]],
    missing: Sequence[str],
    registry: MemorableDateRegistry,
    schema: MemorableDateSchema,
    date_query: DateQuery,
    identity: Mapping[str, Any] | None,
    language: str,
    current_date: date,
) -> str:
    if not people and not missing:
        return _no_records_fallback(language)
    clauses: list[str] = []
    for person in people:
        owner = resolve_display_name(person, language)
        phrase = _owned_memorable_date_phrase(owner, schema, language)
        clauses.append(
            _format_memorable_date_clause(
                phrase,
                person.get(schema.source_field),
                registry,
                schema,
                date_query,
                language,
                current_date,
            )
        )
    address = _speaker_address(identity, language)
    if language == "zh":
        clauses.extend(f'家庭资料中没有找到“{name}”' for name in missing)
    else:
        clauses.extend(f'no household record matched "{name}"' for name in missing)
    return _finish_clauses(clauses, address, language)


def _format_edge_memorable_date(
    edges: Sequence[Mapping[str, Any]],
    registry: MemorableDateRegistry,
    schema: MemorableDateSchema,
    date_query: DateQuery,
    identity: Mapping[str, Any],
    language: str,
    current_date: date,
) -> str:
    matching = [
        edge
        for edge in edges
        if edge.get("relation") == schema.source_type
        and isinstance(edge.get(schema.source_field), str)
    ]
    if not matching:
        return _no_records_fallback(language)
    if len(matching) > 1:
        return _ambiguous_fallback(language)
    edge = matching[0]
    related = edge.get("related_entity")
    name = (
        resolve_display_name(related, language)
        if isinstance(related, Mapping)
        else None
    )
    if name and INTERNAL_ID_PATTERN.fullmatch(name):
        name = None
    address = _speaker_address(identity, language)
    label = _memorable_date_label(schema, language)
    if language == "zh":
        phrase = f"您与{name}的{label}" if name else f"您的{label}"
    else:
        phrase = f"your {label} with {name}" if name else f"your {label}"
    clause = _format_memorable_date_clause(
        phrase,
        edge.get(schema.source_field),
        registry,
        schema,
        date_query,
        language,
        current_date,
    )
    return _finish_clauses([clause], address, language)


def _owned_memorable_date_phrase(
    owner: str,
    schema: MemorableDateSchema,
    language: str,
) -> str:
    label = _memorable_date_label(schema, language)
    if language == "zh":
        return f"{owner}的{label}"
    possessive = f"{owner}'" if owner.endswith("s") else f"{owner}'s"
    return f"{possessive} {label}"


def _memorable_date_label(schema: MemorableDateSchema, language: str) -> str:
    return schema.label.get(language) or schema.label.get("en") or schema.id


def _format_memorable_date_clause(
    phrase: str,
    value: Any,
    registry: MemorableDateRegistry,
    schema: MemorableDateSchema,
    date_query: DateQuery,
    language: str,
    current_date: date,
) -> str:
    occurrence = registry.occurrence(schema, value, as_of=current_date)
    if occurrence is None:
        return (
            f"家庭资料中没有记录{phrase}"
            if language == "zh"
            else f"{phrase} is not recorded"
        )
    if date_query == "stored":
        value_text = _localized_date(occurrence.stored_date.isoformat(), language)
        return (
            f"{phrase}是{value_text}"
            if language == "zh"
            else f"{phrase} is {value_text}"
        )
    if occurrence.days_until == 0:
        return f"{phrase}就是今天" if language == "zh" else f"{phrase} is today"
    return (
        f"{phrase}还有{occurrence.days_until}天"
        if language == "zh"
        else f"{phrase} is in {occurrence.days_until} days"
    )


def _finish_clauses(
    clauses: Sequence[str],
    address: str | None,
    language: str,
) -> str:
    prefix = f"{address}，" if address and language == "zh" else (
        f"{address}, " if address else ""
    )
    separator = "；" if language == "zh" else "; "
    answer = prefix + separator.join(clauses) + ("。" if language == "zh" else ".")
    return answer if prefix or language == "zh" else answer[0].upper() + answer[1:]


def _format_named_relationships(
    people: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    missing: Sequence[str],
    identity: Mapping[str, Any] | None,
    language: str,
) -> str:
    address = _speaker_address(identity, language)
    if language == "zh":
        prefix = f"{address}，" if address else ""
        clauses: list[str] = []
        for person in people:
            name = resolve_display_name(person, language)
            label = labels.get(str(person.get("id")))
            if label == "self":
                clauses.append(f"您是{name}")
            elif label is not None:
                clauses.append(f"{name}是您的{_relationship_label(label, language)}")
            else:
                clauses.append(
                    f"{name}记录在家庭资料中，但目前没有查到此人与您的亲属关系"
                )
        clauses.extend(f'家庭资料中没有找到“{name}”' for name in missing)
        return f"{prefix}{'；'.join(clauses)}。"

    prefix = f"{address}, " if address else ""
    clauses = []
    for person in people:
        name = resolve_display_name(person, language)
        label = labels.get(str(person.get("id")))
        if label == "self":
            clauses.append(f"you are {name}")
        elif label is not None:
            clauses.append(f"{name} is your {_relationship_label(label, language)}")
        else:
            clauses.append(
                f"{name} is recorded in the household graph, but no relationship "
                "to you was found"
            )
    clauses.extend(f'no household record matched "{name}"' for name in missing)
    answer = f"{prefix}{'; '.join(clauses)}."
    return answer if prefix else answer[0].upper() + answer[1:]


def _speaker_address(
    identity: Mapping[str, Any] | None,
    language: str,
) -> str | None:
    if identity is None:
        return None
    # ``resolve_person_reference(..., mode="address")`` deliberately falls
    # back to a person's name.  A salutation must be stricter: use only an
    # explicit user preference and otherwise omit the vocative.
    if not identity.get("address_as"):
        return None
    value = resolve_person_reference(identity, language, mode="address")
    return value if value and not INTERNAL_ID_PATTERN.fullmatch(value) else None


def _relative_label(relative: str, language: str, plural: bool) -> str:
    if language == "zh":
        return {
            "spouse": "配偶",
            "wife": "太太",
            "husband": "丈夫",
            "children": "孩子",
            "son": "儿子",
            "daughter": "女儿",
            "parents": "父母",
            "father": "父亲",
            "mother": "母亲",
            "father_in_law": "岳父",
            "mother_in_law": "岳母",
        }.get(relative, "亲属")
    singular = {
        "spouse": "spouse",
        "wife": "wife",
        "husband": "husband",
        "children": "child",
        "son": "son",
        "daughter": "daughter",
        "parents": "parent",
        "father": "father",
        "mother": "mother",
        "father_in_law": "father-in-law",
        "mother_in_law": "mother-in-law",
    }.get(relative, "relative")
    if not plural:
        return singular
    return {"child": "children", "wife": "wives"}.get(singular, f"{singular}s")


def _relationship_label(label: str, language: str) -> str:
    if language == "zh":
        return {
            "wife": "太太",
            "husband": "丈夫",
            "spouse": "配偶",
            "daughter": "女儿",
            "son": "儿子",
            "child": "孩子",
            "mother": "母亲",
            "father": "父亲",
            "parent": "父母之一",
            "mother_in_law": "岳母",
            "father_in_law": "岳父",
            "parent_in_law": "岳父母之一",
        }.get(label, "亲属")
    return label.replace("_", "-")


def _described_relative(
    person: Mapping[str, Any],
    name: str,
    relative: str,
) -> str:
    if relative != "children":
        return name
    if person.get("gender") == "male":
        return f"儿子{name}"
    if person.get("gender") == "female":
        return f"女儿{name}"
    return name


def _chinese_count(count: int) -> str:
    return {0: "零个", 1: "一个", 2: "两个", 3: "三个", 4: "四个"}.get(
        count,
        f"{count}个",
    )


def _join_localized(values: Sequence[str], language: str) -> str:
    if len(values) <= 1:
        return values[0] if values else ""
    conjunction = "和" if language == "zh" else " and "
    separator = "、" if language == "zh" else ", "
    if len(values) == 2:
        return conjunction.join(values)
    return separator.join(values[:-1]) + conjunction + values[-1]


def _localized_date(value: str, language: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return value
    if language == "zh":
        return f"{parsed.year}年{parsed.month}月{parsed.day}日"
    months = (
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    )
    return f"{months[parsed.month - 1]} {parsed.day}, {parsed.year}"


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def _grounding_fallback(language: str) -> str:
    if language == "zh":
        return "老管家目前无法从家庭资料中核实这项信息。"
    return "The butler could not verify that from the household graph."


def _no_records_fallback(language: str) -> str:
    if language == "zh":
        return "家庭资料中没有找到与这个问题匹配的信息。"
    return "No matching information was found in the household graph."


def _ambiguous_fallback(language: str) -> str:
    if language == "zh":
        return "家庭资料中有多个可能的结果，请说明您指的是哪一位。"
    return "Multiple household records match; please clarify which one you mean."


def _safe_log_token(value: str, maximum_length: int = 128) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.:-]", "_", value)
    return sanitized[:maximum_length]
