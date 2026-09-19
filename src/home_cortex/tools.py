from collections.abc import Awaitable, Callable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ValidationError

from .calculate import CalculationError, evaluate_expression
from .calendar import CalendarAuthorizationError, CalendarService, CalendarUnavailableError
from .retrieval import RetrievalService
from .writing import ItemWritingService
from .mutation_service import MutationService
from .mutation_ir import NamedWriteItemArguments
from .semantic_ir import AgentRequestContext
from .household_fact_engine import HouseholdFactEngine
from .semantic_schema import SemanticSchemaRegistry
from .tool_catalog import (
    TOOLS,
    WRITE_TOOLS,
    CalculateArguments,
    CheckAvailabilityArguments,
    GetEntityArguments,
    GetRelationshipsArguments,
    ListEventsArguments,
    ResolveEntityAliasArguments,
    ToolArguments,
    get_tool_definitions,
)
_GRAPH_OPERATIONS = frozenset(
    {"resolve_entity_alias", "get_entity", "get_relationships"}
)
_MUTATION_OPERATIONS = frozenset({"write_item"})

_caller_entity_id: ContextVar[str | None] = ContextVar(
    "home_cortex_caller_entity_id",
    default=None,
)


_request_context: ContextVar[AgentRequestContext | None] = ContextVar(
    "home_cortex_request_context", default=None,
)


@contextmanager
def tool_caller_scope(entity_id: str | None):
    """Bind the authenticated person ID for the current tool dispatch."""
    token = _caller_entity_id.set(entity_id)
    try:
        yield
    finally:
        _caller_entity_id.reset(token)


def current_caller_entity_id() -> str | None:
    return _caller_entity_id.get()


Handler = Callable[[BaseModel], Awaitable[Any]]


class ToolDispatcher:
    """Validate and execute the small allowlist of model-facing tools."""

    def __init__(
        self,
        retrieval: RetrievalService,
        allowed_tools: Sequence[str] | None = None,
        *,
        calendar: CalendarService | None = None,
        writing: ItemWritingService | None = None,
        household_id: str | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.calendar = calendar
        self.writing = writing
        self.household_id = household_id
        self.mutations = (
            MutationService(
                writing,
                HouseholdFactEngine(
                    self,
                    SemanticSchemaRegistry(writing.catalog, writing.ontology),
                ),
            )
            if writing is not None
            else None
        )
        argument_models: dict[str, type[BaseModel]] = {
            "resolve_entity_alias": ResolveEntityAliasArguments,
            "get_entity": GetEntityArguments,
            "get_relationships": GetRelationshipsArguments,
            "calculate": CalculateArguments,
            "calendar.list_events": ListEventsArguments,
            "calendar.check_availability": CheckAvailabilityArguments,
            "write_item": NamedWriteItemArguments,
        }
        handlers: dict[str, Handler] = {
            "resolve_entity_alias": self._resolve_entity_alias,
            "get_entity": self._get_entity,
            "get_relationships": self._get_relationships,
            "calculate": self._calculate,
            "calendar.list_events": self._list_events,
            "calendar.check_availability": self._check_availability,
            "write_item": self._write_item,
        }
        selected_public = tuple(allowed_tools) if allowed_tools is not None else tuple(
            name
            for name in handlers
            if name not in _GRAPH_OPERATIONS | _MUTATION_OPERATIONS
        )
        unknown = sorted(set(selected_public) - (handlers.keys() - _GRAPH_OPERATIONS))
        if unknown:
            raise ValueError(f"Unknown tool names: {', '.join(unknown)}")
        if "write_item" in selected_public and self.writing is None:
            raise ValueError("write_item requires an ItemWritingService")
        self._public_tools = frozenset(selected_public)
        self._argument_models = argument_models
        self._handlers = handlers

    async def dispatch(
        self,
        tool_name: str,
        arguments: Any,
        *,
        caller_entity_id: str | None = None,
        request_context: AgentRequestContext | None = None,
    ) -> dict[str, Any]:
        token = _request_context.set(request_context)
        try:
            return await self._dispatch(
                tool_name, arguments,
                caller_entity_id=(request_context.caller_entity_id
                                  if request_context else caller_entity_id),
                allow_internal=False,
            )
        finally:
            _request_context.reset(token)

    async def dispatch_internal(
        self,
        tool_name: str,
        arguments: Any,
        *,
        caller_entity_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute graph operations unavailable to model tool calls."""
        return await self._dispatch(
            tool_name,
            arguments,
            caller_entity_id=caller_entity_id,
            allow_internal=True,
        )

    async def _dispatch(
        self,
        tool_name: str,
        arguments: Any,
        *,
        caller_entity_id: str | None,
        allow_internal: bool,
    ) -> dict[str, Any]:
        argument_model = self._argument_models.get(tool_name)
        handler = self._handlers.get(tool_name)
        internal_allowed = allow_internal and tool_name in _GRAPH_OPERATIONS
        if (
            argument_model is None
            or handler is None
            or (tool_name not in self._public_tools and not internal_allowed)
        ):
            return self._error(
                tool_name,
                "unknown_tool",
                f"Tool {tool_name!r} is not available",
                available_tools=sorted(self._public_tools),
            )

        if not isinstance(arguments, dict):
            return self._error(
                tool_name,
                "invalid_arguments",
                "Tool arguments must be a JSON object",
            )

        try:
            validated = argument_model.model_validate(arguments)
        except ValidationError as error:
            return self._error(
                tool_name,
                "invalid_arguments",
                "Tool arguments failed validation",
                details=[
                    {
                        "field": ".".join(str(part) for part in item["loc"]),
                        "message": item["msg"],
                        "type": item["type"],
                    }
                    for item in error.errors(include_input=False, include_url=False)
                ],
            )

        if caller_entity_id is None:
            caller_entity_id = current_caller_entity_id()
        try:
            with tool_caller_scope(caller_entity_id):
                result = await handler(validated)
        except CalculationError as error:
            return self._error(
                tool_name,
                "calculation_error",
                str(error),
            )
        except CalendarAuthorizationError as error:
            return self._error(
                tool_name,
                "unauthorized",
                str(error),
            )
        except CalendarUnavailableError:
            return self._error(
                tool_name,
                "calendar_unavailable",
                "The calendar service is temporarily unavailable",
            )
        except ValueError as error:
            return self._error(
                tool_name,
                "invalid_arguments",
                str(error),
            )
        except Exception:
            return self._error(
                tool_name,
                "tool_execution_failed",
                (
                    "The tool could not complete its mutation"
                    if tool_name in _MUTATION_OPERATIONS
                    else "The tool could not complete its read operation"
                ),
            )

        return {
            "ok": True,
            "tool": tool_name,
            "result": result,
        }

    async def _resolve_entity_alias(
        self,
        arguments: ToolArguments,
    ) -> list[dict[str, Any]]:
        assert isinstance(arguments, ResolveEntityAliasArguments)
        return await self.retrieval.resolve_entity_alias(
            arguments.text,
            entity_type=arguments.entity_type,
            limit=arguments.limit,
            speaker_id=arguments.speaker_id,
            household_id=arguments.household_id,
        )

    async def _get_entity(
        self,
        arguments: ToolArguments,
    ) -> list[dict[str, Any]]:
        assert isinstance(arguments, GetEntityArguments)
        record = await self.retrieval.get_entity(arguments.entity_id)
        return [] if record is None else [record]

    async def _get_relationships(
        self,
        arguments: ToolArguments,
    ) -> list[dict[str, Any]]:
        assert isinstance(arguments, GetRelationshipsArguments)
        return await self.retrieval.get_relationships(
            arguments.entity_id,
            relation=arguments.relation,
            direction=arguments.direction,
            limit=arguments.limit,
            include_ended=arguments.include_ended,
        )

    async def _calculate(self, arguments: ToolArguments) -> dict[str, Any]:
        assert isinstance(arguments, CalculateArguments)
        value = evaluate_expression(arguments.expression)
        return {"result": value}

    async def _write_item(self, arguments: BaseModel) -> dict[str, Any]:
        assert isinstance(arguments, NamedWriteItemArguments)
        assert self.mutations is not None
        context = _request_context.get() or AgentRequestContext(
            caller_entity_id=current_caller_entity_id(), household_id=self.household_id,
            assistant_id="writer", assistant_display_name="writer",
            current_time=datetime.now(timezone.utc), locale="en",
        )
        response = await self.mutations.execute(arguments.root, context)
        return response["result"]

    async def _list_events(self, arguments: ToolArguments) -> dict[str, Any]:
        assert isinstance(arguments, ListEventsArguments)
        return await self._calendar_service().list_events(
            start=arguments.start,
            end=arguments.end,
            calendar_id=arguments.calendar,
            person_id=arguments.person,
            limit=arguments.limit,
            caller_entity_id=current_caller_entity_id(),
        )

    async def _check_availability(self, arguments: ToolArguments) -> dict[str, Any]:
        assert isinstance(arguments, CheckAvailabilityArguments)
        return await self._calendar_service().check_availability(
            start=arguments.start,
            end=arguments.end,
            calendar_id=arguments.calendar,
            person_id=arguments.person,
            caller_entity_id=current_caller_entity_id(),
        )

    def _calendar_service(self) -> CalendarService:
        if self.calendar is None:
            raise CalendarUnavailableError(
                "The calendar service is not configured"
            )
        return self.calendar

    @staticmethod
    def _error(
        tool_name: str,
        code: str,
        message: str,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "tool": tool_name,
            "error": {
                "code": code,
                "message": message,
                **extra,
            },
        }
