from collections.abc import Awaitable, Callable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .calculate import CalculationError, evaluate_expression
from .calendar import (
    CALENDAR_ID_PATTERN,
    CalendarAuthorizationError,
    CalendarService,
    CalendarUnavailableError,
    PERSON_ID_PATTERN,
)
from .retrieval import RetrievalService
from .record_ids import RECORD_ID_PATTERN as CANONICAL_RECORD_ID_PATTERN

TABLE_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"
RECORD_ID_PATTERN = CANONICAL_RECORD_ID_PATTERN
_GRAPH_OPERATIONS = frozenset(
    {"resolve_entity_alias", "get_entity", "get_relationships"}
)

_caller_entity_id: ContextVar[str | None] = ContextVar(
    "home_cortex_caller_entity_id",
    default=None,
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


class ToolArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )


class ResolveEntityAliasArguments(ToolArguments):
    text: str = Field(min_length=1, max_length=256)
    entity_type: str | None = Field(default=None, pattern=TABLE_NAME_PATTERN)
    limit: int | None = Field(default=None, ge=1, le=100)
    speaker_id: str | None = Field(default=None, pattern=RECORD_ID_PATTERN)
    household_id: str | None = Field(default=None, pattern=RECORD_ID_PATTERN)


class GetEntityArguments(ToolArguments):
    entity_id: str = Field(pattern=RECORD_ID_PATTERN)


class GetRelationshipsArguments(ToolArguments):
    entity_id: str = Field(pattern=RECORD_ID_PATTERN)
    relation: str | None = Field(default=None, pattern=TABLE_NAME_PATTERN)
    direction: Literal["out", "in", "both"] | None = None
    include_ended: bool = False
    limit: int | None = Field(default=None, ge=1, le=100)


class CalculateArguments(ToolArguments):
    expression: str = Field(min_length=1, max_length=256)


class ListEventsArguments(ToolArguments):
    start: str = Field(min_length=1)
    end: str = Field(min_length=1)
    calendar: str | None = Field(default=None, pattern=CALENDAR_ID_PATTERN)
    person: str | None = Field(default=None, pattern=PERSON_ID_PATTERN)
    limit: int | None = Field(default=None, ge=1, le=100)


class CheckAvailabilityArguments(ToolArguments):
    start: str = Field(min_length=1)
    end: str = Field(min_length=1)
    calendar: str | None = Field(default=None, pattern=CALENDAR_ID_PATTERN)
    person: str | None = Field(default=None, pattern=PERSON_ID_PATTERN)


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate exact arithmetic locally. Use this for any numerical "
                "computation instead of estimating. Supports + - * / // % **, "
                "parentheses, numeric literals, pi, e, tau, and functions such "
                "as abs, round, min, max, sqrt, log, exp, sin, and cos. "
                "Arbitrary code is rejected."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "expression": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 256,
                        "description": (
                            "Arithmetic expression to evaluate, for example "
                            "'(4350 * 12) / 365' or '2 + 3 * 4'."
                        ),
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar.list_events",
            "description": (
                "List household calendar events in a date-time range. "
                "Use for schedules, plans, and questions such as what the "
                "speaker has tomorrow. Compute start and end from the trusted "
                "household clock; never guess the current date. Dates preserve "
                "timezone. Defaults to calendars the authenticated speaker may "
                "read. Do not pass another person's ID or a calendar ID unless "
                "the user asked about that calendar and the caller is "
                "authorized. If complete is false, do not present the returned "
                "events as the complete schedule; identify unavailable_calendars "
                "or truncated_calendars as partial sources. Read-only; events are "
                "not stored in the household graph."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Inclusive range start as an ISO 8601 date or "
                            "datetime. Date-only values use the household "
                            "timezone."
                        ),
                    },
                    "end": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Range end as an ISO 8601 date or datetime. "
                            "Datetimes are exclusive. A date-only end equal to "
                            "start means that whole household-local day."
                        ),
                    },
                    "calendar": {
                        "type": "string",
                        "pattern": CALENDAR_ID_PATTERN,
                        "description": (
                            "Optional Cortex calendar ID, such as jian_primary. "
                            "Never a Google credential or provider token."
                        ),
                    },
                    "person": {
                        "type": "string",
                        "pattern": PERSON_ID_PATTERN,
                        "description": (
                            "Optional person record ID whose authorized "
                            "calendars should be queried, such as "
                            "person:household_member."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum number of events to return.",
                    },
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar.check_availability",
            "description": (
                "Check whether a date-time window is free of busy events on "
                "authorized household calendars. Compute start and end from "
                "the trusted household clock. available is true only when every "
                "requested calendar was read and has no busy conflicts. If "
                "checked is false, do not claim the window is free. Read-only. "
                "Unauthorized calendars fail closed."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Inclusive window start as an ISO 8601 date or "
                            "datetime."
                        ),
                    },
                    "end": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Window end as an ISO 8601 date or datetime. "
                            "Datetimes are exclusive. A date-only end equal to "
                            "start means that whole household-local day."
                        ),
                    },
                    "calendar": {
                        "type": "string",
                        "pattern": CALENDAR_ID_PATTERN,
                        "description": "Optional Cortex calendar ID to check.",
                    },
                    "person": {
                        "type": "string",
                        "pattern": PERSON_ID_PATTERN,
                        "description": (
                            "Optional person record ID whose authorized "
                            "calendars should be checked."
                        ),
                    },
                },
                "required": ["start", "end"],
            },
        },
    },
]


def get_tool_definitions(tool_names: Sequence[str]) -> list[dict[str, Any]]:
    """Return definitions for an agent's allowlisted tools, in policy order."""
    catalog = {tool["function"]["name"]: tool for tool in TOOLS}
    unknown = sorted(set(tool_names) - catalog.keys())
    if unknown:
        raise ValueError(f"Unknown tool names: {', '.join(unknown)}")
    return [catalog[name] for name in tool_names]


Handler = Callable[[ToolArguments], Awaitable[Any]]


class ToolDispatcher:
    """Validate and execute the small allowlist of model-facing tools."""

    def __init__(
        self,
        retrieval: RetrievalService,
        allowed_tools: Sequence[str] | None = None,
        *,
        calendar: CalendarService | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.calendar = calendar
        argument_models: dict[str, type[ToolArguments]] = {
            "resolve_entity_alias": ResolveEntityAliasArguments,
            "get_entity": GetEntityArguments,
            "get_relationships": GetRelationshipsArguments,
            "calculate": CalculateArguments,
            "calendar.list_events": ListEventsArguments,
            "calendar.check_availability": CheckAvailabilityArguments,
        }
        handlers: dict[str, Handler] = {
            "resolve_entity_alias": self._resolve_entity_alias,
            "get_entity": self._get_entity,
            "get_relationships": self._get_relationships,
            "calculate": self._calculate,
            "calendar.list_events": self._list_events,
            "calendar.check_availability": self._check_availability,
        }
        selected_public = tuple(allowed_tools) if allowed_tools is not None else tuple(
            name for name in handlers if name not in _GRAPH_OPERATIONS
        )
        unknown = sorted(set(selected_public) - (handlers.keys() - _GRAPH_OPERATIONS))
        if unknown:
            raise ValueError(f"Unknown tool names: {', '.join(unknown)}")
        self._public_tools = frozenset(selected_public)
        self._argument_models = argument_models
        self._handlers = handlers

    async def dispatch(
        self,
        tool_name: str,
        arguments: Any,
        *,
        caller_entity_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._dispatch(
            tool_name,
            arguments,
            caller_entity_id=caller_entity_id,
            allow_internal=False,
        )

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
                "The tool could not complete its read operation",
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
