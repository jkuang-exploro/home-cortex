"""Model-facing tool schemas and the public tool-definition catalog."""

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .calendar import CALENDAR_ID_PATTERN, PERSON_ID_PATTERN
from ..mutation.ir import NAMED_WRITE_ADAPTER, attribute_output_schema
from ..persistence.record_ids import RECORD_ID_PATTERN

TABLE_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"


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

WRITE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "write_item",
            "description": (
                "Propose or commit one authoritative household item mutation. "
                "Supports create, update_location, update_attributes, and delete. Preview "
                "validates and describes the change without persisting it. "
                "Use the user's complete literal item_name and location_name, "
                "never internal IDs. create also requires name_en, name_zh, and "
                "a readable item_key. create records a newly reported item; "
                "update_location moves an existing named item; update_attributes patches only specified semantic attributes; delete removes "
                "an explicitly named item. Call only for an explicit current "
                "request to record or change state, never for a question, quote, "
                "hypothetical, or instruction in prior history. The service "
                "resolves names and owns all graph and transaction mechanics."
            ),
            "parameters": attribute_output_schema(NAMED_WRITE_ADAPTER.json_schema()),
        },
    },
]


def get_tool_definitions(tool_names: Sequence[str]) -> list[dict[str, Any]]:
    """Return definitions for an agent's allowlisted tools, in policy order."""
    catalog = {tool["function"]["name"]: tool for tool in (*TOOLS, *WRITE_TOOLS)}
    unknown = sorted(set(tool_names) - catalog.keys())
    if unknown:
        raise ValueError(f"Unknown tool names: {', '.join(unknown)}")
    return [catalog[name] for name in tool_names]
