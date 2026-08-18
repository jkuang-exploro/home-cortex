from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .retrieval import RetrievalService

TABLE_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"
RECORD_ID_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*:[A-Za-z0-9_-]+$"


class ToolArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )


class SearchEntitiesArguments(ToolArguments):
    text: str = Field(min_length=1)
    entity_type: str | None = Field(default=None, pattern=TABLE_NAME_PATTERN)
    limit: int | None = Field(default=None, ge=1, le=100)


class GetRelationshipsArguments(ToolArguments):
    entity_id: str = Field(pattern=RECORD_ID_PATTERN)
    relation: str | None = Field(default=None, pattern=TABLE_NAME_PATTERN)
    limit: int | None = Field(default=None, ge=1, le=100)


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_entities",
            "description": (
                "Search known home-graph entities by record ID, multilingual "
                "name aliases, or other text fields. "
                "Pass only the distinctive entity name or ID, not the user's "
                "full question. Use this before requesting relationships."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Case-insensitive entity name in any stored language, "
                            "short text, or record ID to find; for example, "
                            "'Fort Cerritos'."
                        ),
                    },
                    "entity_type": {
                        "type": "string",
                        "pattern": TABLE_NAME_PATTERN,
                        "description": (
                            "Optional node table, such as person or location."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum number of entities to return.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_relationships",
            "description": (
                "Get incoming and outgoing relationships for one known entity. "
                "Each relationship preserves stable IDs and includes complete "
                "records in entity and related_entity for reasoning and "
                "localized display-name selection."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "pattern": RECORD_ID_PATTERN,
                        "description": "Entity record ID in table:record_id format.",
                    },
                    "relation": {
                        "type": "string",
                        "pattern": TABLE_NAME_PATTERN,
                        "description": (
                            "Optional relationship table, such as resides_in."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum number of relationships to return.",
                    },
                },
                "required": ["entity_id"],
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


Handler = Callable[[ToolArguments], Awaitable[list[dict[str, Any]]]]


class ToolDispatcher:
    """Validate and execute the small allowlist of model-facing tools."""

    def __init__(
        self,
        retrieval: RetrievalService,
        allowed_tools: Sequence[str] | None = None,
    ) -> None:
        self.retrieval = retrieval
        argument_models: dict[str, type[ToolArguments]] = {
            "search_entities": SearchEntitiesArguments,
            "get_relationships": GetRelationshipsArguments,
        }
        handlers: dict[str, Handler] = {
            "search_entities": self._search_entities,
            "get_relationships": self._get_relationships,
        }
        selected = (
            tuple(allowed_tools) if allowed_tools is not None else tuple(handlers)
        )
        unknown = sorted(set(selected) - handlers.keys())
        if unknown:
            raise ValueError(f"Unknown tool names: {', '.join(unknown)}")
        self._argument_models = {name: argument_models[name] for name in selected}
        self._handlers = {name: handlers[name] for name in selected}

    async def dispatch(
        self,
        tool_name: str,
        arguments: Any,
    ) -> dict[str, Any]:
        argument_model = self._argument_models.get(tool_name)
        handler = self._handlers.get(tool_name)
        if argument_model is None or handler is None:
            return self._error(
                tool_name,
                "unknown_tool",
                f"Tool {tool_name!r} is not available",
                available_tools=sorted(self._handlers),
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

        try:
            result = await handler(validated)
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

    async def _search_entities(
        self,
        arguments: ToolArguments,
    ) -> list[dict[str, Any]]:
        assert isinstance(arguments, SearchEntitiesArguments)
        return await self.retrieval.search_entities(
            arguments.text,
            entity_type=arguments.entity_type,
            limit=arguments.limit,
        )

    async def _get_relationships(
        self,
        arguments: ToolArguments,
    ) -> list[dict[str, Any]]:
        assert isinstance(arguments, GetRelationshipsArguments)
        return await self.retrieval.get_relationships(
            arguments.entity_id,
            relation=arguments.relation,
            limit=arguments.limit,
        )

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
