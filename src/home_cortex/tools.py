from collections.abc import Awaitable, Callable
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
                "Each relationship includes the complete linked record in "
                "related_entity."
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

Handler = Callable[[ToolArguments], Awaitable[list[dict[str, Any]]]]


class ToolDispatcher:
    """Validate and execute the small allowlist of model-facing tools."""

    def __init__(self, retrieval: RetrievalService) -> None:
        self.retrieval = retrieval
        self._argument_models: dict[str, type[ToolArguments]] = {
            "search_entities": SearchEntitiesArguments,
            "get_relationships": GetRelationshipsArguments,
        }
        self._handlers: dict[str, Handler] = {
            "search_entities": self._search_entities,
            "get_relationships": self._get_relationships,
        }

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
