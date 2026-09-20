"""Explicit execution boundary for planner-approved household mutations."""
from __future__ import annotations

from typing import Any

from ..facts.engine import HouseholdFactEngine
from .ir import NAMED_WRITE_ADAPTER, NamedWriteRequest
from ..semantic.ir import AgentRequestContext
from .semantic import NamedItemWritingService
from .writing import ItemWritingService


class MutationService:
    """Resolve, validate, preview, and commit typed item mutation requests."""

    def __init__(
        self,
        writing: ItemWritingService,
        engine: HouseholdFactEngine,
    ) -> None:
        self.writing = writing
        self.named_writing = NamedItemWritingService(writing, engine)

    async def execute(
        self,
        request: NamedWriteRequest | dict[str, Any],
        context: AgentRequestContext,
    ) -> dict[str, Any]:
        typed = NAMED_WRITE_ADAPTER.validate_python(request)
        result = await self.named_writing.mutate(typed, context)
        return {"ok": True, "tool": "write_item", "result": result}
