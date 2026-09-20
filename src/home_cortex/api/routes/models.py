"""Model discovery route."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ...agents import list_agents
from ..dependencies import authenticate_request
from ..providers import list_bare_models
from ..schemas import MODEL_CREATED


router = APIRouter()


@router.get("/v1/models")
async def models(request: Request) -> dict[str, Any]:
    authenticate_request(request)
    agents = [
        {
            "id": definition.display_name,
            "object": "model",
            "created": MODEL_CREATED,
            "owned_by": "home-cortex",
            "kind": "agent",
        }
        for definition in list_agents()
    ]
    agent_ids = {item["id"] for item in agents}
    bare = [
        {
            "id": item["id"],
            "object": "model",
            "created": MODEL_CREATED,
            "owned_by": item["owned_by"],
            "kind": "model",
        }
        for item in await list_bare_models(request)
        if item["id"] not in agent_ids
    ]
    return {"object": "list", "data": [*agents, *bare]}

