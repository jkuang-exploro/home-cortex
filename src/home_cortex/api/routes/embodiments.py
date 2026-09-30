"""Read-only embodiment directory; physical client writes use /v1 routes."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ...agents.presence import EmbodimentDirectory
from ...spatial.primitives import SpatialContractError
from ..dependencies import authenticate_request, agent_definition
from ..errors import APIError


router = APIRouter()


def embodiment_directory(request: Request) -> EmbodimentDirectory:
    directory = getattr(request.app.state, "embodiment_directory", None)
    if not isinstance(directory, EmbodimentDirectory):
        raise APIError(503, "embodiments_unavailable", "Embodiment directory is not configured")
    return directory


@router.get("/api/embodiments")
async def list_embodiments(request: Request, agent_id: str | None = None) -> list[dict[str, Any]]:
    authenticate_request(request)
    directory = embodiment_directory(request)
    return (directory.list_for_agent(agent_definition(agent_id).entity_id)
            if agent_id else directory.list())


@router.get("/api/embodiments/{embodiment_id}")
async def get_embodiment(embodiment_id: str, request: Request) -> dict[str, Any]:
    authenticate_request(request)
    try:
        return embodiment_directory(request).get(embodiment_id)
    except SpatialContractError as error:
        raise APIError(404, "embodiment_not_found", "Embodiment was not found") from error
