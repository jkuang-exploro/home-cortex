"""Canonical embodiment telemetry. Clients submit one fused estimate."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ...agents.embodiments import SessionProtocolError
from ...spatial.presence import telemetry_state_as_mapping
from ...spatial.primitives import SpatialContractError
from ..dependencies import authenticate_request
from ..errors import APIError
from .session import SESSION_ID_HEADER, _session_error, embodiment_session


router = APIRouter()


@router.post("/v1/embodiments/{embodiment_id}/telemetry")
async def submit_telemetry(
    embodiment_id: str,
    body: dict[str, Any],
    request: Request,
) -> dict[str, Any]:
    authenticate_request(request)
    if body.get("embodiment_id") != embodiment_id:
        raise APIError(422, "invalid_telemetry", "Path embodiment_id must match the sample")
    try:
        return embodiment_session(request).submit_telemetry(
            body, session_id=request.headers.get(SESSION_ID_HEADER)
        )
    except SessionProtocolError as error:
        raise _session_error(error) from error
    except SpatialContractError as error:
        raise APIError(422, "invalid_telemetry", str(error)) from error


@router.get("/v1/embodiments/{embodiment_id}/telemetry")
async def latest_telemetry(embodiment_id: str, request: Request) -> dict[str, Any]:
    authenticate_request(request)
    try:
        session = embodiment_session(request)
        session.view(embodiment_id)
        state = session.presence.latest(embodiment_id)
    except SessionProtocolError as error:
        raise _session_error(error) from error
    except SpatialContractError as error:
        raise APIError(422, "invalid_telemetry", str(error)) from error
    return telemetry_state_as_mapping(state)
