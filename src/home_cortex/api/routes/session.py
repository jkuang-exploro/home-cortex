"""Runtime session for a client that already has a durable embodiment."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ...agents.embodiments import SessionProtocolError
from ...agents.session import EmbodimentSession
from ...spatial.primitives import SpatialContractError
from ..dependencies import authenticate_request
from ..errors import APIError


router = APIRouter()
SESSION_ID_HEADER = "X-Embodiment-Session-ID"


def embodiment_session(request: Request) -> EmbodimentSession:
    session = getattr(request.app.state, "embodiment_session", None)
    if not isinstance(session, EmbodimentSession):
        raise APIError(503, "session_unavailable", "Embodiment sessions are not configured")
    return session


@router.post("/v1/embodiments/{embodiment_id}/session")
async def open_session(
    embodiment_id: str, body: dict[str, Any], request: Request,
) -> dict[str, Any]:
    authenticate_request(request)
    return _change(request, embodiment_id, body, connect=True)


@router.get("/v1/embodiments/{embodiment_id}/session")
async def read_session(embodiment_id: str, request: Request) -> dict[str, Any]:
    authenticate_request(request)
    try:
        return embodiment_session(request).view(embodiment_id)
    except SessionProtocolError as error:
        raise _session_error(error) from error


@router.delete("/v1/embodiments/{embodiment_id}/session")
async def close_session(embodiment_id: str, request: Request) -> dict[str, Any]:
    authenticate_request(request)
    try:
        return embodiment_session(request).disconnect(
            embodiment_id, session_id=request.headers.get(SESSION_ID_HEADER)
        )
    except SessionProtocolError as error:
        raise _session_error(error) from error


@router.post("/v1/embodiments/{embodiment_id}/session/capabilities")
async def refresh_capabilities(
    embodiment_id: str, body: dict[str, Any], request: Request,
) -> dict[str, Any]:
    authenticate_request(request)
    return _change(request, embodiment_id, body, connect=False)


@router.post("/v1/embodiments/{embodiment_id}/session/heartbeat")
async def session_heartbeat(embodiment_id: str, request: Request) -> dict[str, Any]:
    authenticate_request(request)
    try:
        return embodiment_session(request).heartbeat(
            embodiment_id, session_id=request.headers.get(SESSION_ID_HEADER)
        )
    except SessionProtocolError as error:
        raise _session_error(error) from error


def _change(
    request: Request, embodiment_id: str, body: dict[str, Any], *, connect: bool,
) -> dict[str, Any]:
    if set(body) != {"available_capabilities"}:
        raise APIError(422, "invalid_session", "Send only available_capabilities")
    available = body["available_capabilities"]
    if not isinstance(available, list):
        raise APIError(422, "invalid_session", "available_capabilities must be a list")
    session = embodiment_session(request)
    try:
        if connect:
            return session.register(embodiment_id, available)
        return session.update_capabilities(
            embodiment_id, available,
            session_id=request.headers.get(SESSION_ID_HEADER),
        )
    except SessionProtocolError as error:
        raise _session_error(error) from error
    except SpatialContractError as error:
        raise APIError(422, "invalid_session", str(error)) from error


def _session_error(error: SessionProtocolError) -> APIError:
    status = {
        "unknown_embodiment": 404,
        "session_offline": 409,
        "unknown_space": 422,
        "conflicting_sample": 409,
        "stale_session": 409,
    }.get(error.code, 422)
    return APIError(status, error.code, str(error))
