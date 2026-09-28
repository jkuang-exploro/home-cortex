"""Canonical embodiment telemetry. Clients submit one fused estimate."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ...spatial.presence import (
    EmbodimentPresence, TelemetryAdmissionError, admission_as_mapping,
    telemetry_state_as_mapping,
)
from ...spatial.primitives import SpatialContractError
from ..dependencies import authenticate_request
from ..errors import APIError


router = APIRouter()


def _presence(request: Request) -> EmbodimentPresence:
    presence = getattr(request.app.state, "embodiment_presence", None)
    if not isinstance(presence, EmbodimentPresence):
        raise APIError(503, "telemetry_unavailable", "Embodiment telemetry is not configured")
    return presence


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
        admission = _presence(request).submit(body)
    except TelemetryAdmissionError as error:
        raise _admission_error(error) from error
    except SpatialContractError as error:
        raise APIError(422, "invalid_telemetry", str(error)) from error
    return admission_as_mapping(admission)


@router.get("/v1/embodiments/{embodiment_id}/telemetry")
async def latest_telemetry(embodiment_id: str, request: Request) -> dict[str, Any]:
    authenticate_request(request)
    try:
        state = _presence(request).latest(embodiment_id)
    except TelemetryAdmissionError as error:
        raise _admission_error(error) from error
    except SpatialContractError as error:
        raise APIError(422, "invalid_telemetry", str(error)) from error
    return telemetry_state_as_mapping(state)


def _admission_error(error: TelemetryAdmissionError) -> APIError:
    status = {"unknown_embodiment": 404, "conflicting_sample": 409}.get(error.code, 422)
    return APIError(status, error.code, str(error))
