"""Fiducial observation contract. Detectors must emit this; they do not own ontology."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from .anchors import parse_local_anchor_id
from ..primitives import SpatialContractError
from .pose import measured_orientation, measured_position
from ..transforms import Pose, pose

_OBSERVATION_FIELDS = frozenset({
    "anchor",
    "anchor_id",
    "confidence",
    "orientation",
    "position",
    "reprojection_error_px",
    "timestamp",
})


@dataclass(frozen=True)
class FiducialObservation:
    """Pose of a surveyed anchor in the camera frame, plus detection quality."""

    anchor_id: str
    pose: Pose
    timestamp: str
    confidence: float | None = None
    reprojection_error_px: float | None = None


def parse_fiducial_observation(value: Mapping[str, Any]) -> FiducialObservation:
    if not isinstance(value, Mapping):
        raise SpatialContractError("fiducial observation must be an object")
    extra = sorted(set(value) - _OBSERVATION_FIELDS)
    if extra:
        raise SpatialContractError(
            f"Unknown fiducial observation fields: {', '.join(extra)}"
        )
    raw_id = value.get("anchor", value.get("anchor_id"))
    if "position" not in value or "orientation" not in value:
        raise SpatialContractError(
            "fiducial observation requires position and orientation"
        )
    if "timestamp" not in value:
        raise SpatialContractError("fiducial observation requires timestamp")
    confidence = _unit_interval(value.get("confidence"), "confidence") if "confidence" in value else None
    error_px = (
        _non_negative(value.get("reprojection_error_px"), "reprojection_error_px")
        if "reprojection_error_px" in value
        else None
    )
    position = measured_position(value["position"])
    orientation = measured_orientation(value["orientation"])
    return FiducialObservation(
        anchor_id=parse_local_anchor_id(raw_id),
        pose=pose(
            x=position.x, y=position.y, z=position.z,
            yaw=orientation.yaw, pitch=orientation.pitch, roll=orientation.roll,
        ),
        timestamp=_timestamp(value.get("timestamp")),
        confidence=confidence,
        reprojection_error_px=error_px,
    )


def parse_fiducial_observations(
    values: Sequence[Mapping[str, Any]] | None,
) -> tuple[FiducialObservation, ...]:
    if values is None:
        return ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise SpatialContractError("fiducial observations must be a list")
    return tuple(parse_fiducial_observation(item) for item in values)


def _timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpatialContractError("timestamp must be an ISO-8601 string")
    try:
        datetime.fromisoformat(value)
    except ValueError as error:
        raise SpatialContractError("timestamp must be an ISO-8601 string") from error
    return value


def _unit_interval(value: Any, field: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0 or value > 1:
        raise SpatialContractError(f"{field} must be a finite number in [0, 1]")
    return float(value)


def _non_negative(value: Any, field: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise SpatialContractError(f"{field} must be a finite non-negative number")
    return float(value)
