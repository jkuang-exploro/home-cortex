"""Runtime robot pose contract. Not a persistent household fact stream."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from ..record_ids import split_record_id
from .contracts import SpatialContractError
from .transforms import Orientation, Pose, Position, pose
from .units import SpatialUnitError, as_meters, as_radians

LOCALIZATION_QUALITY = (
    "unknown",
    "initializing",
    "localized",
    "degraded",
    "lost",
)
_POSE_QUALITY = frozenset({"localized", "degraded"})
_UNCERTAINTY_FIELDS = (
    "x_sigma",
    "y_sigma",
    "z_sigma",
    "yaw_sigma",
    "pitch_sigma",
    "roll_sigma",
)
_RUNTIME_FIELDS = frozenset({
    "agent",
    "orientation",
    "position",
    "quality",
    "source",
    "space",
    "timestamp",
    "uncertainty",
})
_POSITION_FIELDS = frozenset({"x", "y", "z"})
_ORIENTATION_FIELDS = frozenset({"pitch", "roll", "yaw"})


@dataclass(frozen=True)
class PoseUncertainty:
    """Non-negative one-sigma terms in meters and radians. Absent means unstated."""

    x_sigma: float | None = None
    y_sigma: float | None = None
    z_sigma: float | None = None
    yaw_sigma: float | None = None
    pitch_sigma: float | None = None
    roll_sigma: float | None = None


@dataclass(frozen=True)
class RuntimePose:
    """Current estimate for an embodied agent in a known space.

    Missing pose is represented by ``estimate is None``. Coordinates are never
    inferred from ``space.json``; the robot may start anywhere in the room.
    """

    agent: str
    space: str
    quality: str
    estimate: Pose | None = None
    timestamp: str | None = None
    source: str | None = None
    uncertainty: PoseUncertainty | None = None


def parse_runtime_pose(value: Mapping[str, Any] | None) -> RuntimePose | None:
    """Parse an ephemeral pose document. ``None`` means no estimate is available."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SpatialContractError("runtime pose must be an object or null")
    extra = sorted(set(value) - _RUNTIME_FIELDS)
    if extra:
        raise SpatialContractError(
            f"Unknown runtime pose fields: {', '.join(extra)}"
        )
    agent = _typed_id(value.get("agent"), "agent", "agent")
    space = _typed_id(value.get("space"), "space", "space")
    has_pose = "position" in value or "orientation" in value
    quality = value.get("quality")
    if quality is None:
        quality = "localized" if has_pose else "unknown"
    if quality not in LOCALIZATION_QUALITY:
        raise SpatialContractError(
            "runtime pose quality must be unknown, initializing, localized, "
            "degraded, or lost"
        )
    if quality == "unknown" and has_pose:
        raise SpatialContractError("unknown runtime pose cannot include coordinates")
    if quality in _POSE_QUALITY and not has_pose:
        raise SpatialContractError(
            f"{quality} runtime pose requires position and orientation"
        )
    estimate = _estimate(value) if has_pose else None
    timestamp = _timestamp(value.get("timestamp")) if "timestamp" in value else None
    source = _source(value.get("source")) if "source" in value else None
    if estimate is not None:
        if timestamp is None:
            raise SpatialContractError("runtime pose estimate requires timestamp")
        if source is None:
            raise SpatialContractError("runtime pose estimate requires source")
    uncertainty = None
    if "uncertainty" in value:
        uncertainty = _uncertainty(value["uncertainty"])
    return RuntimePose(
        agent=agent,
        space=space,
        quality=quality,
        estimate=estimate,
        timestamp=timestamp,
        source=source,
        uncertainty=uncertainty,
    )


def runtime_pose_as_mapping(value: RuntimePose) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "agent": value.agent,
        "space": value.space,
        "quality": value.quality,
    }
    if value.estimate is not None:
        payload["position"] = {
            "x": value.estimate.position.x,
            "y": value.estimate.position.y,
            "z": value.estimate.position.z,
        }
        payload["orientation"] = {
            "yaw": value.estimate.orientation.yaw,
            "pitch": value.estimate.orientation.pitch,
            "roll": value.estimate.orientation.roll,
        }
    if value.timestamp is not None:
        payload["timestamp"] = value.timestamp
    if value.source is not None:
        payload["source"] = value.source
    if value.uncertainty is not None:
        payload["uncertainty"] = {
            field: getattr(value.uncertainty, field)
            for field in _UNCERTAINTY_FIELDS
            if getattr(value.uncertainty, field) is not None
        }
    return payload


def measured_position(value: Any) -> Position:
    item = _object(value, "position", _POSITION_FIELDS)
    missing = sorted(_POSITION_FIELDS - set(item))
    if missing:
        raise SpatialContractError(
            f"position requires x, y, z; missing {', '.join(missing)}"
        )
    try:
        return Position(
            as_meters(item["x"]), as_meters(item["y"]), as_meters(item["z"])
        )
    except SpatialUnitError as error:
        raise SpatialContractError(str(error)) from error


def measured_orientation(value: Any) -> Orientation:
    item = _object(value, "orientation", _ORIENTATION_FIELDS)
    missing = sorted(_ORIENTATION_FIELDS - set(item))
    if missing:
        raise SpatialContractError(
            f"orientation requires yaw, pitch, roll; missing {', '.join(missing)}"
        )
    try:
        return Orientation(
            as_radians(item["yaw"]),
            as_radians(item["pitch"]),
            as_radians(item["roll"]),
        )
    except SpatialUnitError as error:
        raise SpatialContractError(str(error)) from error


def _estimate(value: Mapping[str, Any]) -> Pose:
    if "position" not in value or "orientation" not in value:
        raise SpatialContractError("runtime pose estimate requires position and orientation")
    position = measured_position(value["position"])
    orientation = measured_orientation(value["orientation"])
    return pose(
        x=position.x, y=position.y, z=position.z,
        yaw=orientation.yaw, pitch=orientation.pitch, roll=orientation.roll,
    )


def _uncertainty(value: Any) -> PoseUncertainty:
    item = _object(value, "uncertainty", frozenset(_UNCERTAINTY_FIELDS))
    fields: dict[str, float] = {}
    for name in _UNCERTAINTY_FIELDS:
        if name not in item:
            continue
        sigma = item[name]
        if type(sigma) not in {int, float} or not math.isfinite(sigma) or sigma < 0:
            raise SpatialContractError(
                f"uncertainty.{name} must be a finite non-negative number"
            )
        fields[name] = float(sigma)
    return PoseUncertainty(**fields)


def _timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpatialContractError("timestamp must be an ISO-8601 string")
    try:
        datetime.fromisoformat(value)
    except ValueError as error:
        raise SpatialContractError("timestamp must be an ISO-8601 string") from error
    return value


def _source(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpatialContractError("source must be a non-empty string")
    return value


def _typed_id(value: Any, field: str, table: str) -> str:
    if not isinstance(value, str):
        raise SpatialContractError(f"{field} must be a {table}: record ID")
    try:
        parsed, _ = split_record_id(value)
    except ValueError as error:
        raise SpatialContractError(f"{field} must be a {table}: record ID") from error
    if parsed != table:
        raise SpatialContractError(f"{field} must be a {table}: record ID")
    return value


def _object(value: Any, field: str, allowed: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpatialContractError(f"{field} must be an object")
    extra = sorted(set(value) - allowed)
    if extra:
        raise SpatialContractError(
            f"Unknown {field} fields: {', '.join(extra)}"
        )
    return value
