"""Canonical client-to-Home-Cortex physical estimate; no sensor processing."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .embodiment import BodyGeometry, LocalFrame, _finite, _object, _typed_id
from .primitives import SpatialContractError
from .transforms import (
    IDENTITY_BASIS, Pose, Position, canonical_basis, local_to_physical,
    physical_to_local, pose, transform_point,
)


@dataclass(frozen=True)
class PhysicalValue:
    """Nominal estimate and non-negative 95% absolute error bound, in field units."""

    value: float
    p95: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite(self.value, "physical value"))
        object.__setattr__(self, "p95", _finite(self.p95, "p95"))
        if self.p95 < 0:
            raise SpatialContractError("p95 must be non-negative")


def physical_value_from_mapping(value: Any) -> PhysicalValue:
    item = _object(value, "physical value", {"value", "p95"})
    return PhysicalValue(item["value"], item["p95"])


def physical_value_as_mapping(value: PhysicalValue) -> dict[str, float]:
    return {"value": value.value, "p95": value.p95}


@dataclass(frozen=True)
class RealtimeTransform:
    """T(space ← embodiment); translations in space coordinates, angles in radians."""

    x: PhysicalValue
    y: PhysicalValue
    z: PhysicalValue
    yaw: PhysicalValue
    pitch: PhysicalValue
    roll: PhysicalValue

    def __post_init__(self) -> None:
        for field in ("x", "y", "z", "yaw", "pitch", "roll"):
            if not isinstance(getattr(self, field), PhysicalValue):
                raise SpatialContractError(f"transform.{field} must be a PhysicalValue")

    def nominal_pose(self) -> Pose:
        """Existing ZYX pose math; does not claim to propagate uncertainty."""
        return pose(x=self.x.value, y=self.y.value, z=self.z.value,
                    yaw=self.yaw.value, pitch=self.pitch.value, roll=self.roll.value)


def realtime_transform_from_mapping(value: Any) -> RealtimeTransform:
    fields = {"x", "y", "z", "yaw", "pitch", "roll"}
    item = _object(value, "transform", fields)
    return RealtimeTransform(*(physical_value_from_mapping(item[field])
                               for field in ("x", "y", "z", "yaw", "pitch", "roll")))


def realtime_transform_as_mapping(value: RealtimeTransform) -> dict[str, dict[str, float]]:
    return {field: physical_value_as_mapping(getattr(value, field))
            for field in ("x", "y", "z", "yaw", "pitch", "roll")}


def _measured_at(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise SpatialContractError("measured_at must be an ISO-8601 timestamp with offset") from error
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise SpatialContractError("measured_at must be an ISO-8601 timestamp with offset")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class PhysicalTelemetry:
    """One measured estimate or explicit absence, scoped to one embodiment."""

    embodiment_id: str
    space_id: str | None
    measured_at: datetime
    validity: str
    transform: RealtimeTransform | None

    def __post_init__(self) -> None:
        _typed_id(self.embodiment_id, "embodiment", "embodiment_id")
        if self.space_id is not None:
            _typed_id(self.space_id, "space", "space_id")
        object.__setattr__(self, "measured_at", _measured_at(self.measured_at))
        if self.validity == "valid":
            if self.space_id is None or not isinstance(self.transform, RealtimeTransform):
                raise SpatialContractError("valid telemetry requires space_id and transform")
        elif self.validity == "no_estimate":
            if self.transform is not None:
                raise SpatialContractError("no_estimate telemetry cannot contain a transform")
        else:
            raise SpatialContractError("validity must be valid or no_estimate")


def physical_telemetry_from_mapping(value: Any) -> PhysicalTelemetry:
    item = _object(value, "telemetry", {"embodiment_id", "space_id", "measured_at", "validity", "transform"})
    transform = None if item["transform"] is None else realtime_transform_from_mapping(item["transform"])
    return PhysicalTelemetry(item["embodiment_id"], item["space_id"], item["measured_at"],
                             item["validity"], transform)


def physical_telemetry_as_mapping(value: PhysicalTelemetry) -> dict[str, Any]:
    return {"embodiment_id": value.embodiment_id, "space_id": value.space_id,
            "measured_at": value.measured_at.isoformat(), "validity": value.validity,
            "transform": realtime_transform_as_mapping(value.transform) if value.transform else None}


def estimate_within_limits(
    telemetry: PhysicalTelemetry, *, now: datetime, max_age_s: float,
    max_position_p95_m: float, max_orientation_p95_rad: float,
    space_basis: Mapping[str, list[float]] | None = None,
) -> bool:
    """Caller-supplied policy over validity, age, and physical SI uncertainty."""
    current = _measured_at(now)
    limits = (_finite(max_age_s, "max_age_s"),
              _finite(max_position_p95_m, "max_position_p95_m"),
              _finite(max_orientation_p95_rad, "max_orientation_p95_rad"))
    if any(limit < 0 for limit in limits):
        raise SpatialContractError("quality limits must be non-negative")
    if telemetry.validity != "valid" or telemetry.transform is None:
        return False
    age = (current - telemetry.measured_at).total_seconds()
    if age < 0 or age > max_age_s:
        return False
    t = telemetry.transform
    basis = canonical_basis(space_basis if space_basis is not None else IDENTITY_BASIS)
    _metric_axes(basis)
    return (all(getattr(t, field).p95 * math.sqrt(sum(component * component for component in basis[field]))
                <= max_position_p95_m for field in ("x", "y", "z"))
            and all(getattr(t, field).p95 <= max_orientation_p95_rad
                    for field in ("yaw", "pitch", "roll")))


def _metric_axes(space_basis: Mapping[str, list[float]]) -> dict[str, tuple[float, float, float]]:
    """Unit directions of a scaled/rotated space basis; shear has no rigid pose."""
    basis = canonical_basis(space_basis)
    axes = {}
    for name, vector in basis.items():
        magnitude = math.sqrt(sum(component * component for component in vector))
        axes[name] = tuple(component / magnitude for component in vector)
    for a, b in (("x", "y"), ("x", "z"), ("y", "z")):
        if abs(sum(x * y for x, y in zip(axes[a], axes[b]))) > 1e-9:
            raise SpatialContractError("rigid embodiment transform requires an orthogonal space basis")
    x, y, z = (axes[name] for name in ("x", "y", "z"))
    cross = (x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0])
    if any(abs(cross[i] - z[i]) > 1e-9 for i in range(3)):
        raise SpatialContractError("rigid embodiment transform requires a right-handed space basis")
    return axes


def embodiment_point_in_space(
    point: Position, transform: RealtimeTransform,
    *, space_basis: Mapping[str, list[float]] | None = None,
) -> Position:
    """Map a metric body point into space coordinates without scaling the body."""
    basis = canonical_basis(space_basis if space_basis is not None else IDENTITY_BASIS)
    axes = _metric_axes(basis)
    rotated = transform_point(point, pose(yaw=transform.yaw.value,
                                          pitch=transform.pitch.value,
                                          roll=transform.roll.value))
    displacement = tuple(sum(axes[name][i] * component for name, component in
                             zip(("x", "y", "z"), rotated.as_tuple())) for i in range(3))
    local = physical_to_local(displacement, basis)
    return Position(transform.x.value + local.x, transform.y.value + local.y,
                    transform.z.value + local.z)


def embodiment_point_in_physical(
    point: Position, transform: RealtimeTransform,
    *, space_basis: Mapping[str, list[float]] | None = None,
) -> Position:
    """Map a body point to physical SI displacement through the existing basis."""
    basis = canonical_basis(space_basis if space_basis is not None else IDENTITY_BASIS)
    return local_to_physical(embodiment_point_in_space(point, transform, space_basis=basis), basis)


def box_corners_in_space(
    geometry: BodyGeometry, frame: LocalFrame, transform: RealtimeTransform,
    *, space_basis: Mapping[str, list[float]] | None = None,
) -> tuple[Position, ...]:
    """Eight nominal box vertices, in deterministic forward/left/up sign order."""
    center = geometry.center.as_tuple()
    axes = (frame.axis("forward").as_tuple(), frame.axis("left").as_tuple(),
            frame.axis("up").as_tuple())
    halves = (geometry.length_m / 2, geometry.width_m / 2, geometry.height_m / 2)
    corners = []
    for forward in (-1, 1):
        for left in (-1, 1):
            for up in (-1, 1):
                signs = (forward, left, up)
                point = Position(*(center[i] + sum(signs[j] * halves[j] * axes[j][i]
                                                   for j in range(3)) for i in range(3)))
                corners.append(embodiment_point_in_space(point, transform, space_basis=space_basis))
    return tuple(corners)
