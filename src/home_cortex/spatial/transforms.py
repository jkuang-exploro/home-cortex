"""Pure pose composition for nested spaces. No graph traversal or ROS TF."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import POSE_FIELDS

# ZYX intrinsic (yaw about +z, then pitch about +y, then roll about +x), z-up.
_GIMBAL = 1e-9
_ORIENTATION_FIELDS = ("yaw", "pitch", "roll")
_POSITION_FIELDS = ("x", "y", "z")


class SpatialTransformError(ValueError):
    """A pose chain or transform cannot be applied."""


@dataclass(frozen=True)
class Position:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class Orientation:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass(frozen=True)
class Pose:
    """Child-in-parent placement: position in meters, orientation in radians."""

    position: Position = Position()
    orientation: Orientation = Orientation()


def pose(
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
) -> Pose:
    return Pose(
        Position(_finite(x, "x"), _finite(y, "y"), _finite(z, "z")),
        Orientation(
            _finite(yaw, "yaw"), _finite(pitch, "pitch"), _finite(roll, "roll")
        ),
    )


def pose_from_mapping(value: Mapping[str, Any]) -> Pose:
    extra = sorted(set(value) - POSE_FIELDS)
    if extra:
        raise SpatialTransformError(
            f"Unknown pose fields: {', '.join(extra)}"
        )
    position = value.get("position", {})
    orientation = value.get("orientation", {})
    if not isinstance(position, Mapping) or not isinstance(orientation, Mapping):
        raise SpatialTransformError("pose position and orientation must be objects")
    for field, allowed in (
        ("position", _POSITION_FIELDS),
        ("orientation", _ORIENTATION_FIELDS),
    ):
        extra_fields = sorted(set(value.get(field, {})) - set(allowed))
        if extra_fields:
            raise SpatialTransformError(
                f"Unknown {field} fields: {', '.join(extra_fields)}"
            )
    return pose(
        x=position.get("x", 0.0),
        y=position.get("y", 0.0),
        z=position.get("z", 0.0),
        yaw=orientation.get("yaw", 0.0),
        pitch=orientation.get("pitch", 0.0),
        roll=orientation.get("roll", 0.0),
    )


def pose_as_mapping(value: Pose) -> dict[str, Any]:
    return {
        "position": {"x": value.position.x, "y": value.position.y, "z": value.position.z},
        "orientation": {
            "yaw": value.orientation.yaw,
            "pitch": value.orientation.pitch,
            "roll": value.orientation.roll,
        },
    }


def compose_pose(parent: Pose, child: Pose) -> Pose:
    """Express ``child`` (in the parent frame) in the same frame as ``parent``."""
    rotation = _matmul(_rotation(parent), _rotation(child))
    translation = _add(
        _matvec(_rotation(parent), child.position.as_tuple()),
        parent.position.as_tuple(),
    )
    return _pose_from_rt(rotation, translation)


def invert_pose(value: Pose) -> Pose:
    rotation = _transpose(_rotation(value))
    translation = _scale(_matvec(rotation, value.position.as_tuple()), -1.0)
    return _pose_from_rt(rotation, translation)


def transform_point(point: Position | Mapping[str, Any] | Sequence[Any], frame: Pose) -> Position:
    """Map a point from ``frame``-local coordinates into the parent frame."""
    local = _as_position(point).as_tuple()
    world = _add(_matvec(_rotation(frame), local), frame.position.as_tuple())
    return Position(*world)


def transform_pose(local: Pose, frame: Pose) -> Pose:
    """Map a pose from ``frame``-local coordinates into the parent frame."""
    return compose_pose(frame, local)


def compose_chain(poses: Sequence[Pose | None]) -> Pose:
    """Compose ancestor-to-descendant child-in-parent poses.

    An empty chain is identity. A missing link is a disconnected transform.
    """
    result = Pose()
    for index, item in enumerate(poses):
        if item is None:
            raise SpatialTransformError(
                f"disconnected transform at step {index}"
            )
        result = compose_pose(result, item)
    return result


def _as_position(value: Position | Mapping[str, Any] | Sequence[Any]) -> Position:
    if isinstance(value, Position):
        return value
    if isinstance(value, Mapping):
        extra = sorted(set(value) - set(_POSITION_FIELDS))
        if extra:
            raise SpatialTransformError(
                f"Unknown position fields: {', '.join(extra)}"
            )
        return Position(
            _finite(value.get("x", 0.0), "x"),
            _finite(value.get("y", 0.0), "y"),
            _finite(value.get("z", 0.0), "z"),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) not in {2, 3}:
            raise SpatialTransformError("point must have 2 or 3 coordinates")
        coords = list(value) + [0.0] * (3 - len(value))
        return Position(
            _finite(coords[0], "x"),
            _finite(coords[1], "y"),
            _finite(coords[2], "z"),
        )
    raise SpatialTransformError("point must be a position, mapping, or sequence")


def _rotation(value: Pose) -> tuple[tuple[float, float, float], ...]:
    yaw, pitch, roll = (
        value.orientation.yaw,
        value.orientation.pitch,
        value.orientation.roll,
    )
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _pose_from_rt(
    rotation: tuple[tuple[float, float, float], ...],
    translation: tuple[float, float, float],
) -> Pose:
    pitch = math.asin(max(-1.0, min(1.0, -rotation[2][0])))
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) < _GIMBAL:
        yaw = math.atan2(-rotation[0][1], rotation[1][1])
        roll = 0.0
    else:
        yaw = math.atan2(rotation[1][0], rotation[0][0])
        roll = math.atan2(rotation[2][1], rotation[2][2])
    return Pose(Position(*translation), Orientation(yaw, pitch, roll))


def _matmul(
    left: tuple[tuple[float, float, float], ...],
    right: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _matvec(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(sum(matrix[i][j] * vector[j] for j in range(3)) for i in range(3))


def _transpose(
    matrix: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(matrix[j][i] for j in range(3)) for i in range(3))


def _add(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _scale(vector: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _finite(value: Any, name: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise SpatialTransformError(f"{name} must be a finite number")
    return float(value)
