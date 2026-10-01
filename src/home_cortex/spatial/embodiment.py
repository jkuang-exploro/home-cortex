"""Persistent embodiment shape and its intrinsic, right-handed body frame."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..persistence.record_ids import split_record_id
from .primitives import SpatialContractError
from .transforms import Position

_AXES = {"+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
         "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
         "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0)}
_CAPABILITY_NAME = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")


def _object(value: Any, field: str, required: set[str], optional: set[str] = set()) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialContractError(f"{field} must be an object")
    missing = required - value.keys()
    extra = value.keys() - required - optional
    if missing or extra:
        raise SpatialContractError(f"{field} fields: missing {sorted(missing)}, unknown {sorted(extra)}")
    return value


def _finite(value: Any, field: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise SpatialContractError(f"{field} must be a finite number")
    if positive and value <= 0:
        raise SpatialContractError(f"{field} must be positive")
    return float(value)


def _typed_id(value: Any, table: str, field: str) -> str:
    try:
        parsed, _ = split_record_id(value)
    except ValueError as error:
        raise SpatialContractError(f"{field} must be a {table}: record ID") from error
    if parsed != table:
        raise SpatialContractError(f"{field} must be a {table}: record ID")
    return value


def _position(value: Any, field: str) -> Position:
    item = _object(value, field, {"x", "y", "z"})
    return Position(*(_finite(item[axis], f"{field}.{axis}") for axis in ("x", "y", "z")))


def _position_mapping(value: Position) -> dict[str, float]:
    return {"x": value.x, "y": value.y, "z": value.z}


@dataclass(frozen=True)
class LocalFrame:
    """Intrinsic axis labels; forward × left = up, with all axes distinct."""

    forward: str
    left: str
    up: str

    def __post_init__(self) -> None:
        if any(not isinstance(axis, str) or axis not in _AXES
               for axis in (self.forward, self.left, self.up)):
            raise SpatialContractError("local_frame directions must be signed x, y, or z axes")
        f, l, u = (_AXES[axis] for axis in (self.forward, self.left, self.up))
        cross = (f[1]*l[2]-f[2]*l[1], f[2]*l[0]-f[0]*l[2], f[0]*l[1]-f[1]*l[0])
        if cross != u:
            raise SpatialContractError("local_frame requires forward × left = up")

    def axis(self, direction: str) -> Position:
        if direction not in {"forward", "left", "up"}:
            raise SpatialContractError(f"unknown local direction {direction!r}")
        return Position(*_AXES[getattr(self, direction)])


def local_frame_from_mapping(value: Any) -> LocalFrame:
    item = _object(value, "local_frame", {"forward", "left", "up"})
    return LocalFrame(item["forward"], item["left"], item["up"])


def local_frame_as_mapping(value: LocalFrame) -> dict[str, str]:
    return {"forward": value.forward, "left": value.left, "up": value.up}


@dataclass(frozen=True)
class BodyGeometry:
    """Box in the embodiment frame: center offset and full dimensions in meters."""

    length_m: float
    width_m: float
    height_m: float
    center: Position

    def __post_init__(self) -> None:
        for field in ("length_m", "width_m", "height_m"):
            object.__setattr__(self, field, _finite(getattr(self, field), field, positive=True))
        if not isinstance(self.center, Position):
            raise SpatialContractError("body geometry center must be a Position")
        _position(_position_mapping(self.center), "geometry.box.center")


def body_geometry_from_mapping(value: Any) -> BodyGeometry:
    geometry = _object(value, "geometry", {"box"})
    box = _object(geometry["box"], "geometry.box", {"length_m", "width_m", "height_m", "center"})
    return BodyGeometry(box["length_m"], box["width_m"], box["height_m"],
                        _position(box["center"], "geometry.box.center"))


def body_geometry_as_mapping(value: BodyGeometry) -> dict[str, Any]:
    return {"box": {"length_m": value.length_m, "width_m": value.width_m,
                    "height_m": value.height_m, "center": _position_mapping(value.center)}}


@dataclass(frozen=True)
class Embodiment:
    """Durable body identity; reconnects refer to this ID rather than creating one."""

    id: str
    name: str
    geometry: BodyGeometry | None = None
    local_frame: LocalFrame | None = None
    agent_id: str | None = None
    capabilities: tuple[str, ...] = ()
    embodiment_type: str = "unspecified"

    def __post_init__(self) -> None:
        _typed_id(self.id, "embodiment", "embodiment.id")
        if not isinstance(self.name, str) or not self.name.strip():
            raise SpatialContractError("embodiment.name must be non-empty")
        if (not isinstance(self.embodiment_type, str) or
                re.fullmatch(r"[a-z][a-z0-9_]*", self.embodiment_type) is None):
            raise SpatialContractError("embodiment.embodiment_type must be a type name")
        if self.geometry is not None and not isinstance(self.geometry, BodyGeometry):
            raise SpatialContractError("embodiment.geometry must be a BodyGeometry")
        if self.local_frame is not None and not isinstance(self.local_frame, LocalFrame):
            raise SpatialContractError("embodiment.local_frame must be a LocalFrame")
        if self.agent_id is not None:
            _typed_id(self.agent_id, "agent", "embodiment.agent_id")
        if not isinstance(self.capabilities, tuple) or any(
            not isinstance(name, str) or _CAPABILITY_NAME.fullmatch(name) is None
            for name in self.capabilities
        ):
            raise SpatialContractError("embodiment capabilities must be namespaced strings")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise SpatialContractError("embodiment capabilities must be unique")
        object.__setattr__(self, "capabilities", tuple(sorted(self.capabilities)))


def embodiment_from_mapping(value: Any) -> Embodiment:
    item = _object(value, "embodiment", {"id", "name"},
                   {"geometry", "local_frame", "agent_id", "capabilities", "embodiment_type"})
    capabilities = item.get("capabilities", [])
    if not isinstance(capabilities, list):
        raise SpatialContractError("embodiment capabilities must be a list")
    return Embodiment(item["id"], item["name"],
                      body_geometry_from_mapping(item["geometry"]) if "geometry" in item else None,
                      local_frame_from_mapping(item["local_frame"]) if "local_frame" in item else None,
                      item.get("agent_id"),
                      tuple(capabilities), item.get("embodiment_type", "unspecified"))


def embodiment_as_mapping(value: Embodiment) -> dict[str, Any]:
    result = {"id": value.id, "name": value.name, "embodiment_type": value.embodiment_type}
    if value.geometry is not None:
        result["geometry"] = body_geometry_as_mapping(value.geometry)
    if value.local_frame is not None:
        result["local_frame"] = local_frame_as_mapping(value.local_frame)
    if value.agent_id is not None:
        result["agent_id"] = value.agent_id
    result["capabilities"] = list(value.capabilities)
    return result
