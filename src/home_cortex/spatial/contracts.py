"""Canonical SI space and placement contracts.

Coordinates belong to a space. Pose on ``located_in`` is interpreted in the
target space and is omitted from legacy edges. This module does not convert
nested frames or talk to ROS.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

LENGTH_UNIT = "m"
ANGLE_UNIT = "rad"
TIME_UNIT = "s"
SPACE_SPATIAL_FIELDS = frozenset({"accessible", "coordinate", "geometry", "navigable"})
POSE_FIELDS = frozenset({"orientation", "position"})
_LENGTH_ALIASES = {
    "m": LENGTH_UNIT,
    "meter": LENGTH_UNIT,
    "meters": LENGTH_UNIT,
    "metre": LENGTH_UNIT,
    "metres": LENGTH_UNIT,
}
_COORDINATE_FIELDS = frozenset({
    "anchors",
    "basis",
    "origin",
    "unit",
    "x_axis",
    "y_axis",
    "z_axis",
})
_GEOMETRY_FIELDS = frozenset({"size", "type"})
_POSITION_FIELDS = frozenset({"x", "y", "z"})
_ORIENTATION_FIELDS = frozenset({"pitch", "roll", "yaw"})
_UP = "up"
_BOX = "box"
_IDENTITY_BASIS = {
    "x": [1.0, 0.0, 0.0],
    "y": [0.0, 1.0, 0.0],
    "z": [0.0, 0.0, 1.0],
}


class SpatialContractError(ValueError):
    """A stored space or placement value is not a valid spatial contract."""


def apply_space_spatial_fields(record: dict[str, Any], *, source: Path | str) -> None:
    """Validate optional space spatial fields and canonicalize units in place."""
    if "coordinate" in record:
        record["coordinate"] = _canonical_coordinate(
            record["coordinate"], source, space_id=record.get("id")
        )
    if "geometry" in record:
        record["geometry"] = _canonical_geometry(record["geometry"], source)
    for field in ("navigable", "accessible"):
        if field in record and type(record[field]) is not bool:
            raise SpatialContractError(
                f"{field} in {_label(source)} must be a boolean"
            )


def reject_non_space_spatial_fields(
    record: Mapping[str, Any],
    table: str,
    *,
    source: Path | str,
) -> None:
    present = sorted(SPACE_SPATIAL_FIELDS.intersection(record))
    if present:
        raise SpatialContractError(
            f"{table} in {_label(source)} cannot define {', '.join(present)}; "
            "spatial metadata belongs to space"
        )


def located_in_endpoints_allowed(source_type: str, target_type: str) -> bool:
    """Item placement uses the edge schema; space→space is physical placement only.

    ``located_in.from_types`` stay item-only so semantic ``contents`` remain
    item-typed. Ingestion still stores a Space located in a Space.
    """
    if source_type == "item" and target_type in {"address", "space"}:
        return True
    return source_type == "space" and target_type == "space"


def apply_located_in_pose(
    record: dict[str, Any],
    *,
    target_type: str,
    source: Path | str,
) -> None:
    """Validate optional pose; coordinates require a space target."""
    if "frame_id" in record:
        raise SpatialContractError(
            f"located_in in {_label(source)} cannot define frame_id; "
            "the target space owns the coordinate system"
        )
    present = POSE_FIELDS.intersection(record)
    if not present:
        return
    if target_type != "space":
        raise SpatialContractError(
            f"located_in pose in {_label(source)} requires a space target, "
            f"not {target_type}"
        )
    if "position" in record:
        record["position"] = _canonical_named_vector(
            record["position"], _POSITION_FIELDS, "position", source
        )
    if "orientation" in record:
        record["orientation"] = _canonical_named_vector(
            record["orientation"], _ORIENTATION_FIELDS, "orientation", source
        )


def _canonical_coordinate(
    value: Any, source: Path | str, *, space_id: str | None = None
) -> dict[str, Any]:
    item = _object(value, "coordinate", source, _COORDINATE_FIELDS)
    if "origin" in item:
        _vector3(item["origin"], "coordinate.origin", source)
    canonical: dict[str, Any] = {
        "unit": _length_unit(item["unit"], source) if "unit" in item else LENGTH_UNIT,
        "basis": _canonical_basis(item, source),
    }
    if "anchors" in item:
        from .anchors import embedded_anchor_as_mapping, parse_embedded_anchors

        parsed = parse_embedded_anchors(item["anchors"], space=space_id)
        if parsed:
            canonical["anchors"] = [embedded_anchor_as_mapping(anchor) for anchor in parsed]
    return canonical


def _canonical_basis(item: Mapping[str, Any], source: Path | str) -> dict[str, list[float]]:
    from .transforms import SpatialTransformError, canonical_basis

    if "basis" in item:
        raw = item["basis"]
    elif any(name in item for name in ("x_axis", "y_axis", "z_axis")):
        raw = _basis_from_legacy_axes(item, source)
    else:
        return {name: list(vector) for name, vector in _IDENTITY_BASIS.items()}
    try:
        return canonical_basis(raw)
    except SpatialTransformError as error:
        raise SpatialContractError(
            f"coordinate.basis in {_label(source)}: {error}"
        ) from error


def _basis_from_legacy_axes(item: Mapping[str, Any], source: Path | str) -> dict[str, Any]:
    axes = {
        "x": item.get("x_axis", _IDENTITY_BASIS["x"]),
        "y": item.get("y_axis", _IDENTITY_BASIS["y"]),
        "z": item.get("z_axis", _IDENTITY_BASIS["z"]),
    }
    if axes["z"] == _UP:
        axes["z"] = list(_IDENTITY_BASIS["z"])
    for name, vector in axes.items():
        _axis_vector(vector, f"coordinate.{name}_axis", source)
    return axes


def _canonical_geometry(value: Any, source: Path | str) -> dict[str, Any]:
    item = _object(value, "geometry", source, _GEOMETRY_FIELDS)
    geometry_type = item.get("type")
    if geometry_type != _BOX:
        raise SpatialContractError(
            f"geometry.type in {_label(source)} must be {_BOX!r}"
        )
    size = _vector3(item.get("size"), "geometry.size", source)
    if any(component <= 0 for component in size):
        raise SpatialContractError(
            f"geometry.size in {_label(source)} must be positive"
        )
    return {"type": _BOX, "size": size}


def _canonical_named_vector(
    value: Any,
    required: frozenset[str],
    field: str,
    source: Path | str,
) -> dict[str, Any]:
    item = _object(value, field, source, required)
    missing = sorted(required - set(item))
    if missing:
        raise SpatialContractError(
            f"{field} in {_label(source)} requires {', '.join(sorted(required))}"
        )
    return {name: _finite_number(item[name], f"{field}.{name}", source) for name in sorted(required)}


def _length_unit(value: Any, source: Path | str) -> str:
    if not isinstance(value, str):
        raise SpatialContractError(
            f"coordinate.unit in {_label(source)} must be {LENGTH_UNIT!r}"
        )
    unit = _LENGTH_ALIASES.get(value.casefold())
    if unit is None:
        raise SpatialContractError(
            f"coordinate.unit in {_label(source)} must be {LENGTH_UNIT!r}"
        )
    return unit


def _axis_vector(value: Any, field: str, source: Path | str) -> list[float | int]:
    vector = _vector3(value, field, source)
    if all(component == 0 for component in vector):
        raise SpatialContractError(f"{field} in {_label(source)} cannot be a zero vector")
    return vector


def _vector3(value: Any, field: str, source: Path | str) -> list[float | int]:
    if not isinstance(value, list) or len(value) != 3:
        raise SpatialContractError(
            f"{field} in {_label(source)} must be a list of three finite numbers"
        )
    return [_finite_number(item, field, source) for item in value]


def _object(
    value: Any,
    field: str,
    source: Path | str,
    allowed: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpatialContractError(f"{field} in {_label(source)} must be an object")
    extra = sorted(set(value) - allowed)
    if extra:
        raise SpatialContractError(
            f"Unknown {field} fields in {_label(source)}: {', '.join(extra)}"
        )
    return value


def _finite_number(value: Any, field: str, source: Path | str) -> float | int:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise SpatialContractError(
            f"{field} in {_label(source)} must be a finite number"
        )
    return value


def _label(source: Path | str) -> str:
    return str(source)
