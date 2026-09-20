"""Structured SI normalization and display conversion. No language parsing."""
from __future__ import annotations

import math
from typing import Any, Mapping

from .primitives import ANGLE_UNIT, LENGTH_UNIT

# Exact International Yard and Pound definitions.
_INCH_METERS = 0.0254
_FOOT_METERS = 0.3048
_LENGTH_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "millimetre": 0.001,
    "millimetres": 0.001,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "centimetre": 0.01,
    "centimetres": 0.01,
    "in": _INCH_METERS,
    "inch": _INCH_METERS,
    "inches": _INCH_METERS,
    "ft": _FOOT_METERS,
    "foot": _FOOT_METERS,
    "feet": _FOOT_METERS,
}
_DISPLAY_LENGTH = {"mm": "mm", "cm": "cm", "m": "m", "in": "in", "ft": "ft"}
_DISPLAY_DECIMALS = {"mm": 3, "cm": 4, "m": 6, "in": 2, "ft": 4}
_ANGLE_TO_RADIANS = {
    "rad": 1.0,
    "radian": 1.0,
    "radians": 1.0,
    ANGLE_UNIT: 1.0,
    "deg": math.pi / 180.0,
    "degree": math.pi / 180.0,
    "degrees": math.pi / 180.0,
}
_DISPLAY_ANGLE = {"rad": "rad", "deg": "deg"}
_IMPERIAL_INCH_DECIMALS = 2


class SpatialUnitError(ValueError):
    """A structured measurement cannot be normalized or formatted."""


def normalize_length(value: Any, unit: str | None = None) -> float:
    """Return meters from a structured length. Does not parse prose."""
    if isinstance(value, Mapping) and unit is None:
        return _normalize_feet_inches(value)
    meters = _finite(value, "length") * _length_factor(unit)
    return meters


def normalize_angle(value: Any, unit: str = ANGLE_UNIT) -> float:
    """Return radians from a structured angle. Does not parse prose."""
    if not isinstance(unit, str):
        raise SpatialUnitError("angle unit must be a string")
    factor = _ANGLE_TO_RADIANS.get(unit.casefold())
    if factor is None:
        raise SpatialUnitError(
            f"Unsupported angle unit {unit!r}; expected rad or deg"
        )
    return _finite(value, "angle") * factor


def as_meters(value: Any, *, default_unit: str = LENGTH_UNIT) -> float:
    """Meters from a number in ``default_unit`` or a ``{value, unit}`` object."""
    if isinstance(value, Mapping) and ("value" in value or "unit" in value):
        extra = sorted(set(value) - {"unit", "value"})
        if extra:
            raise SpatialUnitError(
                f"Unknown structured length fields: {', '.join(extra)}"
            )
        if "value" not in value:
            raise SpatialUnitError("structured length requires value")
        return normalize_length(value["value"], value.get("unit", default_unit))
    if isinstance(value, Mapping):
        return normalize_length(value)
    return normalize_length(value, default_unit)


def as_radians(value: Any, *, default_unit: str = ANGLE_UNIT) -> float:
    """Radians from a number in ``default_unit`` or a ``{value, unit}`` object."""
    if isinstance(value, Mapping):
        extra = sorted(set(value) - {"unit", "value"})
        if extra:
            raise SpatialUnitError(
                f"Unknown structured angle fields: {', '.join(extra)}"
            )
        if "value" not in value:
            raise SpatialUnitError("structured angle requires value")
        return normalize_angle(value["value"], value.get("unit", default_unit))
    return normalize_angle(value, default_unit)


def format_length(
    meters: Any,
    *,
    unit: str | None = None,
    unit_system: str | None = None,
) -> str:
    """Render meters for a later display layer. Does not alter stored SI values."""
    length = _finite(meters, "length")
    if unit_system is not None:
        if unit is not None:
            raise SpatialUnitError("Pass unit or unit_system, not both")
        system = unit_system.casefold()
        if system == "metric":
            unit = LENGTH_UNIT
        elif system == "imperial":
            return _format_feet_inches(length)
        else:
            raise SpatialUnitError(
                f"Unsupported unit_system {unit_system!r}; expected metric or imperial"
            )
    if unit is None:
        unit = LENGTH_UNIT
    key = unit.casefold()
    if key not in _DISPLAY_LENGTH:
        raise SpatialUnitError(
            f"Unsupported length unit {unit!r}; expected mm, cm, m, in, or ft"
        )
    display = _DISPLAY_LENGTH[key]
    amount = length / _LENGTH_TO_METERS[display]
    return f"{_format_number(amount, _DISPLAY_DECIMALS[display])} {display}"


def format_angle(radians: Any, *, unit: str = "deg") -> str:
    """Render radians for a later display layer. Does not alter stored SI values."""
    angle = _finite(radians, "angle")
    if not isinstance(unit, str):
        raise SpatialUnitError("angle unit must be a string")
    key = unit.casefold()
    if key in {"rad", "radian", "radians", ANGLE_UNIT}:
        return f"{_format_number(angle)} rad"
    if key in {"deg", "degree", "degrees"}:
        return f"{_format_number(angle * 180.0 / math.pi)} deg"
    raise SpatialUnitError(f"Unsupported angle unit {unit!r}; expected rad or deg")


def _normalize_feet_inches(parts: Mapping[str, Any]) -> float:
    allowed = {"ft", "foot", "feet", "in", "inch", "inches"}
    extra = sorted(set(parts) - allowed)
    if extra:
        raise SpatialUnitError(
            f"Unknown imperial fields: {', '.join(extra)}; expected ft and in"
        )
    feet = parts.get("ft", parts.get("foot", parts.get("feet", 0)))
    inches = parts.get("in", parts.get("inch", parts.get("inches", 0)))
    return _finite(feet, "ft") * _FOOT_METERS + _finite(inches, "in") * _INCH_METERS


def _format_feet_inches(meters: float) -> str:
    inches = meters / _INCH_METERS
    sign = "-" if inches < 0 else ""
    inches = abs(inches)
    rounded = round(inches, _IMPERIAL_INCH_DECIMALS)
    feet, rem = divmod(rounded, 12.0)
    feet_i = int(feet)
    if feet_i and rem:
        return f"{sign}{feet_i} ft {_format_number(rem)} in"
    if feet_i:
        return f"{sign}{feet_i} ft"
    return f"{sign}{_format_number(rem)} in"


def _length_factor(unit: str | None) -> float:
    if unit is None:
        raise SpatialUnitError("length unit is required")
    if not isinstance(unit, str):
        raise SpatialUnitError("length unit must be a string")
    factor = _LENGTH_TO_METERS.get(unit.casefold())
    if factor is None:
        raise SpatialUnitError(
            f"Unsupported length unit {unit!r}; expected mm, cm, m, in, or ft"
        )
    return factor


def _finite(value: Any, name: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise SpatialUnitError(f"{name} must be a finite number")
    return float(value)


def _format_number(value: float, decimals: int = 6) -> str:
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return text or "0"
