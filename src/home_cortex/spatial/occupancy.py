"""Deterministic nominal rigid-box occupancy from canonical embodiment telemetry."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .embodiment import BodyGeometry, LocalFrame
from .primitives import SpatialContractError
from .telemetry import RealtimeTransform, box_corners_in_space
from .transforms import Position


@dataclass(frozen=True)
class AxisAlignedEnvelope:
    """Min/max coordinates in one named space's local frame."""

    minimum: Position
    maximum: Position

    def __post_init__(self) -> None:
        if any(lo > hi for lo, hi in zip(self.minimum.as_tuple(), self.maximum.as_tuple())):
            raise SpatialContractError("envelope minimum must not exceed maximum")

    def as_mapping(self) -> dict[str, dict[str, float]]:
        return {
            "minimum": dict(zip(("x", "y", "z"), self.minimum.as_tuple())),
            "maximum": dict(zip(("x", "y", "z"), self.maximum.as_tuple())),
        }


def envelope_of_points(points: Sequence[Position]) -> AxisAlignedEnvelope:
    if not points:
        raise SpatialContractError("an envelope requires at least one point")
    coordinates = tuple(point.as_tuple() for point in points)
    return AxisAlignedEnvelope(
        Position(*(min(point[axis] for point in coordinates) for axis in range(3))),
        Position(*(max(point[axis] for point in coordinates) for axis in range(3))),
    )


def nominal_box_envelope(
    geometry: BodyGeometry,
    frame: LocalFrame,
    transform: RealtimeTransform,
    *,
    space_basis: Mapping[str, list[float]] | None = None,
) -> AxisAlignedEnvelope:
    """Axis-aligned envelope of the eight nominal oriented box vertices."""
    return envelope_of_points(
        box_corners_in_space(geometry, frame, transform, space_basis=space_basis)
    )


def translation_p95_envelope(
    nominal: AxisAlignedEnvelope, transform: RealtimeTransform,
) -> AxisAlignedEnvelope:
    """Inflate by per-axis translation p95 only; orientation p95 stays separate.

    This is not a joint 95% occupancy claim. It intentionally leaves the
    physical box size and localization uncertainty as distinct inputs.
    """
    deltas = (transform.x.p95, transform.y.p95, transform.z.p95)
    return AxisAlignedEnvelope(
        Position(*(value - delta for value, delta in zip(nominal.minimum.as_tuple(), deltas))),
        Position(*(value + delta for value, delta in zip(nominal.maximum.as_tuple(), deltas))),
    )
