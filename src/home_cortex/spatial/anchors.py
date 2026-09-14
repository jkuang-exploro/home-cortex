"""Optional surveyed localization anchors. Never required by a space."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Mapping

from ..record_ids import split_record_id
from .contracts import SpatialContractError
from .pose import measured_orientation, measured_position
from .transforms import Orientation, Position
from .units import SpatialUnitError, as_meters

ANCHOR_KINDS = ("fiducial", "natural_landmark", "dock", "survey_point")
_ANCHOR_FIELDS = frozenset({
    "id",
    "kind",
    "orientation",
    "position",
    "recognition",
    "space",
    "survey",
    "type",
})
_RECOGNITION_FIELDS = frozenset({"family", "marker_id"})
_SURVEY_FIELDS = frozenset({"method", "uncertainty_m"})


@dataclass(frozen=True)
class AnchorRecognition:
    family: str
    marker_id: str | int | None = None


@dataclass(frozen=True)
class AnchorSurvey:
    method: str
    uncertainty_m: float | None = None


@dataclass(frozen=True)
class SurveyedAnchor:
    """Tape-measured reference in a space. Deleting all anchors leaves the space valid."""

    id: str
    space: str
    position: Position
    kind: str = "survey_point"
    orientation: Orientation | None = None
    recognition: AnchorRecognition | None = None
    survey: AnchorSurvey | None = None


def parse_surveyed_anchor(value: Mapping[str, Any]) -> SurveyedAnchor:
    if not isinstance(value, Mapping):
        raise SpatialContractError("surveyed anchor must be an object")
    extra = sorted(set(value) - _ANCHOR_FIELDS)
    if extra:
        raise SpatialContractError(
            f"Unknown surveyed anchor fields: {', '.join(extra)}"
        )
    if "type" in value and "kind" in value and value["type"] != value["kind"]:
        raise SpatialContractError("surveyed anchor type and kind must match")
    kind = value.get("kind", value.get("type", "survey_point"))
    if kind not in ANCHOR_KINDS:
        raise SpatialContractError(
            "surveyed anchor type must be fiducial, natural_landmark, dock, "
            "or survey_point"
        )
    if "position" not in value:
        raise SpatialContractError("surveyed anchor requires position")
    return SurveyedAnchor(
        id=_typed_id(value.get("id"), "id", "anchor"),
        space=_typed_id(value.get("space"), "space", "space"),
        position=measured_position(value["position"]),
        kind=kind,
        orientation=(
            measured_orientation(value["orientation"])
            if "orientation" in value
            else None
        ),
        recognition=(
            _recognition(value["recognition"]) if "recognition" in value else None
        ),
        survey=_survey(value["survey"]) if "survey" in value else None,
    )


def parse_surveyed_anchors(
    values: Sequence[Mapping[str, Any]] | None,
) -> tuple[SurveyedAnchor, ...]:
    """Empty or omitted collections are valid; spaces do not depend on anchors."""
    if values is None:
        return ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise SpatialContractError("surveyed anchors must be a list")
    anchors = tuple(parse_surveyed_anchor(item) for item in values)
    seen: set[str] = set()
    for anchor in anchors:
        if anchor.id in seen:
            raise SpatialContractError(f"Duplicate surveyed anchor ID {anchor.id}")
        seen.add(anchor.id)
    return anchors


def surveyed_anchor_as_mapping(value: SurveyedAnchor) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": value.id,
        "space": value.space,
        "type": value.kind,
        "position": {
            "x": value.position.x,
            "y": value.position.y,
            "z": value.position.z,
        },
    }
    if value.orientation is not None:
        payload["orientation"] = {
            "yaw": value.orientation.yaw,
            "pitch": value.orientation.pitch,
            "roll": value.orientation.roll,
        }
    if value.recognition is not None:
        recognition: dict[str, Any] = {"family": value.recognition.family}
        if value.recognition.marker_id is not None:
            recognition["marker_id"] = value.recognition.marker_id
        payload["recognition"] = recognition
    if value.survey is not None:
        survey: dict[str, Any] = {"method": value.survey.method}
        if value.survey.uncertainty_m is not None:
            survey["uncertainty_m"] = value.survey.uncertainty_m
        payload["survey"] = survey
    return payload


def _recognition(value: Any) -> AnchorRecognition:
    item = _object(value, "recognition", _RECOGNITION_FIELDS)
    family = item.get("family")
    if not isinstance(family, str) or not family.strip():
        raise SpatialContractError("recognition.family must be a non-empty string")
    marker_id = item.get("marker_id")
    if marker_id is not None and type(marker_id) not in {int, str}:
        raise SpatialContractError("recognition.marker_id must be a string or integer")
    if type(marker_id) is int and marker_id < 0:
        raise SpatialContractError("recognition.marker_id must be non-negative")
    return AnchorRecognition(family=family, marker_id=marker_id)


def _survey(value: Any) -> AnchorSurvey:
    item = _object(value, "survey", _SURVEY_FIELDS)
    method = item.get("method")
    if not isinstance(method, str) or not method.strip():
        raise SpatialContractError("survey.method must be a non-empty string")
    uncertainty = None
    if "uncertainty_m" in item:
        try:
            uncertainty = as_meters(item["uncertainty_m"])
        except SpatialUnitError as error:
            raise SpatialContractError(str(error)) from error
        if not math.isfinite(uncertainty) or uncertainty < 0:
            raise SpatialContractError(
                "survey.uncertainty_m must be a finite non-negative length"
            )
    return AnchorSurvey(method=method, uncertainty_m=uncertainty)


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
