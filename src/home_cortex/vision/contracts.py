"""Visual evidence contracts. Detector-native types do not cross this boundary.

A VisualObservation is evidence only. It does not create items or write
``located_in``. Bounding boxes are normalized image coordinates in [0, 1].
``observer`` may be null; Epic 2 does not depend on Epic 1 spatial pose.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from ..record_ids import split_record_id

CLIP_STATUSES = ("pending", "available", "failed")
_OBSERVATION_FIELDS = frozenset({
    "detection",
    "evidence",
    "id",
    "identity",
    "observed_at",
    "observer",
    "producer",
    "source",
})
_SOURCE_FIELDS = frozenset({"camera_id", "device_id"})
_DETECTION_FIELDS = frozenset({"bbox", "category", "confidence"})
_BBOX_FIELDS = frozenset({"x_max", "x_min", "y_max", "y_min"})
_IDENTITY_FIELDS = frozenset({"candidate_item_ids", "canonical_item_id"})
_EVIDENCE_FIELDS = frozenset({"clip_id", "crop_ref"})
_PRODUCER_FIELDS = frozenset({
    "detector",
    "detector_version",
    "pipeline",
    "pipeline_version",
})
_CLIP_FIELDS = frozenset({
    "artifact_ref",
    "end_time",
    "id",
    "start_time",
    "status",
})
_ARTIFACT_REF = re.compile(r"^sha256:[0-9a-f]{64}(?:\.[A-Za-z0-9]+)?$")


class VisionContractError(ValueError):
    """A visual observation or clip metadata value is invalid."""


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in normalized image coordinates, origin at top-left."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class ObservationSource:
    device_id: str
    camera_id: str


@dataclass(frozen=True)
class Detection:
    category: str
    confidence: float
    bbox: BoundingBox


@dataclass(frozen=True)
class ObservationIdentity:
    canonical_item_id: str | None = None
    candidate_item_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservationEvidence:
    crop_ref: str | None = None
    clip_id: str | None = None


@dataclass(frozen=True)
class ObservationProducer:
    pipeline: str
    pipeline_version: str
    detector: str | None = None
    detector_version: str | None = None


@dataclass(frozen=True)
class VisualObservation:
    """One structured visual observation. Not a household mutation."""

    id: str
    observed_at: str
    source: ObservationSource
    detection: Detection
    identity: ObservationIdentity
    producer: ObservationProducer
    observer: Mapping[str, Any] | None = None
    evidence: ObservationEvidence = ObservationEvidence()


@dataclass(frozen=True)
class EvidenceClip:
    """Metadata for media associated with an observation. Bytes live in the artifact store."""

    id: str
    status: str
    start_time: str
    end_time: str | None = None
    artifact_ref: str | None = None


def parse_visual_observation(value: Mapping[str, Any]) -> VisualObservation:
    item = _object(value, "visual observation", _OBSERVATION_FIELDS)
    if "observer" in item and item["observer"] is not None and not isinstance(item["observer"], Mapping):
        raise VisionContractError("observer must be an object or null")
    return VisualObservation(
        id=_typed_id(item.get("id"), "id", "observation"),
        observed_at=_timestamp(item.get("observed_at"), "observed_at"),
        source=_source(item.get("source")),
        detection=_detection(item.get("detection")),
        identity=_identity(item.get("identity")) if "identity" in item else ObservationIdentity(),
        producer=_producer(item.get("producer")),
        observer=None if item.get("observer") is None else dict(item["observer"]),
        evidence=_evidence(item.get("evidence")) if "evidence" in item else ObservationEvidence(),
    )


def visual_observation_as_mapping(value: VisualObservation) -> dict[str, Any]:
    return {
        "id": value.id,
        "observed_at": value.observed_at,
        "source": {
            "device_id": value.source.device_id,
            "camera_id": value.source.camera_id,
        },
        "detection": {
            "category": value.detection.category,
            "confidence": value.detection.confidence,
            "bbox": {
                "x_min": value.detection.bbox.x_min,
                "y_min": value.detection.bbox.y_min,
                "x_max": value.detection.bbox.x_max,
                "y_max": value.detection.bbox.y_max,
            },
        },
        "identity": {
            "canonical_item_id": value.identity.canonical_item_id,
            "candidate_item_ids": list(value.identity.candidate_item_ids),
        },
        "observer": None if value.observer is None else dict(value.observer),
        "evidence": {
            "crop_ref": value.evidence.crop_ref,
            "clip_id": value.evidence.clip_id,
        },
        "producer": {
            "pipeline": value.producer.pipeline,
            "pipeline_version": value.producer.pipeline_version,
            "detector": value.producer.detector,
            "detector_version": value.producer.detector_version,
        },
    }


def parse_evidence_clip(value: Mapping[str, Any]) -> EvidenceClip:
    item = _object(value, "evidence clip", _CLIP_FIELDS)
    status = item.get("status")
    if status not in CLIP_STATUSES:
        raise VisionContractError("clip status must be pending, available, or failed")
    artifact_ref = item.get("artifact_ref")
    if artifact_ref is not None:
        artifact_ref = _artifact_ref(artifact_ref)
    if status == "available" and artifact_ref is None:
        raise VisionContractError("available clip requires artifact_ref")
    start = _timestamp(item.get("start_time"), "start_time")
    end = None if item.get("end_time") is None else _timestamp(item.get("end_time"), "end_time")
    if end is not None and datetime.fromisoformat(end) < datetime.fromisoformat(start):
        raise VisionContractError("clip end_time must not precede start_time")
    return EvidenceClip(
        id=_typed_id(item.get("id"), "id", "clip"),
        status=status,
        start_time=start,
        end_time=end,
        artifact_ref=artifact_ref,
    )


def evidence_clip_as_mapping(value: EvidenceClip) -> dict[str, Any]:
    return {
        "id": value.id,
        "status": value.status,
        "start_time": value.start_time,
        "end_time": value.end_time,
        "artifact_ref": value.artifact_ref,
    }


def dumps_observation(value: VisualObservation) -> str:
    return json.dumps(
        visual_observation_as_mapping(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def parse_observations_jsonl(text: str) -> tuple[VisualObservation, ...]:
    lines = [line for line in text.splitlines() if line.strip()]
    return tuple(parse_visual_observation(json.loads(line)) for line in lines)


def dumps_observations_jsonl(values: Iterable[VisualObservation]) -> str:
    return "".join(dumps_observation(item) + "\n" for item in values)


def _source(value: Any) -> ObservationSource:
    item = _object(value, "source", _SOURCE_FIELDS)
    return ObservationSource(
        device_id=_typed_id(item.get("device_id"), "source.device_id", "device"),
        camera_id=_typed_id(item.get("camera_id"), "source.camera_id", "camera"),
    )


def _detection(value: Any) -> Detection:
    item = _object(value, "detection", _DETECTION_FIELDS)
    category = item.get("category")
    if not isinstance(category, str) or not category.strip():
        raise VisionContractError("detection.category must be a non-empty string")
    return Detection(
        category=category.strip(),
        confidence=_unit_interval(item.get("confidence"), "detection.confidence"),
        bbox=_bbox(item.get("bbox")),
    )


def _bbox(value: Any) -> BoundingBox:
    item = _object(value, "bbox", _BBOX_FIELDS)
    box = BoundingBox(
        x_min=_unit_interval(item.get("x_min"), "bbox.x_min"),
        y_min=_unit_interval(item.get("y_min"), "bbox.y_min"),
        x_max=_unit_interval(item.get("x_max"), "bbox.x_max"),
        y_max=_unit_interval(item.get("y_max"), "bbox.y_max"),
    )
    if not (box.x_min < box.x_max and box.y_min < box.y_max):
        raise VisionContractError("bbox requires 0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1")
    return box


def _identity(value: Any) -> ObservationIdentity:
    if value is None:
        return ObservationIdentity()
    item = _object(value, "identity", _IDENTITY_FIELDS)
    canonical = item.get("canonical_item_id")
    if canonical is not None:
        canonical = _typed_id(canonical, "identity.canonical_item_id", "item")
    raw_candidates = item.get("candidate_item_ids", [])
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
        raise VisionContractError("identity.candidate_item_ids must be a list")
    candidates = tuple(
        _typed_id(candidate, "identity.candidate_item_ids", "item")
        for candidate in raw_candidates
    )
    return ObservationIdentity(canonical_item_id=canonical, candidate_item_ids=candidates)


def _evidence(value: Any) -> ObservationEvidence:
    if value is None:
        return ObservationEvidence()
    item = _object(value, "evidence", _EVIDENCE_FIELDS)
    crop_ref = item.get("crop_ref")
    clip_id = item.get("clip_id")
    return ObservationEvidence(
        crop_ref=None if crop_ref is None else _artifact_ref(crop_ref),
        clip_id=None if clip_id is None else _typed_id(clip_id, "evidence.clip_id", "clip"),
    )


def _producer(value: Any) -> ObservationProducer:
    item = _object(value, "producer", _PRODUCER_FIELDS)
    pipeline = item.get("pipeline")
    version = item.get("pipeline_version")
    if not isinstance(pipeline, str) or not pipeline.strip():
        raise VisionContractError("producer.pipeline must be a non-empty string")
    if not isinstance(version, str) or not version.strip():
        raise VisionContractError("producer.pipeline_version must be a non-empty string")
    detector = item.get("detector")
    detector_version = item.get("detector_version")
    if detector is not None and (not isinstance(detector, str) or not detector.strip()):
        raise VisionContractError("producer.detector must be a string or null")
    if detector_version is not None and (
        not isinstance(detector_version, str) or not detector_version.strip()
    ):
        raise VisionContractError("producer.detector_version must be a string or null")
    return ObservationProducer(
        pipeline=pipeline.strip(),
        pipeline_version=version.strip(),
        detector=None if detector is None else detector.strip(),
        detector_version=None if detector_version is None else detector_version.strip(),
    )


def _typed_id(value: Any, field: str, table: str) -> str:
    if not isinstance(value, str):
        raise VisionContractError(f"{field} must be a {table}: record ID")
    try:
        parsed, _ = split_record_id(value)
    except ValueError as error:
        raise VisionContractError(f"{field} must be a {table}: record ID") from error
    if parsed != table:
        raise VisionContractError(f"{field} must be a {table}: record ID")
    return value


def _timestamp(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VisionContractError(f"{field} must be an ISO-8601 string")
    try:
        datetime.fromisoformat(value)
    except ValueError as error:
        raise VisionContractError(f"{field} must be an ISO-8601 string") from error
    return value


def _unit_interval(value: Any, field: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0 or value > 1:
        raise VisionContractError(f"{field} must be a finite number in [0, 1]")
    return float(value)


def _artifact_ref(value: Any) -> str:
    if not isinstance(value, str) or _ARTIFACT_REF.fullmatch(value) is None:
        raise VisionContractError("artifact_ref must be sha256:<hex> with optional suffix")
    return value


def _object(value: Any, field: str, allowed: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VisionContractError(f"{field} must be an object")
    extra = sorted(set(value) - allowed)
    if extra:
        raise VisionContractError(f"Unknown {field} fields: {', '.join(extra)}")
    return value
