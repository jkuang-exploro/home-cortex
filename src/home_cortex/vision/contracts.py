"""Canonical visual-evidence contracts.

Detector-native values stop at the edge adapter. A visual observation records
belief and evidence; it never creates an item, confirms identity, or changes a
household location. Raw artifact bytes live behind an artifact store.
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
from ..spatial.pose import (
    PoseUncertainty,
    RuntimePose,
    parse_runtime_pose,
    runtime_pose_as_mapping,
)
from ..spatial.transforms import Position

ARTIFACT_KINDS = ("video_clip", "crop", "thumbnail", "frame", "embedding")
CLIP_STATES = ("pending", "available", "failed")
CANDIDATE_STATES = ("active", "ignored", "merged", "expired")
ENROLLMENT_STATES = ("active", "revoked")

_ARTIFACT_REF = re.compile(r"^sha256:[0-9a-f]{64}(?:\.[A-Za-z0-9]+)?$")
_MEDIA_TYPE = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")
_OBSERVATION_FIELDS = frozenset({
    "artifacts", "captured_at", "detector_belief", "id", "object_position_estimate",
    "observed_at", "observer_pose", "producer_profile_id",
    "candidate_hypotheses", "item_hypotheses", "source", "track",
})
_SOURCE_FIELDS = frozenset({"camera_id", "device_id"})
_DETECTOR_BELIEF_FIELDS = frozenset({"bbox", "category", "confidence"})
_BBOX_FIELDS = frozenset({"x_max", "x_min", "y_max", "y_min"})
_TRACK_REF_FIELDS = frozenset({"session_id", "track_key"})
_CANDIDATE_HYPOTHESIS_FIELDS = frozenset({"candidate_id", "confidence"})
_ITEM_HYPOTHESIS_FIELDS = frozenset({
    "confidence", "item_id", "supporting_enrollment_ids",
})
_ARTIFACT_FIELDS = frozenset({"byte_length", "kind", "media_type", "ref"})
_OBJECT_POSITION_FIELDS = frozenset({"position", "space", "uncertainty"})
_POSITION_FIELDS = frozenset({"x", "y", "z"})
_UNCERTAINTY_FIELDS = frozenset({"x_sigma", "y_sigma", "z_sigma"})
_PRODUCER_FIELDS = frozenset({
    "detector_name", "detector_version", "id", "pipeline_name",
    "pipeline_version", "recognizer_name", "recognizer_version",
})
_TRACK_FIELDS = frozenset({
    "clip_ids", "last_observed_at", "observation_ids", "session_id",
    "source", "started_at", "track_key",
})
_CANDIDATE_FIELDS = frozenset({
    "created_at", "id", "merged_into_candidate_id", "seed_observation_ids", "state",
})
_CLIP_FIELDS = frozenset({
    "artifact", "end_time", "failure_code", "id", "observation_ids",
    "requested_at", "resolved_at", "source", "start_time", "state",
})
_ENROLLMENT_FIELDS = frozenset({
    "candidate_id", "confirmed_at", "confirmed_by", "evidence_observation_ids",
    "id", "item_id", "label_text", "revoked_at", "revoked_by", "state",
})


class VisionContractError(ValueError):
    """A value does not satisfy the canonical visual-evidence boundary."""


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in normalized image coordinates, origin at top-left."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class ObservationSource:
    """Physical capture source, independent of the model pipeline."""

    device_id: str
    camera_id: str


@dataclass(frozen=True)
class DetectorBelief:
    """A detector's category belief for one image region, not an identity."""

    category: str
    confidence: float
    bbox: BoundingBox


@dataclass(frozen=True)
class TrackReference:
    """Reference to an edge-local track, scoped by a perception session."""

    session_id: str
    track_key: str


@dataclass(frozen=True)
class VisualCandidateHypothesis:
    """Model belief that evidence belongs to a persistent visual candidate."""

    candidate_id: str
    confidence: float


@dataclass(frozen=True)
class HouseholdItemHypothesis:
    """Machine-proposed item identity grounded by human enrollments."""

    item_id: str
    confidence: float
    supporting_enrollment_ids: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactReference:
    """Storage-neutral artifact metadata. The referenced bytes are external."""

    ref: str
    kind: str
    media_type: str
    byte_length: int


@dataclass(frozen=True)
class ObjectPositionEstimate:
    """Optional object position in a known Epic 1 space coordinate system."""

    space: str
    position: Position
    uncertainty: PoseUncertainty | None = None


@dataclass(frozen=True)
class ProducerProfile:
    """Immutable, reusable pipeline/model provenance."""

    id: str
    pipeline_name: str
    pipeline_version: str
    detector_name: str | None = None
    detector_version: str | None = None
    recognizer_name: str | None = None
    recognizer_version: str | None = None


@dataclass(frozen=True)
class VisualObservation:
    """Immutable evidence that a producer reported one region at one time."""

    id: str
    captured_at: str
    observed_at: str
    source: ObservationSource
    detector_belief: DetectorBelief
    producer_profile_id: str
    track: TrackReference | None = None
    candidate_hypotheses: tuple[VisualCandidateHypothesis, ...] = ()
    item_hypotheses: tuple[HouseholdItemHypothesis, ...] = ()
    observer_pose: RuntimePose | None = None
    object_position_estimate: ObjectPositionEstimate | None = None
    artifacts: tuple[ArtifactReference, ...] = ()


@dataclass(frozen=True)
class VisualTrack:
    """Ephemeral edge grouping; it is never a SurrealDB domain record."""

    session_id: str
    track_key: str
    source: ObservationSource
    started_at: str
    last_observed_at: str
    observation_ids: tuple[str, ...] = ()
    clip_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class VisualCandidate:
    """Persistent visual identity anchor; not a category or household item."""

    id: str
    created_at: str
    state: str
    seed_observation_ids: tuple[str, ...]
    merged_into_candidate_id: str | None = None


@dataclass(frozen=True)
class EvidenceClip:
    """Asynchronous clip metadata; clip bytes remain in an artifact store."""

    id: str
    state: str
    requested_at: str
    start_time: str
    end_time: str
    source: ObservationSource
    observation_ids: tuple[str, ...]
    artifact: ArtifactReference | None = None
    failure_code: str | None = None
    resolved_at: str | None = None


@dataclass(frozen=True)
class VisualEnrollment:
    """Auditable human confirmation linking a candidate to a household item."""

    id: str
    candidate_id: str
    item_id: str
    confirmed_at: str
    confirmed_by: str
    evidence_observation_ids: tuple[str, ...]
    label_text: str | None = None
    state: str = "active"
    revoked_at: str | None = None
    revoked_by: str | None = None


def parse_visual_observation(value: Mapping[str, Any]) -> VisualObservation:
    item = _object(value, "visual observation", _OBSERVATION_FIELDS)
    candidate_hypotheses = _sequence(
        item.get("candidate_hypotheses", []), "candidate_hypotheses"
    )
    item_hypotheses = _sequence(
        item.get("item_hypotheses", []), "item_hypotheses"
    )
    artifacts = tuple(
        _artifact(entry) for entry in _sequence(item.get("artifacts", []), "artifacts")
    )
    if any(artifact.kind == "video_clip" for artifact in artifacts):
        raise VisionContractError(
            "video_clip artifacts belong to EvidenceClip metadata"
        )
    captured_at = _timestamp(item.get("captured_at"), "captured_at")
    observed_at = _timestamp(item.get("observed_at"), "observed_at")
    _ordered_times(captured_at, observed_at, "observed_at")
    parsed_candidate_hypotheses = tuple(
        _candidate_hypothesis(entry) for entry in candidate_hypotheses
    )
    if len({entry.candidate_id for entry in parsed_candidate_hypotheses}) != len(
        parsed_candidate_hypotheses
    ):
        raise VisionContractError(
            "candidate_hypotheses must not repeat a candidate_id"
        )
    parsed_item_hypotheses = tuple(
        _item_hypothesis(entry) for entry in item_hypotheses
    )
    if len({entry.item_id for entry in parsed_item_hypotheses}) != len(
        parsed_item_hypotheses
    ):
        raise VisionContractError("item_hypotheses must not repeat an item_id")
    return VisualObservation(
        id=_typed_id(item.get("id"), "id", "visual_observation"),
        captured_at=captured_at,
        observed_at=observed_at,
        source=_source(item.get("source")),
        detector_belief=_detector_belief(item.get("detector_belief")),
        producer_profile_id=_typed_id(
            item.get("producer_profile_id"), "producer_profile_id", "vision_producer"
        ),
        track=None if item.get("track") is None else _track_reference(item["track"]),
        candidate_hypotheses=parsed_candidate_hypotheses,
        item_hypotheses=parsed_item_hypotheses,
        observer_pose=_observer_pose(item.get("observer_pose")),
        object_position_estimate=(
            None if item.get("object_position_estimate") is None
            else _object_position_estimate(item["object_position_estimate"])
        ),
        artifacts=artifacts,
    )


def visual_observation_as_mapping(value: VisualObservation) -> dict[str, Any]:
    return {
        "id": value.id,
        "captured_at": value.captured_at,
        "observed_at": value.observed_at,
        "source": _source_as_mapping(value.source),
        "detector_belief": {
            "category": value.detector_belief.category,
            "confidence": value.detector_belief.confidence,
            "bbox": _bbox_as_mapping(value.detector_belief.bbox),
        },
        "producer_profile_id": value.producer_profile_id,
        "track": None if value.track is None else {
            "session_id": value.track.session_id,
            "track_key": value.track.track_key,
        },
        "candidate_hypotheses": [
            {"candidate_id": candidate.candidate_id, "confidence": candidate.confidence}
            for candidate in value.candidate_hypotheses
        ],
        "item_hypotheses": [
            {
                "item_id": hypothesis.item_id,
                "confidence": hypothesis.confidence,
                "supporting_enrollment_ids": list(
                    hypothesis.supporting_enrollment_ids
                ),
            }
            for hypothesis in value.item_hypotheses
        ],
        "observer_pose": (
            None if value.observer_pose is None
            else runtime_pose_as_mapping(value.observer_pose)
        ),
        "object_position_estimate": _object_position_as_mapping(
            value.object_position_estimate
        ),
        "artifacts": [_artifact_as_mapping(artifact) for artifact in value.artifacts],
    }


def parse_producer_profile(value: Mapping[str, Any]) -> ProducerProfile:
    item = _object(value, "producer profile", _PRODUCER_FIELDS)
    detector_name, detector_version = _optional_pair(
        item, "detector_name", "detector_version"
    )
    recognizer_name, recognizer_version = _optional_pair(
        item, "recognizer_name", "recognizer_version"
    )
    return ProducerProfile(
        id=_typed_id(item.get("id"), "id", "vision_producer"),
        pipeline_name=_text(item.get("pipeline_name"), "pipeline_name"),
        pipeline_version=_text(item.get("pipeline_version"), "pipeline_version"),
        detector_name=detector_name,
        detector_version=detector_version,
        recognizer_name=recognizer_name,
        recognizer_version=recognizer_version,
    )


def producer_profile_as_mapping(value: ProducerProfile) -> dict[str, Any]:
    return {
        "id": value.id,
        "pipeline_name": value.pipeline_name,
        "pipeline_version": value.pipeline_version,
        "detector_name": value.detector_name,
        "detector_version": value.detector_version,
        "recognizer_name": value.recognizer_name,
        "recognizer_version": value.recognizer_version,
    }


def parse_artifact_reference(value: Mapping[str, Any]) -> ArtifactReference:
    return _artifact(value)


def artifact_reference_as_mapping(value: ArtifactReference) -> dict[str, Any]:
    return _artifact_as_mapping(value)


def parse_visual_track(value: Mapping[str, Any]) -> VisualTrack:
    item = _object(value, "visual track", _TRACK_FIELDS)
    started = _timestamp(item.get("started_at"), "started_at")
    last = _timestamp(item.get("last_observed_at"), "last_observed_at")
    _ordered_times(started, last, "track last_observed_at")
    return VisualTrack(
        session_id=_typed_id(item.get("session_id"), "session_id", "perception"),
        track_key=_text(item.get("track_key"), "track_key"),
        source=_source(item.get("source")),
        started_at=started,
        last_observed_at=last,
        observation_ids=_typed_ids(
            item.get("observation_ids", []), "observation_ids", "visual_observation"
        ),
        clip_ids=_typed_ids(item.get("clip_ids", []), "clip_ids", "evidence_clip"),
    )


def visual_track_as_mapping(value: VisualTrack) -> dict[str, Any]:
    return {
        "session_id": value.session_id,
        "track_key": value.track_key,
        "source": _source_as_mapping(value.source),
        "started_at": value.started_at,
        "last_observed_at": value.last_observed_at,
        "observation_ids": list(value.observation_ids),
        "clip_ids": list(value.clip_ids),
    }


def parse_visual_candidate(value: Mapping[str, Any]) -> VisualCandidate:
    item = _object(value, "visual candidate", _CANDIDATE_FIELDS)
    state = _choice(item.get("state"), "candidate state", CANDIDATE_STATES)
    seed_ids = _typed_ids(
        item.get("seed_observation_ids"), "seed_observation_ids",
        "visual_observation", require_nonempty=True,
    )
    merged_into = item.get("merged_into_candidate_id")
    if merged_into is not None:
        merged_into = _typed_id(
            merged_into, "merged_into_candidate_id", "visual_candidate"
        )
    if (state == "merged") != (merged_into is not None):
        raise VisionContractError(
            "merged candidate state requires merged_into_candidate_id, and other states forbid it"
        )
    candidate_id = _typed_id(item.get("id"), "id", "visual_candidate")
    if merged_into == candidate_id:
        raise VisionContractError("candidate cannot be merged into itself")
    return VisualCandidate(
        id=candidate_id,
        created_at=_timestamp(item.get("created_at"), "created_at"),
        state=state,
        seed_observation_ids=seed_ids,
        merged_into_candidate_id=merged_into,
    )


def visual_candidate_as_mapping(value: VisualCandidate) -> dict[str, Any]:
    return {
        "id": value.id,
        "created_at": value.created_at,
        "state": value.state,
        "seed_observation_ids": list(value.seed_observation_ids),
        "merged_into_candidate_id": value.merged_into_candidate_id,
    }


def parse_evidence_clip(value: Mapping[str, Any]) -> EvidenceClip:
    item = _object(value, "evidence clip", _CLIP_FIELDS)
    state = _choice(item.get("state"), "clip state", CLIP_STATES)
    artifact = None if item.get("artifact") is None else _artifact(item["artifact"])
    failure_code = (
        None if item.get("failure_code") is None
        else _text(item["failure_code"], "failure_code")
    )
    resolved_at = (
        None if item.get("resolved_at") is None
        else _timestamp(item["resolved_at"], "resolved_at")
    )
    if state == "available":
        if artifact is None or artifact.kind != "video_clip":
            raise VisionContractError("available clip requires a video_clip artifact")
        if failure_code is not None:
            raise VisionContractError("available clip cannot include failure_code")
        if resolved_at is None:
            raise VisionContractError("available clip requires resolved_at")
    elif state == "failed":
        if failure_code is None:
            raise VisionContractError("failed clip requires failure_code")
        if artifact is not None:
            raise VisionContractError("failed clip cannot include an artifact")
        if resolved_at is None:
            raise VisionContractError("failed clip requires resolved_at")
    elif artifact is not None or failure_code is not None or resolved_at is not None:
        raise VisionContractError(
            "pending clip cannot include artifact, failure_code, or resolved_at"
        )
    start = _timestamp(item.get("start_time"), "start_time")
    end = _timestamp(item.get("end_time"), "end_time")
    _ordered_times(start, end, "clip end_time")
    requested_at = _timestamp(item.get("requested_at"), "requested_at")
    if resolved_at is not None:
        _ordered_times(requested_at, resolved_at, "clip resolved_at")
    return EvidenceClip(
        id=_typed_id(item.get("id"), "id", "evidence_clip"),
        state=state,
        requested_at=requested_at,
        start_time=start,
        end_time=end,
        source=_source(item.get("source")),
        observation_ids=_typed_ids(
            item.get("observation_ids"), "observation_ids", "visual_observation",
            require_nonempty=True,
        ),
        artifact=artifact,
        failure_code=failure_code,
        resolved_at=resolved_at,
    )


def evidence_clip_as_mapping(value: EvidenceClip) -> dict[str, Any]:
    return {
        "id": value.id,
        "state": value.state,
        "requested_at": value.requested_at,
        "start_time": value.start_time,
        "end_time": value.end_time,
        "source": _source_as_mapping(value.source),
        "observation_ids": list(value.observation_ids),
        "artifact": None if value.artifact is None else _artifact_as_mapping(value.artifact),
        "failure_code": value.failure_code,
        "resolved_at": value.resolved_at,
    }


def parse_visual_enrollment(value: Mapping[str, Any]) -> VisualEnrollment:
    item = _object(value, "visual enrollment", _ENROLLMENT_FIELDS)
    state = _choice(item.get("state", "active"), "enrollment state", ENROLLMENT_STATES)
    revoked_at = (
        None if item.get("revoked_at") is None
        else _timestamp(item["revoked_at"], "revoked_at")
    )
    revoked_by = (
        None if item.get("revoked_by") is None
        else _typed_id(item["revoked_by"], "revoked_by", "person")
    )
    if state == "revoked":
        if revoked_at is None or revoked_by is None:
            raise VisionContractError("revoked enrollment requires revoked_at and revoked_by")
    elif revoked_at is not None or revoked_by is not None:
        raise VisionContractError("active enrollment cannot include revocation fields")
    confirmed_at = _timestamp(item.get("confirmed_at"), "confirmed_at")
    if revoked_at is not None:
        _ordered_times(confirmed_at, revoked_at, "enrollment revoked_at")
    return VisualEnrollment(
        id=_typed_id(item.get("id"), "id", "visual_enrollment"),
        candidate_id=_typed_id(
            item.get("candidate_id"), "candidate_id", "visual_candidate"
        ),
        item_id=_typed_id(item.get("item_id"), "item_id", "item"),
        confirmed_at=confirmed_at,
        confirmed_by=_typed_id(item.get("confirmed_by"), "confirmed_by", "person"),
        evidence_observation_ids=_typed_ids(
            item.get("evidence_observation_ids"), "evidence_observation_ids",
            "visual_observation", require_nonempty=True,
        ),
        label_text=(
            None if item.get("label_text") is None
            else _text(item["label_text"], "label_text")
        ),
        state=state,
        revoked_at=revoked_at,
        revoked_by=revoked_by,
    )


def visual_enrollment_as_mapping(value: VisualEnrollment) -> dict[str, Any]:
    return {
        "id": value.id,
        "candidate_id": value.candidate_id,
        "item_id": value.item_id,
        "confirmed_at": value.confirmed_at,
        "confirmed_by": value.confirmed_by,
        "evidence_observation_ids": list(value.evidence_observation_ids),
        "label_text": value.label_text,
        "state": value.state,
        "revoked_at": value.revoked_at,
        "revoked_by": value.revoked_by,
    }


def dumps_observation(value: VisualObservation) -> str:
    return json.dumps(
        visual_observation_as_mapping(value), ensure_ascii=False,
        separators=(",", ":"), sort_keys=True,
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


def _source_as_mapping(value: ObservationSource) -> dict[str, str]:
    return {"device_id": value.device_id, "camera_id": value.camera_id}


def _detector_belief(value: Any) -> DetectorBelief:
    item = _object(value, "detector belief", _DETECTOR_BELIEF_FIELDS)
    return DetectorBelief(
        category=_text(item.get("category"), "detector_belief.category"),
        confidence=_unit_interval(item.get("confidence"), "detector_belief.confidence"),
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
        raise VisionContractError(
            "bbox requires 0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1"
        )
    return box


def _bbox_as_mapping(value: BoundingBox) -> dict[str, float]:
    return {
        "x_min": value.x_min, "y_min": value.y_min,
        "x_max": value.x_max, "y_max": value.y_max,
    }


def _track_reference(value: Any) -> TrackReference:
    item = _object(value, "track reference", _TRACK_REF_FIELDS)
    return TrackReference(
        session_id=_typed_id(item.get("session_id"), "track.session_id", "perception"),
        track_key=_text(item.get("track_key"), "track.track_key"),
    )


def _candidate_hypothesis(value: Any) -> VisualCandidateHypothesis:
    item = _object(
        value, "visual candidate hypothesis", _CANDIDATE_HYPOTHESIS_FIELDS
    )
    return VisualCandidateHypothesis(
        candidate_id=_typed_id(
            item.get("candidate_id"),
            "candidate_hypothesis.candidate_id",
            "visual_candidate",
        ),
        confidence=_unit_interval(
            item.get("confidence"), "candidate_hypothesis.confidence"
        ),
    )


def _item_hypothesis(value: Any) -> HouseholdItemHypothesis:
    item = _object(value, "household item hypothesis", _ITEM_HYPOTHESIS_FIELDS)
    return HouseholdItemHypothesis(
        item_id=_typed_id(item.get("item_id"), "item_hypothesis.item_id", "item"),
        confidence=_unit_interval(
            item.get("confidence"), "item_hypothesis.confidence"
        ),
        supporting_enrollment_ids=_typed_ids(
            item.get("supporting_enrollment_ids"),
            "item_hypothesis.supporting_enrollment_ids",
            "visual_enrollment",
            require_nonempty=True,
        ),
    )


def _artifact(value: Any) -> ArtifactReference:
    item = _object(value, "artifact reference", _ARTIFACT_FIELDS)
    ref = item.get("ref")
    if not isinstance(ref, str) or _ARTIFACT_REF.fullmatch(ref) is None:
        raise VisionContractError("artifact ref must be sha256:<hex> with optional suffix")
    kind = _choice(item.get("kind"), "artifact kind", ARTIFACT_KINDS)
    media_type = item.get("media_type")
    if not isinstance(media_type, str) or _MEDIA_TYPE.fullmatch(media_type) is None:
        raise VisionContractError("artifact media_type must be a MIME type")
    byte_length = item.get("byte_length")
    if type(byte_length) is not int or byte_length < 0:
        raise VisionContractError("artifact byte_length must be a non-negative integer")
    return ArtifactReference(ref, kind, media_type.lower(), byte_length)


def _artifact_as_mapping(value: ArtifactReference) -> dict[str, Any]:
    return {
        "ref": value.ref, "kind": value.kind,
        "media_type": value.media_type, "byte_length": value.byte_length,
    }


def _observer_pose(value: Any) -> RuntimePose | None:
    if value is None:
        return None
    try:
        return parse_runtime_pose(value)
    except ValueError as error:
        raise VisionContractError(f"observer_pose: {error}") from error


def _object_position_estimate(value: Any) -> ObjectPositionEstimate:
    item = _object(value, "object position estimate", _OBJECT_POSITION_FIELDS)
    position_item = _object(item.get("position"), "object position", _POSITION_FIELDS)
    position = Position(
        _finite(position_item.get("x"), "object position.x"),
        _finite(position_item.get("y"), "object position.y"),
        _finite(position_item.get("z"), "object position.z"),
    )
    uncertainty = None
    if item.get("uncertainty") is not None:
        uncertainty_item = _object(
            item["uncertainty"], "object position uncertainty", _UNCERTAINTY_FIELDS
        )
        uncertainty = PoseUncertainty(
            x_sigma=_optional_nonnegative(uncertainty_item, "x_sigma"),
            y_sigma=_optional_nonnegative(uncertainty_item, "y_sigma"),
            z_sigma=_optional_nonnegative(uncertainty_item, "z_sigma"),
        )
    return ObjectPositionEstimate(
        space=_typed_id(item.get("space"), "object position space", "space"),
        position=position,
        uncertainty=uncertainty,
    )


def _object_position_as_mapping(
    value: ObjectPositionEstimate | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    result: dict[str, Any] = {
        "space": value.space,
        "position": {
            "x": value.position.x, "y": value.position.y, "z": value.position.z,
        },
        "uncertainty": None,
    }
    if value.uncertainty is not None:
        result["uncertainty"] = {
            name: getattr(value.uncertainty, name)
            for name in ("x_sigma", "y_sigma", "z_sigma")
            if getattr(value.uncertainty, name) is not None
        }
    return result


def _optional_pair(
    value: Mapping[str, Any], name_field: str, version_field: str
) -> tuple[str | None, str | None]:
    name = value.get(name_field)
    version = value.get(version_field)
    if (name is None) != (version is None):
        raise VisionContractError(f"{name_field} and {version_field} must appear together")
    if name is None:
        return None, None
    return _text(name, name_field), _text(version, version_field)


def _typed_ids(
    value: Any, field: str, table: str, *, require_nonempty: bool = False,
) -> tuple[str, ...]:
    items = _sequence(value, field)
    if require_nonempty and not items:
        raise VisionContractError(f"{field} must contain at least one record ID")
    parsed = tuple(_typed_id(item, field, table) for item in items)
    if len(set(parsed)) != len(parsed):
        raise VisionContractError(f"{field} must not contain duplicate record IDs")
    return parsed


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
        raise VisionContractError(f"{field} must be a timezone-aware ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise VisionContractError(
            f"{field} must be a timezone-aware ISO-8601 string"
        ) from error
    if parsed.utcoffset() is None:
        raise VisionContractError(f"{field} must be a timezone-aware ISO-8601 string")
    return value


def _ordered_times(start: str, end: str, field: str) -> None:
    if datetime.fromisoformat(end) < datetime.fromisoformat(start):
        raise VisionContractError(f"{field} must not precede its start time")


def _choice(value: Any, field: str, choices: Sequence[str]) -> str:
    if value not in choices:
        raise VisionContractError(f"{field} must be one of {', '.join(choices)}")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VisionContractError(f"{field} must be a non-empty string")
    return value.strip()


def _finite(value: Any, field: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise VisionContractError(f"{field} must be a finite number")
    return float(value)


def _unit_interval(value: Any, field: str) -> float:
    result = _finite(value, field)
    if result < 0 or result > 1:
        raise VisionContractError(f"{field} must be a finite number in [0, 1]")
    return result


def _optional_nonnegative(value: Mapping[str, Any], field: str) -> float | None:
    if field not in value:
        return None
    result = _finite(value[field], f"uncertainty.{field}")
    if result < 0:
        raise VisionContractError(f"uncertainty.{field} must be non-negative")
    return result


def _sequence(value: Any, field: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise VisionContractError(f"{field} must be a list")
    return list(value)


def _object(value: Any, field: str, allowed: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VisionContractError(f"{field} must be an object")
    extra = sorted(set(value) - allowed)
    if extra:
        raise VisionContractError(f"Unknown {field} fields: {', '.join(extra)}")
    return value
