"""Visual evidence contracts and local media artifacts. No camera or detector."""

from .artifacts import ArtifactStore
from .contracts import (
    CLIP_STATUSES,
    BoundingBox,
    Detection,
    EvidenceClip,
    ObservationEvidence,
    ObservationIdentity,
    ObservationProducer,
    ObservationSource,
    VisionContractError,
    VisualObservation,
    dumps_observation,
    dumps_observations_jsonl,
    evidence_clip_as_mapping,
    parse_evidence_clip,
    parse_observations_jsonl,
    parse_visual_observation,
    visual_observation_as_mapping,
)

__all__ = (
    "CLIP_STATUSES",
    "ArtifactStore",
    "BoundingBox",
    "Detection",
    "EvidenceClip",
    "ObservationEvidence",
    "ObservationIdentity",
    "ObservationProducer",
    "ObservationSource",
    "VisionContractError",
    "VisualObservation",
    "dumps_observation",
    "dumps_observations_jsonl",
    "evidence_clip_as_mapping",
    "parse_evidence_clip",
    "parse_observations_jsonl",
    "parse_visual_observation",
    "visual_observation_as_mapping",
)
