"""Canonical visual evidence contracts and local artifact storage."""
import hashlib
import json
from pathlib import Path

import pytest

from home_cortex.vision.artifacts import ArtifactStore
from home_cortex.vision.contracts import (
    VisionContractError,
    artifact_reference_as_mapping,
    dumps_observation,
    dumps_observations_jsonl,
    evidence_clip_as_mapping,
    parse_artifact_reference,
    parse_evidence_clip,
    parse_observations_jsonl,
    parse_producer_profile,
    parse_visual_candidate,
    parse_visual_enrollment,
    parse_visual_observation,
    parse_visual_track,
    producer_profile_as_mapping,
    visual_candidate_as_mapping,
    visual_enrollment_as_mapping,
    visual_observation_as_mapping,
    visual_track_as_mapping,
)

STAMP = "2026-09-13T12:00:00-07:00"
OBSERVED = "2026-09-13T12:00:01-07:00"
LATER = "2026-09-13T12:00:04-07:00"
SOURCE = {"device_id": "device:dev_macbook", "camera_id": "camera:built_in"}


def _observation(**fields):
    payload = {
        "id": "visual_observation:mug_1",
        "captured_at": STAMP,
        "observed_at": OBSERVED,
        "source": SOURCE,
        "detector_belief": {
            "category": "mug",
            "confidence": 0.91,
            "bbox": {"x_min": 0.31, "y_min": 0.24, "x_max": 0.49, "y_max": 0.63},
        },
        "producer_profile_id": "vision_producer:mac_v1",
    }
    payload.update(fields)
    return payload


def _artifact(kind="crop", media_type="image/jpeg"):
    return {
        "ref": f"sha256:{'a' * 64}.jpg",
        "kind": kind,
        "media_type": media_type,
        "byte_length": 2048,
    }


def _clip(**fields):
    payload = {
        "id": "evidence_clip:clip_1",
        "state": "pending",
        "requested_at": STAMP,
        "start_time": STAMP,
        "end_time": LATER,
        "source": SOURCE,
        "observation_ids": ["visual_observation:mug_1"],
    }
    payload.update(fields)
    return payload


def test_observation_only_requires_detector_belief_and_provenance() -> None:
    observation = parse_visual_observation(_observation())
    assert observation.detector_belief.category == "mug"
    assert observation.detector_belief.confidence == pytest.approx(0.91)
    assert observation.track is None
    assert observation.candidate_hypotheses == ()
    assert observation.item_hypotheses == ()
    assert observation.observer_pose is None
    assert observation.object_position_estimate is None
    assert observation.artifacts == ()


def test_observation_can_carry_typed_spatial_enrichment() -> None:
    observation = parse_visual_observation(_observation(
        observer_pose={
            "agent": "agent:microduck",
            "space": "space:kitchen",
            "quality": "localized",
            "position": {"x": 1.0, "y": 2.0, "z": 0.0},
            "orientation": {"yaw": 0.1, "pitch": 0.0, "roll": 0.0},
            "timestamp": STAMP,
            "source": "fiducial-localizer:v1",
        },
        object_position_estimate={
            "space": "space:kitchen",
            "position": {"x": 1.5, "y": 2.2, "z": 0.8},
            "uncertainty": {"x_sigma": 0.05, "y_sigma": 0.06, "z_sigma": 0.1},
        },
    ))
    assert observation.observer_pose is not None
    assert observation.observer_pose.estimate is not None
    assert observation.object_position_estimate is not None
    assert observation.object_position_estimate.position.z == pytest.approx(0.8)
    assert observation.object_position_estimate.uncertainty is not None
    assert observation.object_position_estimate.uncertainty.x_sigma == pytest.approx(0.05)


def test_detector_native_and_identity_shortcut_fields_are_rejected() -> None:
    belief = _observation()["detector_belief"]
    belief["class_id"] = 47
    belief["xyxy"] = [10, 20, 30, 40]
    with pytest.raises(VisionContractError, match="Unknown detector belief fields"):
        parse_visual_observation(_observation(detector_belief=belief))
    with pytest.raises(VisionContractError, match="Unknown visual observation fields: identity"):
        parse_visual_observation(_observation(identity={"canonical_item_id": "item:mug"}))


def test_candidate_hypotheses_target_visual_candidates_only() -> None:
    observation = parse_visual_observation(_observation(
        candidate_hypotheses=[
            {"candidate_id": "visual_candidate:abc123", "confidence": 0.84}
        ]
    ))
    assert observation.candidate_hypotheses[0].candidate_id == "visual_candidate:abc123"
    with pytest.raises(VisionContractError, match="visual_candidate"):
        parse_visual_observation(_observation(
            candidate_hypotheses=[{"candidate_id": "item:mug", "confidence": 0.84}]
        ))


def test_machine_item_hypothesis_is_scored_and_grounded_by_enrollment() -> None:
    observation = parse_visual_observation(_observation(
        item_hypotheses=[{
            "item_id": "item:black_coffee_mug",
            "confidence": 0.92,
            "supporting_enrollment_ids": ["visual_enrollment:black_mug"],
        }]
    ))
    hypothesis = observation.item_hypotheses[0]
    assert hypothesis.item_id == "item:black_coffee_mug"
    assert hypothesis.confidence == pytest.approx(0.92)
    assert hypothesis.supporting_enrollment_ids == (
        "visual_enrollment:black_mug",
    )

    with pytest.raises(VisionContractError, match="at least one"):
        parse_visual_observation(_observation(item_hypotheses=[{
            "item_id": "item:black_coffee_mug",
            "confidence": 0.92,
            "supporting_enrollment_ids": [],
        }]))
    with pytest.raises(VisionContractError, match="item:"):
        parse_visual_observation(_observation(item_hypotheses=[{
            "item_id": "visual_candidate:abc123",
            "confidence": 0.92,
            "supporting_enrollment_ids": ["visual_enrollment:black_mug"],
        }]))


def test_legacy_or_collapsed_identity_fields_are_rejected() -> None:
    with pytest.raises(VisionContractError, match="recognition_hypotheses"):
        parse_visual_observation(_observation(recognition_hypotheses=[]))
    with pytest.raises(VisionContractError, match="canonical_item_id"):
        parse_visual_observation(_observation(
            canonical_item_id="item:black_coffee_mug"
        ))


@pytest.mark.parametrize(
    "field,target",
    [
        (
            "candidate_hypotheses",
            {"candidate_id": "visual_candidate:abc123", "confidence": 0.8},
        ),
        (
            "item_hypotheses",
            {
                "item_id": "item:black_coffee_mug",
                "confidence": 0.8,
                "supporting_enrollment_ids": ["visual_enrollment:black_mug"],
            },
        ),
    ],
)
def test_observation_rejects_duplicate_hypothesis_targets(field, target) -> None:
    with pytest.raises(VisionContractError, match="must not repeat"):
        parse_visual_observation(_observation(**{field: [target, target]}))


def test_video_clip_artifact_cannot_bypass_clip_lifecycle() -> None:
    with pytest.raises(VisionContractError, match="EvidenceClip"):
        parse_visual_observation(_observation(
            artifacts=[_artifact(kind="video_clip", media_type="video/mp4")]
        ))


@pytest.mark.parametrize(
    "bbox",
    [
        {"x_min": 0.6, "y_min": 0.1, "x_max": 0.4, "y_max": 0.5},
        {"x_min": -0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.5},
        {"x_min": 0.1, "y_min": 0.1, "x_max": 1.1, "y_max": 0.5},
        {"x_min": 0.1, "y_min": 0.4, "x_max": 0.4, "y_max": 0.4},
    ],
)
def test_invalid_bbox_is_rejected(bbox) -> None:
    belief = _observation()["detector_belief"]
    belief["bbox"] = bbox
    with pytest.raises(VisionContractError, match="bbox"):
        parse_visual_observation(_observation(detector_belief=belief))


def test_observation_rejects_naive_timestamp_and_untyped_source() -> None:
    with pytest.raises(VisionContractError, match="timezone-aware"):
        parse_visual_observation(_observation(observed_at="2026-09-13T12:00:00"))
    with pytest.raises(VisionContractError, match="observed_at"):
        parse_visual_observation(_observation(
            captured_at=OBSERVED,
            observed_at=STAMP,
        ))
    with pytest.raises(VisionContractError, match="device"):
        parse_visual_observation(_observation(source={
            "device_id": "macbook", "camera_id": "camera:built_in"
        }))


def test_producer_profile_deduplicates_model_provenance() -> None:
    profile = parse_producer_profile({
        "id": "vision_producer:mac_yolo11_v1",
        "pipeline_name": "edge_vision",
        "pipeline_version": "1.0.0",
        "detector_name": "yolo11n",
        "detector_version": "sha256:weights",
        "recognizer_name": "clip",
        "recognizer_version": "vit-b-32",
    })
    assert profile.detector_name == "yolo11n"
    assert profile.recognizer_version == "vit-b-32"
    assert parse_producer_profile(producer_profile_as_mapping(profile)) == profile
    with pytest.raises(VisionContractError, match="must appear together"):
        parse_producer_profile({
            "id": "vision_producer:bad",
            "pipeline_name": "edge_vision",
            "pipeline_version": "1.0.0",
            "detector_name": "yolo11n",
        })


def test_track_is_session_scoped_and_can_reference_multiple_clips() -> None:
    track = parse_visual_track({
        "session_id": "perception:mac_run_1",
        "track_key": "local-17",
        "source": SOURCE,
        "started_at": STAMP,
        "last_observed_at": LATER,
        "observation_ids": ["visual_observation:mug_1", "visual_observation:mug_2"],
        "clip_ids": ["evidence_clip:before", "evidence_clip:after"],
    })
    assert track.track_key == "local-17"
    assert len(track.clip_ids) == 2
    assert parse_visual_track(visual_track_as_mapping(track)) == track


def test_candidate_is_a_persistent_identity_anchor_with_seed_evidence() -> None:
    candidate = parse_visual_candidate({
        "id": "visual_candidate:abc123",
        "created_at": STAMP,
        "state": "active",
        "seed_observation_ids": ["visual_observation:mug_1"],
    })
    assert candidate.id == "visual_candidate:abc123"
    assert not hasattr(candidate, "item_id")
    assert parse_visual_candidate(visual_candidate_as_mapping(candidate)) == candidate
    with pytest.raises(VisionContractError, match="at least one"):
        parse_visual_candidate({
            "id": "visual_candidate:empty",
            "created_at": STAMP,
            "state": "active",
            "seed_observation_ids": [],
        })


def test_candidate_merge_is_explicit_and_cannot_target_self() -> None:
    merged = parse_visual_candidate({
        "id": "visual_candidate:duplicate",
        "created_at": STAMP,
        "state": "merged",
        "seed_observation_ids": ["visual_observation:mug_1"],
        "merged_into_candidate_id": "visual_candidate:canonical",
    })
    assert merged.merged_into_candidate_id == "visual_candidate:canonical"
    with pytest.raises(VisionContractError, match="cannot be merged into itself"):
        parse_visual_candidate({
            "id": "visual_candidate:same",
            "created_at": STAMP,
            "state": "merged",
            "seed_observation_ids": ["visual_observation:mug_1"],
            "merged_into_candidate_id": "visual_candidate:same",
        })


@pytest.mark.parametrize("state", ["active", "ignored", "expired"])
def test_nonmerged_candidate_states_forbid_a_merge_target(state) -> None:
    candidate = parse_visual_candidate({
        "id": "visual_candidate:abc123",
        "created_at": STAMP,
        "state": state,
        "seed_observation_ids": ["visual_observation:mug_1"],
    })
    assert candidate.state == state
    with pytest.raises(VisionContractError, match="other states forbid"):
        parse_visual_candidate({
            "id": "visual_candidate:abc123",
            "created_at": STAMP,
            "state": state,
            "seed_observation_ids": ["visual_observation:mug_1"],
            "merged_into_candidate_id": "visual_candidate:survivor",
        })


def test_clip_lifecycle_is_independent_and_many_to_many() -> None:
    observation = parse_visual_observation(_observation())
    pending = parse_evidence_clip(_clip(
        observation_ids=["visual_observation:mug_1", "visual_observation:mug_2"]
    ))
    assert observation.artifacts == ()
    assert pending.state == "pending"
    assert len(pending.observation_ids) == 2

    available = parse_evidence_clip(_clip(
        state="available",
        artifact=_artifact(kind="video_clip", media_type="video/mp4"),
        resolved_at=LATER,
    ))
    assert available.artifact is not None
    assert available.artifact.kind == "video_clip"

    failed = parse_evidence_clip(_clip(
        state="failed",
        failure_code="encoder_failed",
        resolved_at=LATER,
    ))
    assert failed.failure_code == "encoder_failed"
    assert failed.artifact is None


@pytest.mark.parametrize(
    "fields,match",
    [
        ({"state": "available"}, "video_clip artifact"),
        ({"state": "failed"}, "failure_code"),
        ({
            "state": "available",
            "artifact": _artifact(kind="video_clip", media_type="video/mp4"),
        }, "resolved_at"),
        ({"state": "failed", "failure_code": "encoder_failed"}, "resolved_at"),
        ({"state": "pending", "failure_code": "late"}, "pending clip"),
        ({"state": "pending", "observation_ids": []}, "at least one"),
    ],
)
def test_invalid_clip_states_are_rejected(fields, match) -> None:
    with pytest.raises(VisionContractError, match=match):
        parse_evidence_clip(_clip(**fields))


def test_enrollment_is_the_only_candidate_to_item_confirmation() -> None:
    active_payload = {
        "id": "visual_enrollment:black_mug",
        "candidate_id": "visual_candidate:abc123",
        "item_id": "item:black_coffee_mug",
        "confirmed_at": STAMP,
        "confirmed_by": "person:jian",
        "evidence_observation_ids": ["visual_observation:mug_1"],
        "label_text": "This is my black coffee mug.",
        "state": "active",
    }
    active = parse_visual_enrollment(active_payload)
    assert active.item_id == "item:black_coffee_mug"
    assert active.confirmed_by == "person:jian"
    assert active.label_text == "This is my black coffee mug."
    assert parse_visual_enrollment(visual_enrollment_as_mapping(active)) == active

    revoked = parse_visual_enrollment({
        **active_payload,
        "state": "revoked",
        "revoked_at": LATER,
        "revoked_by": "person:jian",
    })
    assert revoked.state == "revoked"
    with pytest.raises(VisionContractError, match="active enrollment"):
        parse_visual_enrollment({**active_payload, "revoked_at": LATER})


def test_observation_and_clip_json_round_trip() -> None:
    original = parse_visual_observation(_observation(
        track={"session_id": "perception:mac_run_1", "track_key": "17"},
        candidate_hypotheses=[
            {"candidate_id": "visual_candidate:abc123", "confidence": 0.84}
        ],
        item_hypotheses=[{
            "item_id": "item:black_coffee_mug",
            "confidence": 0.92,
            "supporting_enrollment_ids": ["visual_enrollment:black_mug"],
        }],
        artifacts=[_artifact()],
    ))
    restored = parse_visual_observation(visual_observation_as_mapping(original))
    assert restored == original
    assert json.loads(dumps_observation(original))["detector_belief"]["category"] == "mug"

    clip = parse_evidence_clip(_clip())
    assert parse_evidence_clip(evidence_clip_as_mapping(clip)) == clip
    artifact = parse_artifact_reference(_artifact())
    assert parse_artifact_reference(artifact_reference_as_mapping(artifact)) == artifact

    second = parse_visual_observation(_observation(id="visual_observation:mug_2"))
    assert parse_observations_jsonl(dumps_observations_jsonl((original, second))) == (
        original,
        second,
    )


def test_artifact_store_write_read_and_hash_stability(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    payload = b"crop-bytes"
    ref = store.put(payload, suffix=".jpg")
    digest = hashlib.sha256(payload).hexdigest()
    assert ref == f"sha256:{digest}.jpg"
    assert store.exists(ref)
    assert store.get(ref) == payload
    assert store.put(payload, suffix=".jpg") == ref
    source = tmp_path / "frame.bin"
    source.write_bytes(payload)
    from_path = store.put(source)
    assert from_path.startswith("sha256:")
    assert store.get(from_path) == payload
