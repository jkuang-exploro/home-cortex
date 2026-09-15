"""Visual observation, clip metadata, and local artifact store. No detector."""
import hashlib
import json
from pathlib import Path

import pytest

from home_cortex.vision.artifacts import ArtifactStore
from home_cortex.vision.contracts import (
    VisionContractError,
    dumps_observation,
    dumps_observations_jsonl,
    evidence_clip_as_mapping,
    parse_evidence_clip,
    parse_observations_jsonl,
    parse_visual_observation,
    visual_observation_as_mapping,
)


def _observation(**fields):
    payload = {
        "id": "observation:mug_1",
        "observed_at": "2026-09-13T12:00:00-07:00",
        "source": {
            "device_id": "device:dev_macbook",
            "camera_id": "camera:built_in",
        },
        "detection": {
            "category": "mug",
            "confidence": 0.91,
            "bbox": {"x_min": 0.31, "y_min": 0.24, "x_max": 0.49, "y_max": 0.63},
        },
        "identity": {"canonical_item_id": None, "candidate_item_ids": []},
        "observer": None,
        "evidence": {"crop_ref": None, "clip_id": None},
        "producer": {
            "pipeline": "edge_vision",
            "pipeline_version": "v1",
            "detector": None,
            "detector_version": None,
        },
    }
    payload.update(fields)
    return payload


def test_valid_observation_without_spatial_state() -> None:
    observation = parse_visual_observation(_observation())
    assert observation.id == "observation:mug_1"
    assert observation.observer is None
    assert observation.detection.category == "mug"
    assert observation.detection.confidence == pytest.approx(0.91)
    assert observation.detection.bbox.x_min == pytest.approx(0.31)
    assert observation.identity.canonical_item_id is None
    assert observation.identity.candidate_item_ids == ()
    assert observation.source.device_id == "device:dev_macbook"
    assert observation.producer.pipeline == "edge_vision"


def test_observer_may_carry_opaque_spatial_fields_later() -> None:
    observation = parse_visual_observation(_observation(observer={
        "space": "space:kitchen",
        "quality": "localized",
    }))
    assert observation.observer is not None
    assert observation.observer["space"] == "space:kitchen"


def test_normalized_bbox_bounds() -> None:
    observation = parse_visual_observation(_observation())
    box = observation.detection.bbox
    assert 0 <= box.x_min < box.x_max <= 1
    assert 0 <= box.y_min < box.y_max <= 1


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
    detection = _observation()["detection"]
    detection["bbox"] = bbox
    with pytest.raises(VisionContractError, match="bbox"):
        parse_visual_observation(_observation(detection=detection))


def test_detector_native_fields_are_rejected() -> None:
    detection = _observation()["detection"]
    detection["class_id"] = 47
    detection["xyxy"] = [10, 20, 30, 40]
    with pytest.raises(VisionContractError, match="Unknown detection fields"):
        parse_visual_observation(_observation(detection=detection))
    with pytest.raises(VisionContractError, match="category"):
        parse_visual_observation(_observation(detection={
            "category": 47,
            "confidence": 0.9,
            "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2},
        }))


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_confidence_is_rejected(confidence) -> None:
    detection = _observation()["detection"]
    detection["confidence"] = confidence
    with pytest.raises(VisionContractError, match="confidence"):
        parse_visual_observation(_observation(detection=detection))


def test_timestamps_and_source_are_required() -> None:
    with pytest.raises(VisionContractError, match="ISO-8601"):
        parse_visual_observation(_observation(observed_at="later"))
    with pytest.raises(VisionContractError, match="device"):
        parse_visual_observation(_observation(source={
            "device_id": "macbook",
            "camera_id": "camera:built_in",
        }))


def test_candidate_identity_fields() -> None:
    observation = parse_visual_observation(_observation(identity={
        "canonical_item_id": "item:mug",
        "candidate_item_ids": ["item:mug", "item:cup"],
    }))
    assert observation.identity.canonical_item_id == "item:mug"
    assert observation.identity.candidate_item_ids == ("item:mug", "item:cup")


def test_evidence_clip_statuses() -> None:
    pending = parse_evidence_clip({
        "id": "clip:track_1",
        "status": "pending",
        "start_time": "2026-09-13T12:00:00-07:00",
        "end_time": None,
        "artifact_ref": None,
    })
    assert pending.status == "pending"
    digest = "a" * 64
    available = parse_evidence_clip({
        "id": "clip:track_1",
        "status": "available",
        "start_time": "2026-09-13T12:00:00-07:00",
        "end_time": "2026-09-13T12:00:04-07:00",
        "artifact_ref": f"sha256:{digest}.mp4",
    })
    assert available.status == "available"
    failed = parse_evidence_clip({
        "id": "clip:track_1",
        "status": "failed",
        "start_time": "2026-09-13T12:00:00-07:00",
    })
    assert failed.status == "failed"
    with pytest.raises(VisionContractError, match="artifact_ref"):
        parse_evidence_clip({
            "id": "clip:track_1",
            "status": "available",
            "start_time": "2026-09-13T12:00:00-07:00",
        })


def test_json_round_trip() -> None:
    original = parse_visual_observation(_observation())
    restored = parse_visual_observation(visual_observation_as_mapping(original))
    assert restored == original
    clip = parse_evidence_clip({
        "id": "clip:track_1",
        "status": "pending",
        "start_time": "2026-09-13T12:00:00-07:00",
    })
    assert parse_evidence_clip(evidence_clip_as_mapping(clip)) == clip


def test_observation_jsonl_round_trip() -> None:
    first = parse_visual_observation(_observation())
    second = parse_visual_observation(_observation(id="observation:mug_2"))
    text = dumps_observations_jsonl((first, second))
    restored = parse_observations_jsonl(text)
    assert restored == (first, second)
    assert json.loads(dumps_observation(first))["detection"]["category"] == "mug"


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
