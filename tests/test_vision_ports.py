"""Transport-neutral observation ingestion and clip-delivery semantics."""
from datetime import datetime, timezone

import pytest

from home_cortex.vision.contracts import (
    dumps_observations_jsonl,
    parse_evidence_clip,
    parse_visual_observation,
)
from home_cortex.vision.ports import (
    ObservationIngestRecord,
    ObservationIngestionService,
    evaluate_evidence_clip_write,
)

CAPTURED = "2026-09-13T12:00:00-07:00"
OBSERVED = "2026-09-13T12:00:00.250000-07:00"
SOURCE = {"device_id": "device:dev_macbook", "camera_id": "camera:built_in"}


def _observation(identifier: str, *, captured_at: str = CAPTURED, confidence=0.9):
    return {
        "id": identifier,
        "captured_at": captured_at,
        "observed_at": OBSERVED,
        "source": SOURCE,
        "detector_belief": {
            "category": "mug",
            "confidence": confidence,
            "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.5},
        },
        "producer_profile_id": "vision_producer:mac_v1",
    }


def _clip(state="pending", **fields):
    payload = {
        "id": "evidence_clip:clip_1",
        "state": state,
        "requested_at": "2026-09-13T12:00:01-07:00",
        "start_time": "2026-09-13T11:59:58-07:00",
        "end_time": "2026-09-13T12:00:03-07:00",
        "source": SOURCE,
        "observation_ids": ["visual_observation:mug_1"],
    }
    if state == "available":
        payload["artifact"] = {
            "ref": f"sha256:{'b' * 64}.mp4",
            "kind": "video_clip",
            "media_type": "video/mp4",
            "byte_length": 4096,
        }
        payload["resolved_at"] = "2026-09-13T12:00:04-07:00"
    if state == "failed":
        payload["failure_code"] = "encoder_failed"
        payload["resolved_at"] = "2026-09-13T12:00:04-07:00"
    payload.update(fields)
    return parse_evidence_clip(payload)


class MemoryObservationRepository:
    def __init__(self) -> None:
        self.records: dict[str, ObservationIngestRecord] = {}

    async def put_if_absent(self, record: ObservationIngestRecord):
        previous = self.records.get(record.observation.id)
        if previous is None:
            self.records[record.observation.id] = record
            return "accepted"
        if previous.content_sha256 == record.content_sha256:
            return "duplicate"
        return "conflict"


@pytest.mark.asyncio
async def test_at_least_once_retry_is_idempotent_and_conflicts_are_visible() -> None:
    repository = MemoryObservationRepository()
    instants = iter([
        datetime(2026, 9, 13, 19, 0, 1, tzinfo=timezone.utc),
        datetime(2026, 9, 13, 19, 0, 2, tzinfo=timezone.utc),
        datetime(2026, 9, 13, 19, 0, 3, tzinfo=timezone.utc),
    ])
    service = ObservationIngestionService(repository, clock=lambda: next(instants))
    payload = _observation("visual_observation:mug_1")

    accepted = await service.ingest(payload)
    duplicate = await service.ingest(payload)
    conflict = await service.ingest({**payload, "detector_belief": {
        **payload["detector_belief"], "confidence": 0.2,
    }})

    assert accepted.disposition == "accepted"
    assert duplicate.disposition == "duplicate"
    assert conflict.disposition == "conflict"
    assert accepted.received_at != duplicate.received_at
    stored = repository.records[accepted.observation_id]
    assert stored.received_at == accepted.received_at
    assert stored.observation.captured_at == CAPTURED


@pytest.mark.asyncio
async def test_out_of_order_delivery_is_accepted_without_rewriting_source_time() -> None:
    repository = MemoryObservationRepository()
    service = ObservationIngestionService(repository)
    later = parse_visual_observation(_observation(
        "visual_observation:later",
        captured_at="2026-09-13T12:00:00.100000-07:00",
    ))
    earlier = parse_visual_observation(_observation(
        "visual_observation:earlier",
        captured_at=CAPTURED,
    ))

    receipts = await service.ingest_many((later, earlier))

    assert [receipt.disposition for receipt in receipts] == ["accepted", "accepted"]
    assert repository.records[earlier.id].observation.captured_at == CAPTURED


@pytest.mark.asyncio
async def test_jsonl_replay_uses_the_same_ingestion_and_idempotency_path() -> None:
    repository = MemoryObservationRepository()
    service = ObservationIngestionService(repository)
    observations = (
        parse_visual_observation(_observation("visual_observation:mug_1")),
        parse_visual_observation(_observation("visual_observation:mug_2")),
    )
    frozen = dumps_observations_jsonl(observations)

    first = await service.replay_jsonl(frozen)
    second = await service.replay_jsonl(frozen)

    assert [result.disposition for result in first] == ["accepted", "accepted"]
    assert [result.disposition for result in second] == ["duplicate", "duplicate"]
    assert len(repository.records) == 2


@pytest.mark.asyncio
async def test_ingestion_clock_must_be_timezone_aware() -> None:
    service = ObservationIngestionService(
        MemoryObservationRepository(),
        clock=lambda: datetime(2026, 9, 13, 12, 0),
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        await service.ingest(_observation("visual_observation:mug_1"))


def test_clip_transition_is_monotonic_idempotent_and_allows_association_growth() -> None:
    pending = _clip()
    expanded = _clip(observation_ids=[
        "visual_observation:mug_1",
        "visual_observation:mug_2",
    ])
    available = _clip("available", observation_ids=expanded.observation_ids)

    assert evaluate_evidence_clip_write(None, pending) == "accepted"
    assert evaluate_evidence_clip_write(pending, pending) == "duplicate"
    assert evaluate_evidence_clip_write(pending, expanded) == "accepted"
    assert evaluate_evidence_clip_write(expanded, available) == "accepted"
    assert evaluate_evidence_clip_write(available, available) == "duplicate"
    assert evaluate_evidence_clip_write(available, _clip("failed")) == "conflict"


def test_clip_cannot_skip_pending_or_rewrite_window() -> None:
    pending = _clip()
    changed_window = _clip(end_time="2026-09-13T12:00:10-07:00")

    assert evaluate_evidence_clip_write(None, _clip("available")) == "conflict"
    assert evaluate_evidence_clip_write(pending, changed_window) == "conflict"
