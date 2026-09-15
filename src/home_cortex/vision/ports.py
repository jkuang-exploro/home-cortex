"""Transport-neutral ports between an edge producer and Home Cortex.

HTTP, WebSocket, message-stream, and replay adapters terminate outside this
module. They all submit the same canonical contracts through these ports.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

from .contracts import (
    ArtifactReference,
    EvidenceClip,
    ObservationSource,
    VisualObservation,
    dumps_observation,
    parse_observations_jsonl,
    parse_visual_observation,
)

WriteDisposition = Literal["accepted", "duplicate", "conflict"]


@dataclass(frozen=True)
class ObservationIngestRecord:
    """Canonical observation plus backend-owned ingestion metadata."""

    observation: VisualObservation
    received_at: str
    content_sha256: str


@dataclass(frozen=True)
class ObservationReceipt:
    """Per-observation acknowledgement returned to a delivery adapter."""

    observation_id: str
    disposition: WriteDisposition
    received_at: str


@dataclass(frozen=True)
class LiveStreamAdvertisement:
    """Short-lived discovery data for an edge-hosted human-facing stream."""

    source: ObservationSource
    transport: str
    endpoint: str
    expires_at: str | None = None


class ObservationRepository(Protocol):
    """Persistence seam with an atomic uniqueness check on observation ID."""

    async def put_if_absent(
        self, record: ObservationIngestRecord
    ) -> WriteDisposition: ...


class ObservationSink(Protocol):
    """Machine-facing observation delivery seam used by every transport."""

    async def ingest(
        self, value: VisualObservation | Mapping[str, Any]
    ) -> ObservationReceipt: ...


class EvidenceClipSink(Protocol):
    """Metadata channel for idempotent clip creation and state transitions."""

    async def apply(self, clip: EvidenceClip) -> WriteDisposition: ...


class ArtifactSink(Protocol):
    """External byte-storage seam; implementations return domain metadata."""

    async def put(
        self, payload: bytes, *, kind: str, media_type: str
    ) -> ArtifactReference: ...


class LiveStreamDirectory(Protocol):
    """Discovery seam. Home Cortex does not receive or proxy media through it."""

    async def resolve(
        self, source: ObservationSource
    ) -> LiveStreamAdvertisement | None: ...


class ObservationIngestionService:
    """Shared validation, idempotency, receipt-time, and replay path."""

    def __init__(
        self,
        repository: ObservationRepository,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    async def ingest(
        self, value: VisualObservation | Mapping[str, Any]
    ) -> ObservationReceipt:
        observation = (
            value if isinstance(value, VisualObservation)
            else parse_visual_observation(value)
        )
        received_at = _received_at(self.clock())
        canonical = dumps_observation(observation).encode("utf-8")
        record = ObservationIngestRecord(
            observation=observation,
            received_at=received_at,
            content_sha256=hashlib.sha256(canonical).hexdigest(),
        )
        disposition = await self.repository.put_if_absent(record)
        if disposition not in {"accepted", "duplicate", "conflict"}:
            raise ValueError(f"unknown repository disposition: {disposition}")
        return ObservationReceipt(observation.id, disposition, received_at)

    async def ingest_many(
        self, values: Sequence[VisualObservation | Mapping[str, Any]]
    ) -> tuple[ObservationReceipt, ...]:
        return tuple([await self.ingest(value) for value in values])

    async def replay_jsonl(self, text: str) -> tuple[ObservationReceipt, ...]:
        """Replay frozen observations through the identical ingestion path."""

        return await self.ingest_many(parse_observations_jsonl(text))


def evaluate_evidence_clip_write(
    previous: EvidenceClip | None,
    incoming: EvidenceClip,
) -> WriteDisposition:
    """Evaluate the monotonic, idempotent EvidenceClip state machine."""

    if previous is None:
        return "accepted" if incoming.state == "pending" else "conflict"
    if incoming == previous:
        return "duplicate"
    if previous.id != incoming.id or not _same_clip_identity(previous, incoming):
        return "conflict"
    if not set(previous.observation_ids).issubset(incoming.observation_ids):
        return "conflict"
    if previous.state != "pending":
        return "conflict"
    return "accepted"


def _same_clip_identity(previous: EvidenceClip, incoming: EvidenceClip) -> bool:
    return (
        previous.requested_at == incoming.requested_at
        and previous.start_time == incoming.start_time
        and previous.end_time == incoming.end_time
        and previous.source == incoming.source
    )


def _received_at(value: datetime) -> str:
    if value.utcoffset() is None:
        raise ValueError("ingestion clock must return a timezone-aware datetime")
    return value.isoformat(timespec="milliseconds")
