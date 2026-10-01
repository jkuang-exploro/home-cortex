"""Latest client-fused embodiment telemetry. Samples are not a fact history.

The caller registers durable embodiment and space IDs. Each submission is one
``PhysicalTelemetry`` value from Ticket 1. A newer ``measured_at`` replaces the
current sample, including a newer ``no_estimate``. An older sample is ignored.
The same sample may be submitted again. A different sample with the same
measurement time is rejected and leaves the current sample in place.
"""
from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .embodiment import Embodiment, _typed_id, embodiment_from_mapping
from .primitives import SpatialContractError
from .telemetry import (
    PhysicalTelemetry, _finite, _measured_at, physical_telemetry_as_mapping,
    physical_telemetry_from_mapping,
)

# Observer window for derived freshness only. It is not a localization label.
DEFAULT_OBSERVER_STALE_AFTER_S = 2.0


class TelemetryAdmissionError(SpatialContractError):
    """A well-formed sample that must not change the current sample."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class TelemetryStatus:
    """Derived from the current sample, the observer clock, and the age window."""

    valid: bool
    unavailable: bool
    fresh: bool
    stale: bool
    stale_after_s: float

    def __post_init__(self) -> None:
        if self.unavailable == self.valid or self.fresh and self.stale:
            raise SpatialContractError("telemetry status flags are inconsistent")
        if (self.fresh or self.stale) and not self.valid:
            raise SpatialContractError("only a valid sample can be fresh or stale")
        if self.stale_after_s < 0:
            raise SpatialContractError("stale_after_s must be non-negative")


@dataclass(frozen=True)
class EmbodimentTelemetryState:
    """Current sample for one registered embodiment, plus derived status."""

    embodiment_id: str
    telemetry: PhysicalTelemetry | None
    status: TelemetryStatus


@dataclass(frozen=True)
class TelemetryAdmission:
    """Result of one submission. ``ignored`` keeps the newer current sample."""

    disposition: str
    state: EmbodimentTelemetryState

    def __post_init__(self) -> None:
        if self.disposition not in {"accepted", "duplicate", "ignored"}:
            raise SpatialContractError("unknown telemetry disposition")


def derive_telemetry_status(
    telemetry: PhysicalTelemetry | None, *, now: datetime, stale_after_s: float,
) -> TelemetryStatus:
    """``valid`` / ``unavailable`` follow the sample; ``fresh`` / ``stale`` follow age.

    A missing sample and ``no_estimate`` are unavailable. A measurement timestamp
    ahead of the observer clock stays valid and is neither fresh nor stale.
    ``p95`` limits stay with ``estimate_within_limits``; this function does not
    invent a position or orientation threshold.
    """
    window = _finite(stale_after_s, "stale_after_s")
    if window < 0:
        raise SpatialContractError("stale_after_s must be non-negative")
    current = _measured_at(now)
    if telemetry is None or telemetry.validity != "valid" or telemetry.transform is None:
        return TelemetryStatus(False, True, False, False, window)
    age = (current - telemetry.measured_at).total_seconds()
    fresh = 0 <= age <= window
    return TelemetryStatus(True, False, fresh, age > window, window)


def telemetry_state_as_mapping(state: EmbodimentTelemetryState) -> dict[str, Any]:
    return {
        "embodiment_id": state.embodiment_id,
        "telemetry": (
            None if state.telemetry is None else physical_telemetry_as_mapping(state.telemetry)
        ),
        "status": {
            "valid": state.status.valid,
            "unavailable": state.status.unavailable,
            "fresh": state.status.fresh,
            "stale": state.status.stale,
            "stale_after_s": state.status.stale_after_s,
        },
    }


def admission_as_mapping(admission: TelemetryAdmission) -> dict[str, Any]:
    return {
        "disposition": admission.disposition,
        "state": telemetry_state_as_mapping(admission.state),
    }


def record_ids_from_node_file(path: Path, table: str) -> frozenset[str]:
    """Read durable ``table:`` IDs. A missing file contributes no registrations."""
    return frozenset(_typed_id(item["id"], table, "id") for item in _node_records(path, table))


def embodiments_from_node_file(path: Path) -> tuple[Embodiment, ...]:
    """Load durable embodiment records; a missing file means no bodies."""
    return tuple(embodiment_from_mapping(item) for item in _node_records(path, "embodiment"))


def embodiment_ids_from_node_file(path: Path) -> frozenset[str]:
    """Validate durable embodiment records and return their IDs."""
    return frozenset(item.id for item in embodiments_from_node_file(path))


def _node_records(path: Path, table: str) -> list[Any]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SpatialContractError(f"{table} node file {path} is not valid JSON") from error
    if not isinstance(payload, list):
        raise SpatialContractError(f"{table} node file must be a JSON list")
    for item in payload:
        if not isinstance(item, Mapping) or "id" not in item:
            raise SpatialContractError(f"{table} record must contain id")
    return payload


class EmbodimentPresence:
    """In-memory current telemetry for registered embodiments."""

    def __init__(
        self,
        *,
        embodiment_ids: Collection[str] = (),
        space_ids: Collection[str] = (),
        stale_after_s: float = DEFAULT_OBSERVER_STALE_AFTER_S,
    ) -> None:
        window = _finite(stale_after_s, "stale_after_s")
        if window < 0:
            raise SpatialContractError("stale_after_s must be non-negative")
        self.stale_after_s = window
        self._embodiments: set[str] = set()
        self._spaces: set[str] = set()
        self._current: dict[str, PhysicalTelemetry] = {}
        self._lock = Lock()
        for embodiment_id in embodiment_ids:
            self.register_embodiment(embodiment_id)
        for space_id in space_ids:
            self.register_space(space_id)

    def register_embodiment(self, embodiment_id: str) -> None:
        parsed = _typed_id(embodiment_id, "embodiment", "embodiment_id")
        with self._lock:
            self._embodiments.add(parsed)

    def register_space(self, space_id: str) -> None:
        parsed = _typed_id(space_id, "space", "space_id")
        with self._lock:
            self._spaces.add(parsed)

    def submit(
        self,
        sample: Mapping[str, Any] | PhysicalTelemetry,
        *,
        now: datetime | None = None,
    ) -> TelemetryAdmission:
        parsed = (
            sample if isinstance(sample, PhysicalTelemetry)
            else physical_telemetry_from_mapping(sample)
        )
        observed = _measured_at(now if now is not None else datetime.now(timezone.utc))
        with self._lock:
            self._require_embodiment(parsed.embodiment_id)
            if parsed.space_id is not None and parsed.space_id not in self._spaces:
                raise TelemetryAdmissionError(
                    "unknown_space",
                    "Telemetry space is not a registered space",
                )
            current = self._current.get(parsed.embodiment_id)
            if current is None or parsed.measured_at > current.measured_at:
                self._current[parsed.embodiment_id] = parsed
                disposition = "accepted"
                stored = parsed
            elif parsed == current:
                disposition = "duplicate"
                stored = current
            elif parsed.measured_at == current.measured_at:
                raise TelemetryAdmissionError(
                    "conflicting_sample",
                    "A different sample with this measured_at is already current",
                )
            else:
                disposition = "ignored"
                stored = current
            state = self._state(parsed.embodiment_id, stored, observed)
        return TelemetryAdmission(disposition, state)

    def reading(self, embodiment_id: str, *, now: datetime | None = None) -> EmbodimentTelemetryState:
        """Telemetry for a view. An id this process has not admitted is unavailable.

        This does not register the id and does not write a persistent record.
        """
        observed_at = _measured_at(now if now is not None else datetime.now(timezone.utc))
        parsed = _typed_id(embodiment_id, "embodiment", "embodiment_id")
        with self._lock:
            if parsed not in self._embodiments:
                return self._state(parsed, None, observed_at)
            return self._state(parsed, self._current.get(parsed), observed_at)

    def latest(self, embodiment_id: str, *, now: datetime | None = None) -> EmbodimentTelemetryState:
        observed = _measured_at(now if now is not None else datetime.now(timezone.utc))
        with self._lock:
            parsed = self._require_embodiment(embodiment_id)
            return self._state(parsed, self._current.get(parsed), observed)

    def _require_embodiment(self, embodiment_id: str) -> str:
        parsed = _typed_id(embodiment_id, "embodiment", "embodiment_id")
        if parsed not in self._embodiments:
            raise TelemetryAdmissionError(
                "unknown_embodiment",
                "Embodiment is not registered",
            )
        return parsed

    def _state(
        self,
        embodiment_id: str,
        telemetry: PhysicalTelemetry | None,
        now: datetime,
    ) -> EmbodimentTelemetryState:
        return EmbodimentTelemetryState(
            embodiment_id,
            telemetry,
            derive_telemetry_status(telemetry, now=now, stale_after_s=self.stale_after_s),
        )
