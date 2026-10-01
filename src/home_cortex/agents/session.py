"""Canonical client lifecycle for one embodiment SurrealDB already stores.

The client authenticates at the HTTP boundary and presents an embodiment id.
Registration reads that persistent record; it does not create a body, choose
an agent, or write ``assigned_to``. This module does not estimate sensors.
"""
from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import datetime
from threading import RLock
from typing import Any

from ..spatial.embodiment import Embodiment
from ..spatial.presence import (
    EmbodimentPresence, TelemetryAdmissionError, admission_as_mapping,
)
from ..spatial.primitives import SpatialContractError
from ..spatial.telemetry import PhysicalTelemetry, physical_telemetry_from_mapping
from .embodiments import (
    EmbodimentCatalog, EmbodimentConnections, RuntimeSession, SessionChange,
    SessionProtocolError,
)


class EmbodimentSession:
    """Register, refresh, and disconnect the session that gates telemetry."""

    def __init__(
        self,
        catalog: EmbodimentCatalog,
        connections: EmbodimentConnections,
        presence: EmbodimentPresence,
    ) -> None:
        self.catalog = catalog
        self.connections = connections
        self.presence = presence
        self._lock = RLock()

    def note_resolved(self, body: Embodiment) -> None:
        """Cache a persistent read so later session checks see the same record."""
        with self._lock:
            self.catalog.cache_resolved(body)
            self.presence.register_embodiment(body.id)

    def register(
        self,
        embodiment_id: str,
        available_capabilities: Collection[str],
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        self._require_known(embodiment_id)
        with self._lock:
            change = self.connections.connect(
                embodiment_id, available_capabilities=available_capabilities, now=now,
            )
            return self.view(embodiment_id, change=change)

    def update_capabilities(
        self,
        embodiment_id: str,
        available_capabilities: Collection[str],
        *,
        session_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        self._require_known(embodiment_id)
        with self._lock:
            self._require_current(embodiment_id, session_id)
            change = self.connections.update_capabilities(
                embodiment_id, available_capabilities, now=now,
            )
            return self.view(embodiment_id, change=change)

    def heartbeat(
        self, embodiment_id: str, *, session_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        self._require_known(embodiment_id)
        with self._lock:
            self._require_current(embodiment_id, session_id)
            change = self.connections.heartbeat(embodiment_id, now=now)
            return self.view(embodiment_id, change=change)

    def disconnect(
        self, embodiment_id: str, *, session_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        self._require_known(embodiment_id)
        with self._lock:
            self._require_current(embodiment_id, session_id, allow_offline=True)
            change = self.connections.disconnect(embodiment_id, now=now)
            return self.view(embodiment_id, change=change)

    def submit_telemetry(
        self,
        sample: Mapping[str, Any] | PhysicalTelemetry,
        *,
        session_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        parsed = (
            sample if isinstance(sample, PhysicalTelemetry)
            else physical_telemetry_from_mapping(sample)
        )
        self._require_known(parsed.embodiment_id)
        with self._lock:
            self._require_current(parsed.embodiment_id, session_id)
            try:
                admission = self.presence.submit(parsed, now=now)
            except TelemetryAdmissionError as error:
                raise SessionProtocolError(error.code, str(error)) from error
            self.connections.heartbeat(parsed.embodiment_id, now=now)
            return {
                "telemetry": admission_as_mapping(admission),
                "presence": self.view(parsed.embodiment_id),
            }

    def view(
        self, embodiment_id: str, *, change: SessionChange | None = None,
    ) -> dict[str, Any]:
        body = self._require_known(embodiment_id)
        payload: dict[str, Any] = {
            "embodiment_id": body.id,
            "agent_id": body.agent_id,
            "configured_capabilities": list(body.capabilities),
            "session": _session_mapping(self.connections.current(embodiment_id)),
        }
        if change is not None:
            payload["disposition"] = change.disposition
        return payload

    def _require_known(self, embodiment_id: str):
        try:
            return self.catalog.get(embodiment_id)
        except SpatialContractError as error:
            raise SessionProtocolError("unknown_embodiment", str(error)) from error

    def _require_current(
        self, embodiment_id: str, session_id: str | None,
        *, allow_offline: bool = False,
    ) -> None:
        current = self.connections.current(embodiment_id)
        if current is None or (not current.online and not allow_offline):
            raise SessionProtocolError(
                "session_offline", "operation requires an online runtime session",
            )
        if session_id != current.session_id:
            raise SessionProtocolError("stale_session", "session id is not current")


def _session_mapping(session: RuntimeSession | None) -> dict[str, Any] | None:
    if session is None:
        return None
    return {
        "session_id": session.session_id,
        "online": session.online,
        "connected_at": session.connected_at.isoformat(),
        "last_seen": session.last_seen.isoformat(),
        "available_capabilities": sorted(session.available_capabilities),
    }
