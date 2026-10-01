"""Read projections of persistent bodies, plus ephemeral runtime sessions.

The catalog mirrors a SurrealDB read. ``agent_id`` is the ``assigned_to`` edge,
not a value the session may set. Connection state lives only in
``EmbodimentConnections``. Model-facing tools in ``capabilities.catalog`` are a
separate policy.
"""
from __future__ import annotations

import secrets
from collections.abc import Collection
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from threading import Lock

from ..spatial.embodiment import Embodiment, _typed_id
from ..spatial.primitives import SpatialContractError
from ..spatial.telemetry import _measured_at
from .registry import list_agents


class EmbodimentCatalog:
    """Cached SurrealDB read of body records and their singular assignment."""

    def __init__(
        self,
        embodiments: Collection[Embodiment] = (),
        *,
        known_agent_ids: Collection[str] | None = None,
    ) -> None:
        agents = (known_agent_ids if known_agent_ids is not None
                  else (agent.entity_id for agent in list_agents()))
        self.known_agent_ids = frozenset(
            _typed_id(agent_id, "agent", "agent_id") for agent_id in agents
        )
        records: dict[str, Embodiment] = {}
        for embodiment in embodiments:
            if not isinstance(embodiment, Embodiment):
                raise SpatialContractError("catalog entries must be Embodiment records")
            if embodiment.id in records:
                raise SpatialContractError(f"duplicate embodiment {embodiment.id!r}")
            if embodiment.agent_id is not None and embodiment.agent_id not in self.known_agent_ids:
                raise SpatialContractError(f"unknown controlling agent {embodiment.agent_id!r}")
            records[embodiment.id] = embodiment
        self._records = records

    @property
    def embodiment_ids(self) -> frozenset[str]:
        return frozenset(self._records)

    def get(self, embodiment_id: str) -> Embodiment:
        _typed_id(embodiment_id, "embodiment", "embodiment_id")
        try:
            return self._records[embodiment_id]
        except KeyError as error:
            raise SpatialContractError(f"unknown embodiment {embodiment_id!r}") from error

    def bodies_for_agent(self, agent_id: str) -> tuple[Embodiment, ...]:
        self._require_agent(agent_id)
        return tuple(record for record in self._ordered()
                     if record.agent_id == agent_id)

    def with_capability(self, capability: str) -> tuple[Embodiment, ...]:
        return tuple(record for record in self._ordered()
                     if capability in record.capabilities)

    def named(self, name: str) -> tuple[Embodiment, ...]:
        return tuple(record for record in self._ordered() if record.name == name)

    def cache_resolved(self, body: Embodiment) -> None:
        """Remember one SurrealDB read. This does not write a body or an assignment."""
        if not isinstance(body, Embodiment):
            raise SpatialContractError("catalog entries must be Embodiment records")
        if body.agent_id is not None:
            self.known_agent_ids = self.known_agent_ids | {
                _typed_id(body.agent_id, "agent", "embodiment.agent_id")
            }
        self._records[body.id] = body

    def _ordered(self) -> tuple[Embodiment, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def _require_agent(self, agent_id: str) -> None:
        _typed_id(agent_id, "agent", "agent_id")
        if agent_id not in self.known_agent_ids:
            raise SpatialContractError(f"unknown controlling agent {agent_id!r}")


class SessionProtocolError(SpatialContractError):
    """A session transition that must not create or rewrite an embodiment."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RuntimeSession:
    """One temporary connection. The embodiment ID lives on the durable record."""

    embodiment_id: str
    session_id: str
    online: bool
    connected_at: datetime
    last_seen: datetime
    available_capabilities: frozenset[str]


@dataclass(frozen=True)
class SessionChange:
    disposition: str
    session: RuntimeSession


class EmbodimentConnections:
    """One in-memory runtime session per embodiment. Facts stay in the catalog."""

    def __init__(self, catalog: EmbodimentCatalog) -> None:
        self.catalog = catalog
        self._sessions: dict[str, RuntimeSession] = {}
        self._sequence = 0
        self._instance_id = secrets.token_hex(16)
        self._lock = Lock()

    def connect(
        self,
        embodiment_id: str,
        *,
        available_capabilities: Collection[str] | None = None,
        now: datetime | None = None,
    ) -> SessionChange:
        """Open a session. A second connect while online replaces that session."""
        body = self.catalog.get(embodiment_id)
        active = self._subset(body, available_capabilities)
        observed = _observed(now)
        with self._lock:
            previous = self._sessions.get(embodiment_id)
            disposition = "replaced" if previous is not None and previous.online else "connected"
            self._sequence += 1
            session = RuntimeSession(
                embodiment_id, f"runtime-session:{self._instance_id}:{self._sequence}",
                True, observed, observed, active,
            )
            self._sessions[embodiment_id] = session
            return SessionChange(disposition, session)

    def update_capabilities(
        self,
        embodiment_id: str,
        available_capabilities: Collection[str],
        *,
        now: datetime | None = None,
    ) -> SessionChange:
        body = self.catalog.get(embodiment_id)
        active = self._subset(body, available_capabilities)
        return self._touch(
            embodiment_id, now, "capabilities",
            available_capabilities=active,
        )

    def heartbeat(self, embodiment_id: str, *, now: datetime | None = None) -> SessionChange:
        self.catalog.get(embodiment_id)
        return self._touch(embodiment_id, now, "heartbeat")

    def disconnect(self, embodiment_id: str, *, now: datetime | None = None) -> SessionChange:
        self.catalog.get(embodiment_id)
        observed = _observed(now)
        with self._lock:
            previous = self._require_online(embodiment_id)
            if not previous.online:
                return SessionChange("disconnected", previous)
            session = replace(
                previous, online=False, last_seen=observed, available_capabilities=frozenset(),
            )
            self._sessions[embodiment_id] = session
            return SessionChange("disconnected", session)

    def current(self, embodiment_id: str) -> RuntimeSession | None:
        """Return the ephemeral session, if this process has one.

        Absence means offline. It does not decide whether the embodiment exists
        or which agent SurrealDB has assigned.
        """
        parsed = _typed_id(embodiment_id, "embodiment", "embodiment_id")
        with self._lock:
            return self._sessions.get(parsed)

    def is_connected(self, embodiment_id: str) -> bool:
        session = self.current(embodiment_id)
        return session is not None and session.online

    def active_capabilities(self, embodiment_id: str) -> frozenset[str]:
        session = self.current(embodiment_id)
        if session is None or not session.online:
            return frozenset()
        return session.available_capabilities

    def active_bodies_for_agent(self, agent_id: str) -> tuple[Embodiment, ...]:
        bodies = self.catalog.bodies_for_agent(agent_id)
        with self._lock:
            return tuple(
                body for body in bodies
                if (session := self._sessions.get(body.id)) is not None and session.online
            )

    def is_currently_embodied(self, agent_id: str) -> bool:
        return bool(self.active_bodies_for_agent(agent_id))

    def _subset(self, body: Embodiment, available: Collection[str] | None) -> frozenset[str]:
        if available is None:
            return frozenset(body.capabilities)
        if isinstance(available, str) or not isinstance(available, Collection):
            raise SessionProtocolError(
                "invalid_capabilities", "available capabilities must be a list of names",
            )
        active = frozenset(available)
        if any(not isinstance(name, str) for name in active):
            raise SessionProtocolError(
                "invalid_capabilities", "available capabilities must be a list of names",
            )
        if not active.issubset(body.capabilities):
            raise SessionProtocolError(
                "capability_not_configured",
                "runtime capabilities must be configured on the embodiment",
            )
        return active

    def _touch(
        self,
        embodiment_id: str,
        now: datetime | None,
        disposition: str,
        *,
        available_capabilities: frozenset[str] | None = None,
    ) -> SessionChange:
        observed = _observed(now)
        with self._lock:
            previous = self._require_online(embodiment_id)
            if not previous.online:
                raise SessionProtocolError("session_offline", "runtime session is offline")
            session = replace(
                previous,
                last_seen=observed,
                available_capabilities=(
                    previous.available_capabilities
                    if available_capabilities is None else available_capabilities
                ),
            )
            self._sessions[embodiment_id] = session
            return SessionChange(disposition, session)

    def _require_online(self, embodiment_id: str) -> RuntimeSession:
        previous = self._sessions.get(embodiment_id)
        if previous is None:
            raise SessionProtocolError("session_offline", "runtime session is offline")
        return previous


def _observed(now: datetime | None) -> datetime:
    return _measured_at(now if now is not None else datetime.now(timezone.utc))
