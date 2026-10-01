"""Read projection and deterministic physical-action availability for embodiments.

Persistent fields come from the identity reader, which production binds to
SurrealDB. Connection, availability, and telemetry come from this process.
The combined document is not written back.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .embodiments import EmbodimentConnections
from .registry import get_agent_by_entity_id, UnknownAgentError
from ..spatial.embodiment import (
    Embodiment, _typed_id, body_geometry_as_mapping, local_frame_as_mapping,
)
from ..spatial.presence import EmbodimentPresence
from ..spatial.primitives import SpatialContractError
from ..spatial.telemetry import physical_telemetry_as_mapping


class PersistentEmbodiments(Protocol):
    """SurrealDB-backed body reads. A runtime session is not a member of this set."""

    async def list(self) -> tuple[Embodiment, ...]:
        """Return every persistent body."""

    async def get(self, embodiment_id: str) -> Embodiment:
        """Return one persistent body, or raise when it is absent."""


@dataclass(frozen=True)
class ActionAvailability:
    available: bool
    code: str
    embodiment_id: str
    agent_id: str
    capability: str

    def as_mapping(self) -> dict[str, Any]:
        return {"available": self.available, "code": self.code,
                "embodiment_id": self.embodiment_id, "agent_id": self.agent_id,
                "capability": self.capability}


class EmbodimentDirectory:
    """Compose a SurrealDB body with the optional runtime session.

    ``linked`` follows the persistent assignment. ``connected`` follows the
    session. Telemetry freshness never implies either one. Nothing here is saved.
    """

    def __init__(self, identities: PersistentEmbodiments, connections: EmbodimentConnections,
                 presence: EmbodimentPresence) -> None:
        self.identities = identities
        self.connections = connections
        self.presence = presence

    async def list(self) -> list[dict[str, Any]]:
        return [self._view(body, detail=False) for body in await self.identities.list()]

    async def list_for_agent(self, agent_id: str) -> list[dict[str, Any]]:
        parsed = _typed_id(agent_id, "agent", "agent_id")
        return [self._view(body, detail=False) for body in await self.identities.list()
                if body.agent_id == parsed]

    async def get(self, embodiment_id: str, *, detail: bool = True) -> dict[str, Any]:
        return self._view(await self.identities.get(embodiment_id), detail=detail)

    async def validate_selection(self, agent_id: str, embodiment_id: str) -> None:
        """Accept a conversation selection from the persistent assignment alone."""
        body = await self.identities.get(embodiment_id)
        if body.agent_id != agent_id:
            raise SpatialContractError("embodiment is not linked to this agent")

    async def action_availability(self, agent_id: str, embodiment_id: str,
                                  capability: str) -> ActionAvailability:
        """Gate one physical capability. Each failure names a separate condition."""
        try:
            body = await self.identities.get(embodiment_id)
        except SpatialContractError:
            code = "unknown_embodiment"
        else:
            if body.agent_id != agent_id:
                code = "embodiment_not_linked"
            elif not self.connections.is_connected(body.id):
                code = "embodiment_offline"
            elif capability not in body.capabilities:
                code = "unsupported_capability"
            elif capability not in self.connections.active_capabilities(body.id):
                code = "capability_unavailable"
            else:
                code = "available"
        return ActionAvailability(code == "available", code, embodiment_id, agent_id, capability)

    def _view(self, body: Embodiment, *, detail: bool) -> dict[str, Any]:
        session = self.connections.current(body.id)
        online = session is not None and session.online
        state = self.presence.reading(body.id)
        sample = state.telemetry
        try:
            agent = get_agent_by_entity_id(body.agent_id) if body.agent_id else None
        except UnknownAgentError:
            agent = None
        advertised = session.available_capabilities if online and session is not None else frozenset()
        result: dict[str, Any] = {
            "id": body.id,
            "name": body.name,
            "embodiment_type": body.embodiment_type,
            "agent": (None if body.agent_id is None else
                      {"id": body.agent_id,
                       "name": agent.display_name if agent else None}),
            "linked": body.agent_id is not None,
            "connected": online,
            "state": ("unlinked" if body.agent_id is None else
                      "linked_online" if online else "linked_offline"),
            "runtime": {
                "connected_at": session.connected_at.isoformat() if session else None,
                "last_seen": session.last_seen.isoformat() if session else None,
            },
            "telemetry": {
                "available": bool(online and state.status.fresh),
                "valid": state.status.valid,
                "fresh": state.status.fresh,
                "space_id": sample.space_id if sample else None,
                "measured_at": sample.measured_at.isoformat() if sample else None,
            },
            "capabilities": [
                {"name": name, "supported": True, "available": name in advertised}
                for name in body.capabilities
            ],
        }
        if detail:
            result["geometry"] = (body_geometry_as_mapping(body.geometry)
                                  if body.geometry is not None else None)
            result["local_frame"] = (local_frame_as_mapping(body.local_frame)
                                     if body.local_frame is not None else None)
            result["telemetry"]["estimate"] = (
                physical_telemetry_as_mapping(sample) if sample is not None else None
            )
        return result
