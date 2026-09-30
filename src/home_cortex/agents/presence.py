"""Read projection and deterministic physical-action availability for embodiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .embodiments import EmbodimentCatalog, EmbodimentConnections
from .registry import get_agent_by_entity_id, UnknownAgentError
from ..spatial.embodiment import body_geometry_as_mapping, local_frame_as_mapping
from ..spatial.presence import EmbodimentPresence
from ..spatial.primitives import SpatialContractError
from ..spatial.telemetry import physical_telemetry_as_mapping


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
    """Compose durable body records, session presence, and latest fused telemetry.

    Connection status comes only from the runtime session. Telemetry validity and
    freshness are independent observations and never imply connectivity.
    """

    def __init__(self, catalog: EmbodimentCatalog, connections: EmbodimentConnections,
                 presence: EmbodimentPresence) -> None:
        self.catalog = catalog
        self.connections = connections
        self.presence = presence

    def list(self) -> list[dict[str, Any]]:
        return [self.get(body_id, detail=False) for body_id in sorted(self.catalog.embodiment_ids)]

    def list_for_agent(self, agent_id: str) -> list[dict[str, Any]]:
        return [self.get(body.id, detail=False)
                for body in self.catalog.bodies_for_agent(agent_id)]

    def get(self, embodiment_id: str, *, detail: bool = True) -> dict[str, Any]:
        body = self.catalog.get(embodiment_id)
        session = self.connections.current(body.id)
        online = session is not None and session.online
        state = self.presence.latest(body.id)
        sample = state.telemetry
        try:
            agent = get_agent_by_entity_id(body.agent_id) if body.agent_id else None
        except UnknownAgentError:
            agent = None
        result: dict[str, Any] = {
            "id": body.id,
            "name": body.name,
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
                {"name": name, "supported": True,
                 "available": bool(online and name in session.available_capabilities)}
                for name in body.capabilities
            ],
        }
        if detail:
            result["geometry"] = body_geometry_as_mapping(body.geometry)
            result["local_frame"] = local_frame_as_mapping(body.local_frame)
            result["telemetry"]["estimate"] = (
                physical_telemetry_as_mapping(sample) if sample is not None else None
            )
        return result

    def validate_selection(self, agent_id: str, embodiment_id: str) -> None:
        body = self.catalog.get(embodiment_id)
        if body.agent_id != agent_id:
            raise SpatialContractError("embodiment is not linked to this agent")

    def action_availability(self, agent_id: str, embodiment_id: str,
                            capability: str) -> ActionAvailability:
        try:
            body = self.catalog.get(embodiment_id)
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
