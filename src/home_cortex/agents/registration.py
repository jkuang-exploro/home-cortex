"""Resolve a persistent embodiment, then open only ephemeral runtime state.

SurrealDB answers which body exists and which agent it is assigned to.
The runtime answers whether that body currently has a session. Startup loads
the graph and does not create ``assigned_to`` edges or require a connection.
"""
from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from ..mutation.embodiments import EmbodimentWritingService
from ..spatial.embodiment import Embodiment
from ..spatial.presence import DEFAULT_OBSERVER_STALE_AFTER_S, EmbodimentPresence
from ..spatial.primitives import SpatialContractError
from ..spatial.telemetry import PhysicalTelemetry
from .embodiments import EmbodimentCatalog, EmbodimentConnections, SessionProtocolError
from .presence import EmbodimentDirectory
from .session import EmbodimentSession


class EmbodimentIdentityReader(Protocol):
    """Read persistent bodies. Implementations must not assign or create them."""

    async def list(self) -> tuple[Embodiment, ...]:
        """Return every stored body, including those with no runtime session."""

    async def get(self, embodiment_id: str) -> Embodiment:
        """Return the stored body, or raise when the id is absent."""


class CatalogIdentityReader:
    """Read a preloaded projection. Used when a test already holds that read."""

    def __init__(self, catalog: EmbodimentCatalog) -> None:
        self.catalog = catalog

    async def list(self) -> tuple[Embodiment, ...]:
        return tuple(
            self.catalog.get(embodiment_id)
            for embodiment_id in sorted(self.catalog.embodiment_ids)
        )

    async def get(self, embodiment_id: str) -> Embodiment:
        return self.catalog.get(embodiment_id)


@dataclass
class EmbodimentRuntime:
    """Process-local handles composed from one SurrealDB read at startup."""

    writing: EmbodimentWritingService
    catalog: EmbodimentCatalog
    connections: EmbodimentConnections
    presence: EmbodimentPresence
    session: EmbodimentSession
    directory: EmbodimentDirectory
    registration: EmbodimentRegistration


class EmbodimentRegistration:
    """Client entry that resolves identity before touching runtime state."""

    def __init__(self, identities: EmbodimentIdentityReader, session: EmbodimentSession) -> None:
        self.identities = identities
        self.session = session

    async def connect(
        self,
        embodiment_id: str,
        available_capabilities: Collection[str],
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        await self._admit(embodiment_id)
        return self.session.register(
            embodiment_id, available_capabilities, now=now,
        )

    async def update_capabilities(
        self,
        embodiment_id: str,
        available_capabilities: Collection[str],
        *,
        session_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        await self._admit(embodiment_id)
        return self.session.update_capabilities(
            embodiment_id, available_capabilities, session_id=session_id, now=now,
        )

    async def heartbeat(
        self, embodiment_id: str, *, session_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        await self._admit(embodiment_id)
        return self.session.heartbeat(embodiment_id, session_id=session_id, now=now)

    async def disconnect(
        self, embodiment_id: str, *, session_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        await self._admit(embodiment_id)
        return self.session.disconnect(embodiment_id, session_id=session_id, now=now)

    async def submit_telemetry(
        self,
        sample: dict[str, Any] | PhysicalTelemetry,
        *,
        session_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        embodiment_id = (
            sample.embodiment_id if isinstance(sample, PhysicalTelemetry)
            else sample.get("embodiment_id")
        )
        if not isinstance(embodiment_id, str):
            raise SessionProtocolError("invalid_telemetry", "embodiment_id is required")
        await self._admit(embodiment_id)
        return self.session.submit_telemetry(sample, session_id=session_id, now=now)

    async def view(self, embodiment_id: str) -> dict[str, Any]:
        await self._admit(embodiment_id)
        return self.session.view(embodiment_id)

    async def _admit(self, embodiment_id: str) -> Embodiment:
        try:
            body = await self.identities.get(embodiment_id)
        except SpatialContractError as error:
            raise SessionProtocolError("unknown_embodiment", str(error)) from error
        self.session.note_resolved(body)
        return body


async def open_embodiment_runtime(
    writing: EmbodimentWritingService,
    *,
    space_ids: Collection[str] = (),
    stale_after_s: float = DEFAULT_OBSERVER_STALE_AFTER_S,
) -> EmbodimentRuntime:
    """Load persistent bodies and start with every session offline.

    Materializing a missing agent node does not relate that agent to a body.
    """
    await writing.ensure_constraints()
    await writing.ensure_registered_agents()
    catalog = await writing.catalog()
    connections = EmbodimentConnections(catalog)
    presence = EmbodimentPresence(
        embodiment_ids=catalog.embodiment_ids,
        space_ids=space_ids,
        stale_after_s=stale_after_s,
    )
    session = EmbodimentSession(catalog, connections, presence)
    return EmbodimentRuntime(
        writing=writing,
        catalog=catalog,
        connections=connections,
        presence=presence,
        session=session,
        directory=EmbodimentDirectory(writing, connections, presence),
        registration=EmbodimentRegistration(writing, session),
    )
