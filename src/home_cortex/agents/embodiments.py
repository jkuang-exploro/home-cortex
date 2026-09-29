"""Agent-to-body association and physical capability lookup.

The embodiment record owns the single persistent ``agent_id`` relationship.
Connection state lives only in ``EmbodimentConnections`` and never changes that
record. Model-facing tools in ``capabilities.catalog`` are a separate policy.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Collection

from ..spatial.embodiment import (
    Embodiment, _typed_id, embodiment_as_mapping,
)
from ..spatial.presence import embodiments_from_node_file
from ..spatial.primitives import SpatialContractError
from .registry import list_agents


class EmbodimentCatalog:
    """Snapshot of durable body records and their singular agent association."""

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

    @classmethod
    def from_node_file(
        cls, path: Path, *, known_agent_ids: Collection[str] | None = None,
    ) -> EmbodimentCatalog:
        return cls(embodiments_from_node_file(path), known_agent_ids=known_agent_ids)

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

    def associate(self, embodiment_id: str, agent_id: str) -> EmbodimentCatalog:
        """Assign an unclaimed body; transfer requires an explicit unassign first."""
        self._require_agent(agent_id)
        current = self.get(embodiment_id)
        if current.agent_id is not None and current.agent_id != agent_id:
            raise SpatialContractError(
                f"embodiment {embodiment_id!r} is already controlled by {current.agent_id!r}"
            )
        return self._replaced(replace(current, agent_id=agent_id))

    def unassign(self, embodiment_id: str) -> EmbodimentCatalog:
        return self._replaced(replace(self.get(embodiment_id), agent_id=None))

    def as_records(self) -> list[dict]:
        return [embodiment_as_mapping(record) for record in self._ordered()]

    def save_node_file(self, path: Path) -> None:
        """Atomically replace the durable JSON source; graph ingestion is separate."""
        payload = json.dumps(self.as_records(), ensure_ascii=False, indent=2) + "\n"
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", delete=False,
            ) as file:
                temp_path = Path(file.name)
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _ordered(self) -> tuple[Embodiment, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def _require_agent(self, agent_id: str) -> None:
        _typed_id(agent_id, "agent", "agent_id")
        if agent_id not in self.known_agent_ids:
            raise SpatialContractError(f"unknown controlling agent {agent_id!r}")

    def _replaced(self, embodiment: Embodiment) -> EmbodimentCatalog:
        records = dict(self._records)
        records[embodiment.id] = embodiment
        return EmbodimentCatalog(records.values(), known_agent_ids=self.known_agent_ids)


class EmbodimentConnections:
    """Ephemeral connection and available-capability state, independent of facts."""

    def __init__(self, catalog: EmbodimentCatalog) -> None:
        self.catalog = catalog
        self._available: dict[str, frozenset[str]] = {}
        self._lock = Lock()

    def connect(
        self, embodiment_id: str, *, available_capabilities: Collection[str] | None = None,
    ) -> None:
        body = self.catalog.get(embodiment_id)
        active = frozenset(
            body.capabilities if available_capabilities is None else available_capabilities
        )
        if not active.issubset(body.capabilities):
            raise SpatialContractError("runtime capabilities must be configured on the embodiment")
        with self._lock:
            self._available[embodiment_id] = active

    def disconnect(self, embodiment_id: str) -> None:
        self.catalog.get(embodiment_id)
        with self._lock:
            self._available.pop(embodiment_id, None)

    def is_connected(self, embodiment_id: str) -> bool:
        self.catalog.get(embodiment_id)
        with self._lock:
            return embodiment_id in self._available

    def active_capabilities(self, embodiment_id: str) -> frozenset[str]:
        self.catalog.get(embodiment_id)
        with self._lock:
            return self._available.get(embodiment_id, frozenset())

    def active_bodies_for_agent(self, agent_id: str) -> tuple[Embodiment, ...]:
        bodies = self.catalog.bodies_for_agent(agent_id)
        with self._lock:
            return tuple(body for body in bodies if body.id in self._available)

    def is_currently_embodied(self, agent_id: str) -> bool:
        return bool(self.active_bodies_for_agent(agent_id))
