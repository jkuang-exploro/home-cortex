"""Deterministic writes for persistent bodies and their single assignment edge.

Only trusted Home Cortex code calls this service. Device session and telemetry
routes have no access to these operations.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, Mapping

from surrealdb import RecordID
from surrealdb.errors import NotFoundError

from ..agents.embodiments import EmbodimentCatalog
from ..agents.registry import list_agents
from ..persistence.db import Database
from ..persistence.record_ids import as_record_id, canonical_record_id, implicit_edge_component
from ..spatial.embodiment import Embodiment, _typed_id, embodiment_as_mapping, embodiment_from_mapping
from ..spatial.primitives import SpatialContractError


class EmbodimentWritingService:
    """SurrealDB is the sole owner of identity, geometry, capabilities, and links."""

    def __init__(self, database: Database) -> None:
        self.database = database
        self._lock = asyncio.Lock()

    async def ensure_constraints(self) -> None:
        """Enforce one agent per embodiment even for concurrent/direct DB writers."""
        await self.database.query(
            "DEFINE INDEX IF NOT EXISTS assigned_to_one_agent "
            "ON TABLE assigned_to FIELDS in UNIQUE;"
        )

    async def list(self) -> tuple[Embodiment, ...]:
        nodes = await self._rows("embodiment")
        edges = await self._rows("assigned_to")
        assignments: dict[str, str] = {}
        for edge in edges:
            body_id = canonical_record_id(edge["in"])
            agent_id = canonical_record_id(edge["out"])
            _typed_id(body_id, "embodiment", "assigned_to.in")
            _typed_id(agent_id, "agent", "assigned_to.out")
            if body_id in assignments:
                raise SpatialContractError(f"multiple assignments for {body_id!r}")
            assignments[body_id] = agent_id
        bodies: list[Embodiment] = []
        for node in nodes:
            mapping = dict(node)
            mapping["id"] = canonical_record_id(mapping["id"])
            if "agent_id" in mapping:
                raise SpatialContractError("legacy embodiment.agent_id must be migrated to assigned_to")
            body = embodiment_from_mapping(mapping)
            bodies.append(replace(body, agent_id=assignments.pop(body.id, None)))
        if assignments:
            raise SpatialContractError("assigned_to references an unknown embodiment")
        return tuple(sorted(bodies, key=lambda body: body.id))

    async def get(self, embodiment_id: str) -> Embodiment:
        _typed_id(embodiment_id, "embodiment", "embodiment_id")
        for body in await self.list():
            if body.id == embodiment_id:
                return body
        raise SpatialContractError(f"unknown embodiment {embodiment_id!r}")

    async def catalog(self) -> EmbodimentCatalog:
        agent_ids = [canonical_record_id(row["id"]) for row in await self._rows("agent")]
        return EmbodimentCatalog(await self.list(), known_agent_ids=agent_ids)

    async def create(self, value: Embodiment | Mapping[str, Any]) -> Embodiment:
        body = self._unassigned_body(value)
        async with self._lock:
            if any(existing.id == body.id for existing in await self.list()):
                raise SpatialContractError(f"embodiment {body.id!r} already exists")
            await self.database.query(
                "BEGIN TRANSACTION; "
                "IF (SELECT VALUE id FROM ONLY $body) != NONE { THROW 'EMBODIMENT_EXISTS'; }; "
                "CREATE $body CONTENT $content; COMMIT TRANSACTION;",
                {"body": as_record_id(body.id), "content": self._content(body)},
            )
        return body

    async def update(self, value: Embodiment | Mapping[str, Any]) -> Embodiment:
        body = self._unassigned_body(value)
        async with self._lock:
            await self.get(body.id)
            await self.database.query(
                "BEGIN TRANSACTION; "
                "IF (SELECT VALUE id FROM ONLY $body) = NONE { THROW 'EMBODIMENT_MISSING'; }; "
                "UPDATE $body CONTENT $content; COMMIT TRANSACTION;",
                {"body": as_record_id(body.id), "content": self._content(body)},
            )
        return await self.get(body.id)

    async def delete(self, embodiment_id: str) -> None:
        _typed_id(embodiment_id, "embodiment", "embodiment_id")
        body = as_record_id(embodiment_id)
        async with self._lock:
            await self.get(embodiment_id)
            await self.database.query(
                "BEGIN TRANSACTION; "
                "IF (SELECT VALUE id FROM ONLY $body) = NONE { THROW 'EMBODIMENT_MISSING'; }; "
                "DELETE assigned_to WHERE in = $body; DELETE $body; COMMIT TRANSACTION;",
                {"body": body},
            )

    async def assign(self, embodiment_id: str, agent_id: str) -> Embodiment:
        _typed_id(embodiment_id, "embodiment", "embodiment_id")
        _typed_id(agent_id, "agent", "agent_id")
        body = as_record_id(embodiment_id)
        agent_record = as_record_id(agent_id)
        # The edge ID is a function of the embodiment alone. Concurrent writers
        # cannot create two independent assignments for one body.
        edge = RecordID("assigned_to", implicit_edge_component(body))
        async with self._lock:
            current = await self.get(embodiment_id)
            if not any(canonical_record_id(row["id"]) == agent_id
                       for row in await self._rows("agent")):
                raise SpatialContractError(f"unknown controlling agent {agent_id!r}")
            if current.agent_id is not None:
                if current.agent_id != agent_id:
                    raise SpatialContractError(
                        f"embodiment {embodiment_id!r} is already assigned to {current.agent_id!r}"
                    )
                return current
            await self.database.query(
                "BEGIN TRANSACTION; "
                "IF (SELECT VALUE id FROM ONLY $body) = NONE { THROW 'EMBODIMENT_MISSING'; }; "
                "IF (SELECT VALUE id FROM ONLY $agent) = NONE { THROW 'AGENT_MISSING'; }; "
                "LET $current = SELECT VALUE out FROM assigned_to WHERE in = $body; "
                "IF array::len($current) > 1 { THROW 'MULTIPLE_ASSIGNMENTS'; }; "
                "IF array::len($current) = 1 AND $current[0] != $agent { THROW 'ASSIGNMENT_CONFLICT'; }; "
                "IF array::len($current) = 0 { RELATE $body->$edge->$agent CONTENT {}; }; "
                "COMMIT TRANSACTION;",
                {"body": body, "agent": agent_record, "edge": edge},
            )
        return await self.get(embodiment_id)

    async def unassign(self, embodiment_id: str) -> Embodiment:
        _typed_id(embodiment_id, "embodiment", "embodiment_id")
        body = as_record_id(embodiment_id)
        async with self._lock:
            await self.database.query(
                "BEGIN TRANSACTION; "
                "IF (SELECT VALUE id FROM ONLY $body) = NONE { THROW 'EMBODIMENT_MISSING'; }; "
                "DELETE assigned_to WHERE in = $body; COMMIT TRANSACTION;",
                {"body": body},
            )
        return await self.get(embodiment_id)

    async def ensure_registered_agents(self) -> None:
        """Materialize configured agent identities as graph nodes, preserving edits."""
        for agent in list_agents():
            await self.database.query(
                "IF (SELECT VALUE id FROM ONLY $agent) = NONE { "
                "CREATE $agent CONTENT $content; };",
                {"agent": as_record_id(agent.entity_id),
                 "content": {"name": {"en": agent.display_name}, "agent_type": agent.id}},
            )

    async def _rows(self, table: str) -> list[dict[str, Any]]:
        try:
            return await self.database.query(
                "SELECT * FROM type::table($table);", {"table": table},
            ) or []
        except NotFoundError as error:
            if f"'{table}'" not in str(error):
                raise
            return []

    @staticmethod
    def _unassigned_body(value: Embodiment | Mapping[str, Any]) -> Embodiment:
        if isinstance(value, Mapping) and "agent_id" in value:
            raise SpatialContractError("assignment must use assign(), not embodiment.agent_id")
        body = value if isinstance(value, Embodiment) else embodiment_from_mapping(value)
        if body.agent_id is not None:
            raise SpatialContractError("assignment must use assign(), not embodiment.agent_id")
        if body.embodiment_type == "unspecified":
            raise SpatialContractError("embodiment_type is required for persistent writes")
        return body

    @staticmethod
    def _content(body: Embodiment) -> dict[str, Any]:
        return {key: value for key, value in embodiment_as_mapping(body).items() if key != "id"}
