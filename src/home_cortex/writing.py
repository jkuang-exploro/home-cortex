"""Canonical semantic mutations for authoritative household item state.

Callers describe one of three item intents.  This module alone translates those
intents into fixed SurrealQL transactions; caller-supplied queries, table names,
edge operations, and predicates are deliberately absent from the contract.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    JsonValue,
    field_validator,
    model_validator,
)
from surrealdb import RecordID
from surrealdb.errors import NotFoundError

from .db import Database
from .edge_schema import EdgeSchemaRegistry
from .ingestion import implicit_edge_component
from .record_ids import (
    RECORD_ID_PATTERN,
    RECORD_ID_RE,
    as_record_id,
    canonical_record_id,
    split_record_id,
)
from .schema_catalog import RuntimeSchemaCatalog
from .semantic_ontology import SemanticOntology

from .mutation_ir import WriteMode

MutationStatus = Literal[
    "APPLIED", "PROPOSED", "NO_CHANGE", "REJECTED", "NOT_FOUND", "CONFLICT"
]


class _WriteModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, str_strip_whitespace=True
    )


class MutationSource(_WriteModel):
    """Opaque caller provenance carried through the mutation result.

    The graph has no audit/provenance schema today, so this hook is intentionally
    not persisted as an invented node or edge property.
    """

    type: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    session_id: str | None = Field(default=None, min_length=1, max_length=256)


class ItemDefinition(_WriteModel):
    id: str = Field(pattern=RECORD_ID_PATTERN)
    type: Literal["item"]
    properties: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_item(self) -> "ItemDefinition":
        table, _ = split_record_id(self.id)
        if table != self.type:
            raise ValueError("item ID must use the item table")
        if not self.properties:
            raise ValueError("item properties cannot be empty")
        forbidden = {"id", "in", "out"}.intersection(self.properties)
        if forbidden:
            raise ValueError("item properties cannot contain graph identity fields")
        return self


class CreateItemRequest(_WriteModel):
    operation: Literal["create"]
    item: ItemDefinition
    location_id: str = Field(pattern=RECORD_ID_PATTERN)
    mode: WriteMode = "commit"
    source: MutationSource | None = None


class UpdateLocationRequest(_WriteModel):
    operation: Literal["update_location"]
    item_id: str = Field(pattern=RECORD_ID_PATTERN)
    location_id: str = Field(pattern=RECORD_ID_PATTERN)
    mode: WriteMode = "commit"
    source: MutationSource | None = None

    @field_validator("item_id")
    @classmethod
    def require_item_id(cls, value: str) -> str:
        if split_record_id(value)[0] != "item":
            raise ValueError("item_id must use the item table")
        return value


class UpdateAttributesRequest(_WriteModel):
    operation: Literal["update_attributes"]
    item_id: str = Field(pattern=RECORD_ID_PATTERN)
    properties: dict[str, JsonValue] = Field(min_length=1)
    mode: WriteMode = "commit"
    source: MutationSource | None = None

    @field_validator("item_id")
    @classmethod
    def require_item_id(cls, value: str) -> str:
        if split_record_id(value)[0] != "item":
            raise ValueError("item_id must use the item table")
        return value


class DeleteItemRequest(_WriteModel):
    operation: Literal["delete"]
    item_id: str = Field(pattern=RECORD_ID_PATTERN)
    mode: WriteMode = "commit"
    source: MutationSource | None = None

    @field_validator("item_id")
    @classmethod
    def require_item_id(cls, value: str) -> str:
        if split_record_id(value)[0] != "item":
            raise ValueError("item_id must use the item table")
        return value


WriteRequest = Annotated[
    CreateItemRequest | UpdateLocationRequest | UpdateAttributesRequest | DeleteItemRequest,
    Field(discriminator="operation"),
]
WRITE_REQUEST_ADAPTER = TypeAdapter(WriteRequest)


class MutationError(_WriteModel):
    code: str
    field: str | None = None
    message: str


class MutationResult(_WriteModel):
    status: MutationStatus
    operation: Literal["create", "update_location", "update_attributes", "delete"] | None = None
    entity_id: str | None = None
    affected_nodes: int = 0
    affected_edges: int = 0
    previous_state: dict[str, Any] | None = None
    resulting_state: dict[str, Any] | None = None
    source: MutationSource | None = None
    reason: str | None = None
    errors: tuple[MutationError, ...] = ()


_CREATE_TRANSACTION = """
BEGIN TRANSACTION;
LET $existing = SELECT VALUE id FROM ONLY $item;
IF $existing != NONE { THROW "WRITE_CONFLICT"; };
LET $destination = SELECT VALUE id FROM ONLY $location;
IF $destination = NONE { THROW "WRITE_LOCATION_MISSING"; };
CREATE $item CONTENT $properties;
RELATE $item->$edge->$location CONTENT {};
COMMIT TRANSACTION;
"""

_UPDATE_LOCATION_TRANSACTION = """
BEGIN TRANSACTION;
LET $existing = SELECT VALUE id FROM ONLY $item;
IF $existing = NONE { THROW "WRITE_ITEM_MISSING"; };
LET $destination = SELECT VALUE id FROM ONLY $location;
IF $destination = NONE { THROW "WRITE_LOCATION_MISSING"; };
DELETE type::table($relation) WHERE in = $item;
RELATE $item->$edge->$location CONTENT {};
COMMIT TRANSACTION;
"""

_SET_INITIAL_LOCATION_TRANSACTION = """
BEGIN TRANSACTION;
LET $existing = SELECT VALUE id FROM ONLY $item;
IF $existing = NONE { THROW "WRITE_ITEM_MISSING"; };
LET $destination = SELECT VALUE id FROM ONLY $location;
IF $destination = NONE { THROW "WRITE_LOCATION_MISSING"; };
RELATE $item->$edge->$location CONTENT {};
COMMIT TRANSACTION;
"""


class ItemWritingService:
    """The sole item-write path from semantic intent to SurrealDB."""

    def __init__(
        self,
        database: Database,
        catalog: RuntimeSchemaCatalog,
        edge_registry: EdgeSchemaRegistry,
        ontology: SemanticOntology | None = None,
    ) -> None:
        self.database = database
        self.catalog = catalog
        self.edge_registry = edge_registry
        self.ontology = ontology or SemanticOntology.load_default()
        location_name = self.ontology.base_relations.get("location")
        if location_name is None:
            raise ValueError("The ontology does not declare item location")
        location_schema = self.edge_registry.resolve(location_name).schema
        if not location_schema.unique_from:
            raise ValueError("Item location must declare a unique source endpoint")
        self.location_relation = location_schema.id
        self.edge_registry.validate_endpoints(
            self.location_relation, "item", "space"
        )
        # All callers in this process converge here. The lock makes validation
        # reads and their following transaction one serialized write decision.
        self._lock = asyncio.Lock()

    async def mutate(self, request: WriteRequest | Mapping[str, Any]) -> MutationResult:
        """Validate and apply one semantic mutation without leaking DB errors."""
        if not isinstance(request, (CreateItemRequest, UpdateLocationRequest, UpdateAttributesRequest, DeleteItemRequest)):
            try:
                request = WRITE_REQUEST_ADAPTER.validate_python(request)
            except (ValidationError, TypeError, ValueError):
                return self._malformed_result(request)
        async with self._lock:
            try:
                if isinstance(request, CreateItemRequest):
                    return await self._create(request)
                if isinstance(request, UpdateLocationRequest):
                    return await self._update_location(request)
                if isinstance(request, UpdateAttributesRequest):
                    return await self._update_attributes(request)
                return await self._delete(request)
            except Exception:
                _, entity_id, _ = self._request_metadata(request)
                return self._storage_rejected(request, entity_id)

    async def _create(self, request: CreateItemRequest) -> MutationResult:
        entity_id = request.item.id
        properties = {"item_type": "unknown", **request.item.properties}
        properties_error = self._validate_item_properties(properties)
        if properties_error is not None:
            return self._rejected(request, entity_id, properties_error)
        if await self._record(entity_id) is not None:
            return self._result(
                request, "CONFLICT", entity_id, reason="ENTITY_ALREADY_EXISTS"
            )
        if await self._current_locations(entity_id):
            return self._result(
                request, "CONFLICT", entity_id, reason="ORPHANED_LOCATION_STATE"
            )
        location_error = await self._validate_location(request.location_id)
        if location_error is not None:
            return self._rejected(request, entity_id, location_error)
        resulting = {
            "entity": {"id": entity_id, **properties},
            "location": request.location_id,
        }
        if request.mode == "preview":
            return self._result(
                request, "PROPOSED", entity_id,
                affected_nodes=1, affected_edges=1, resulting_state=resulting,
            )
        try:
            await self.database.query(
                _CREATE_TRANSACTION,
                {
                    "item": as_record_id(entity_id),
                    "location": as_record_id(request.location_id),
                    "edge": self._location_edge_id(entity_id, request.location_id),
                    "properties": dict(properties),
                },
            )
        except Exception:
            return self._database_rejected(request, entity_id)
        return self._result(
            request, "APPLIED", entity_id,
            affected_nodes=1, affected_edges=1, resulting_state=resulting,
        )

    async def _update_location(
        self, request: UpdateLocationRequest
    ) -> MutationResult:
        entity_id = request.item_id
        if await self._record(entity_id) is None:
            return self._result(request, "NOT_FOUND", entity_id, reason="ITEM_NOT_FOUND")
        location_error = await self._validate_location(request.location_id)
        if location_error is not None:
            return self._rejected(request, entity_id, location_error)
        current = await self._current_locations(entity_id)
        if len(current) > 1:
            return self._result(
                request, "CONFLICT", entity_id,
                previous_state={"locations": current},
                reason="MULTIPLE_CURRENT_LOCATIONS",
            )
        previous = current[0] if current else None
        previous_state = {"location": previous}
        resulting_state = {"location": request.location_id}
        if previous == request.location_id:
            return self._result(
                request, "NO_CHANGE", entity_id,
                previous_state=previous_state,
                resulting_state=resulting_state,
                reason="LOCATION_UNCHANGED",
            )
        affected_edges = 1 + int(previous is not None)
        if request.mode == "preview":
            return self._result(
                request, "PROPOSED", entity_id,
                affected_edges=affected_edges,
                previous_state=previous_state,
                resulting_state=resulting_state,
            )
        try:
            await self.database.query(
                (
                    _UPDATE_LOCATION_TRANSACTION
                    if previous is not None
                    else _SET_INITIAL_LOCATION_TRANSACTION
                ),
                {
                    "item": as_record_id(entity_id),
                    "location": as_record_id(request.location_id),
                    "edge": self._location_edge_id(entity_id, request.location_id),
                    "relation": self.location_relation,
                },
            )
        except Exception:
            return self._database_rejected(
                request, entity_id, previous_state=previous_state
            )
        return self._result(
            request, "APPLIED", entity_id,
            affected_edges=affected_edges,
            previous_state=previous_state,
            resulting_state=resulting_state,
        )

    async def _update_attributes(self, request: UpdateAttributesRequest) -> MutationResult:
        entity_id = request.item_id
        entity = await self._record(entity_id)
        if entity is None:
            return self._result(request, "NOT_FOUND", entity_id, reason="ITEM_NOT_FOUND")
        writable = {prop.fields[0] for prop in self.ontology.properties.values()
                    if prop.item_writable and prop.fields}
        if set(request.properties) - writable:
            return self._rejected(request, entity_id, MutationError(
                code="ATTRIBUTE_NOT_WRITABLE", message="Only declared item attributes may be updated"))
        error = self._validate_item_properties(request.properties, require_name=False)
        if error:
            return self._rejected(request, entity_id, error)
        previous = {key: entity.get(key) for key in request.properties}
        resulting = dict(request.properties)
        if previous == resulting:
            return self._result(request, "NO_CHANGE", entity_id,
                                previous_state=previous, resulting_state=resulting)
        if request.mode == "preview":
            return self._result(request, "PROPOSED", entity_id, affected_nodes=1,
                                previous_state=previous, resulting_state=resulting)
        try:
            await self.database.query("""
BEGIN TRANSACTION;
LET $existing = SELECT * FROM ONLY $item;
IF $existing = NONE { THROW "WRITE_ITEM_MISSING"; };
IF $existing != $expected { THROW "WRITE_CONFLICT"; };
UPDATE $item MERGE $properties;
COMMIT TRANSACTION;
""", {"item": as_record_id(entity_id), "expected": {**entity, "id": as_record_id(entity_id)},
       "properties": dict(request.properties)})
        except Exception:
            return self._database_rejected(request, entity_id, previous_state=previous)
        return self._result(request, "APPLIED", entity_id, affected_nodes=1,
                            previous_state=previous, resulting_state=resulting)

    async def _delete(self, request: DeleteItemRequest) -> MutationResult:
        entity_id = request.item_id
        entity = await self._record(entity_id)
        if entity is None:
            return self._result(request, "NOT_FOUND", entity_id, reason="ITEM_NOT_FOUND")
        incident = await self._incident_edges(entity_id)
        previous_state = {
            "entity": {"id": entity_id},
            "incident_relationships": incident,
        }
        if request.mode == "preview":
            return self._result(
                request, "PROPOSED", entity_id,
                affected_nodes=1,
                affected_edges=sum(incident.values()),
                previous_state=previous_state,
                resulting_state={"entity": None, "incident_relationships": {}},
            )
        statement, variables = self._delete_transaction(entity_id, tuple(incident))
        try:
            await self.database.query(statement, variables)
        except Exception:
            return self._database_rejected(
                request, entity_id, previous_state=previous_state
            )
        return self._result(
            request, "APPLIED", entity_id,
            affected_nodes=1,
            affected_edges=sum(incident.values()),
            previous_state=previous_state,
            resulting_state={"entity": None, "incident_relationships": {}},
        )

    def _validate_item_properties(
        self, properties: Mapping[str, Any], *, require_name: bool = True
    ) -> MutationError | None:
        item_schema = self.catalog.entities.get("item")
        if item_schema is None:
            return MutationError(
                code="ITEM_SCHEMA_UNAVAILABLE", field="item.type",
                message="The deployed schema does not define item entities",
            )
        unknown = sorted(set(properties) - set(item_schema.properties))
        if unknown:
            return MutationError(
                code="UNKNOWN_ITEM_PROPERTY", field="item.properties",
                message="Item properties are not declared by the deployed schema",
            )
        for name, value in properties.items():
            if name == "item_type" and isinstance(value, str) and not value.strip():
                return MutationError(code="ITEM_TYPE_REQUIRED", message="Item category must be non-empty")
            expected = item_schema.property_types.get(name, "unknown")
            if not self._matches_kind(value, expected):
                return MutationError(
                    code="INVALID_ITEM_PROPERTY_TYPE",
                    field=f"item.properties.{name}",
                    message="Item property does not match the deployed schema type",
                )
        name = properties.get("name")
        if require_name and not self._valid_name(name):
            return MutationError(
                code="ITEM_NAME_REQUIRED", field="item.properties.name",
                message="Item name must be non-empty text, aliases, or localized text",
            )
        return None

    async def _validate_location(self, location_id: str) -> MutationError | None:
        location_type, _ = split_record_id(location_id)
        try:
            self.edge_registry.validate_endpoints(
                self.location_relation, "item", location_type
            )
        except ValueError:
            return MutationError(
                code="INVALID_LOCATION_TYPE", field="location_id",
                message="The destination type is not valid for item location",
            )
        if await self._record(location_id) is None:
            return MutationError(
                code="LOCATION_NOT_FOUND", field="location_id",
                message="The destination location does not exist",
            )
        return None

    async def _record(self, entity_id: str) -> dict[str, Any] | None:
        value = await self.database.query(
            "SELECT * FROM ONLY $record;", {"record": as_record_id(entity_id)}
        )
        if value is None:
            return None
        if isinstance(value, list):
            value = value[0] if value else None
        if not isinstance(value, Mapping):
            return None
        return {key: self._json_value(item) for key, item in value.items()}

    async def _current_locations(self, entity_id: str) -> list[str]:
        try:
            value = await self.database.query(
                "SELECT VALUE out FROM type::table($relation) WHERE in = $item;",
                {
                    "relation": self.location_relation,
                    "item": as_record_id(entity_id),
                },
            )
        except NotFoundError:
            return []
        if value is None:
            return []
        values = value if isinstance(value, list) else [value]
        return sorted(canonical_record_id(item) for item in values)

    async def _incident_edges(self, entity_id: str) -> dict[str, int]:
        item_relations = self._item_relations()
        incident: dict[str, int] = {}
        for relation in item_relations:
            try:
                value = await self.database.query(
                    "SELECT count() AS count FROM type::table($relation) "
                    "WHERE in = $item OR out = $item GROUP ALL;",
                    {"relation": relation, "item": as_record_id(entity_id)},
                )
            except NotFoundError:
                continue
            records = value if isinstance(value, list) else ([value] if value else [])
            count = int(records[0].get("count", 0)) if records else 0
            if count:
                incident[relation] = count
        return incident

    def _delete_transaction(
        self, entity_id: str, incident_relations: tuple[str, ...]
    ) -> tuple[str, dict[str, Any]]:
        statements = [
            "BEGIN TRANSACTION;",
            "LET $existing = SELECT VALUE id FROM ONLY $item;",
            'IF $existing = NONE { THROW "WRITE_ITEM_MISSING"; };',
        ]
        variables: dict[str, Any] = {"item": as_record_id(entity_id)}
        for index, relation in enumerate(incident_relations):
            name = f"relation_{index}"
            variables[name] = relation
            statements.append(
                f"DELETE type::table(${name}) WHERE in = $item OR out = $item;"
            )
        statements.extend(("DELETE $item;", "COMMIT TRANSACTION;"))
        return "\n".join(statements), variables

    def _item_relations(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.edge_registry.relationship_names
            if "item" in self.edge_registry.get(name).from_types
            or "item" in self.edge_registry.get(name).to_types
        )

    def _location_edge_id(self, item_id: str, location_id: str) -> RecordID:
        item = as_record_id(item_id)
        location = as_record_id(location_id)
        identifier = (
            f"{implicit_edge_component(item)}__{implicit_edge_component(location)}"
        )
        return RecordID(self.location_relation, identifier)

    @staticmethod
    def _valid_name(value: Any) -> bool:
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, Mapping):
            return bool(value) and all(
                isinstance(text, str) and bool(text.strip()) for text in value.values()
            )
        if isinstance(value, list):
            return bool(value) and all(
                isinstance(text, str) and bool(text.strip()) for text in value
            ) and len(value) == len(set(value))
        return False

    @staticmethod
    def _matches_kind(value: JsonValue, kind: str) -> bool:
        if kind in {"unknown", "any"}:
            return True
        if kind == "boolean":
            return isinstance(value, bool)
        if kind == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if kind == "number":
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )
        if kind == "string":
            return isinstance(value, str)
        if kind == "object":
            return isinstance(value, Mapping)
        if kind == "collection":
            return isinstance(value, list)
        if kind == "record":
            return isinstance(value, str) and bool(RECORD_ID_RE.fullmatch(value))
        if kind == "date":
            if not isinstance(value, str):
                return False
            try:
                date.fromisoformat(value)
            except ValueError:
                return False
            return True
        if kind == "datetime":
            if not isinstance(value, str):
                return False
            try:
                datetime.fromisoformat(value)
            except ValueError:
                return False
            return True
        return False

    @classmethod
    def _json_value(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): cls._json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_value(item) for item in value]
        if hasattr(value, "table_name") and hasattr(value, "id"):
            return canonical_record_id(value)
        return value

    @staticmethod
    def _request_metadata(
        request: WriteRequest,
    ) -> tuple[str, str, MutationSource | None]:
        entity_id = request.item.id if isinstance(request, CreateItemRequest) else request.item_id
        return request.operation, entity_id, request.source

    def _result(
        self,
        request: WriteRequest,
        status: MutationStatus,
        entity_id: str,
        **values: Any,
    ) -> MutationResult:
        return MutationResult(
            status=status,
            operation=request.operation,
            entity_id=entity_id,
            source=request.source,
            **values,
        )

    def _rejected(
        self,
        request: WriteRequest,
        entity_id: str,
        error: MutationError,
    ) -> MutationResult:
        return self._result(
            request, "REJECTED", entity_id, reason=error.code, errors=(error,)
        )

    def _database_rejected(
        self,
        request: WriteRequest,
        entity_id: str,
        *,
        previous_state: dict[str, Any] | None = None,
    ) -> MutationResult:
        error = MutationError(
            code="TRANSACTION_FAILED",
            message="The item mutation transaction was not applied",
        )
        return self._result(
            request, "REJECTED", entity_id,
            previous_state=previous_state,
            reason=error.code,
            errors=(error,),
        )

    def _storage_rejected(
        self,
        request: WriteRequest,
        entity_id: str,
    ) -> MutationResult:
        error = MutationError(
            code="STORAGE_FAILURE",
            message="The item mutation could not validate authoritative state",
        )
        return self._result(
            request, "REJECTED", entity_id,
            reason=error.code,
            errors=(error,),
        )

    @staticmethod
    def _malformed_result(raw: Any) -> MutationResult:
        operation = raw.get("operation") if isinstance(raw, Mapping) else None
        if operation not in {"create", "update_location", "update_attributes", "delete"}:
            operation = None
        item = raw.get("item") if isinstance(raw, Mapping) else None
        entity_id = (
            item.get("id") if isinstance(item, Mapping)
            else raw.get("item_id") if isinstance(raw, Mapping)
            else None
        )
        if not isinstance(entity_id, str) or not entity_id.startswith("item:"):
            entity_id = None
        validation_error = MutationError(
            code="INVALID_REQUEST",
            message="The semantic item mutation request failed validation",
        )
        return MutationResult(
            status="REJECTED",
            operation=operation,
            entity_id=entity_id,
            reason=validation_error.code,
            errors=(validation_error,),
        )
