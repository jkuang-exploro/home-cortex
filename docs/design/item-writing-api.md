# Canonical item Writing API

`ItemWritingService` is the single semantic write boundary for active household
items. Callers submit a discriminated `WriteRequest`; the service validates the
request against the runtime node catalog, semantic ontology, and edge registry,
then owns every SurrealDB statement. There is no caller field for SurrealQL,
tables, predicates, node operations, or edge operations.

The public semantic surface is:

```text
create(item, location_id, mode, source?)
update_location(item_id, location_id, mode, source?)
delete(item_id, mode, source?)
```

These are the only operation discriminator values. `mode` is `preview` or
`commit`. The return value is always `MutationResult`, with stable statuses,
counts, semantic before/after state, reason codes, and sanitized errors. Expected
missing, conflict, validation, and transaction failures do not expose database
responses or exception text.

The repository calls the physical relationship `located_in`. The semantic
`location` relation is resolved through the active ontology and edge registry;
the Writing API does not introduce the ticket's illustrative `locate_in` table.
Destination endpoint validation therefore follows the same declared ownership as
the read path.

## Transaction and consistency boundaries

Each commit is one multi-statement SurrealDB transaction:

- create: verify absence and destination, create the item, relate its location;
- update_location: verify item and destination, delete the old `located_in`,
  relate the new location;
- delete: verify the item, delete every incident relationship whose schema admits
  an Item endpoint, delete the node.

The service serializes validation and mutation decisions with one asynchronous
lock. All in-process LLM, user, and future perception callers must share this
service instance. Combined with the database transaction, this preserves at most
one current item location and prevents partial compound writes. Code that writes
these nodes or edges directly would be a competing mutation path and is outside
the contract.

Preview takes the same lock and performs the same schema and state validation,
but sends no mutating statement. A same-location update returns `NO_CHANGE` in
either mode. Create conflicts, missing delete targets, and repeated updates have
deterministic results.

Edge deletion is schema-derived. The service inspects all registered edge schemas
that permit an Item endpoint, reports their incident counts, and includes the
nonempty applicable tables in the delete transaction. This avoids hardcoding only
`located_in` while avoiding errors against registered but never-created empty
tables. Because all item writers must converge on this service, no canonical
writer can add an edge between that inspection and the transaction.

## LLM adapter and activation

`get_tool_definitions(["write_item"])` exposes the discriminated request schema.
`ToolDispatcher(..., ["write_item"], writing=service)` validates the tool call and
delegates directly to `ItemWritingService.mutate`. The adapter has no mutation
logic of its own.

The household steward is not automatically granted the tool. Its existing
allowlist remains unchanged. A caller must explicitly allowlist `write_item` and
provide the canonical service. This ticket adds the callable contract and runtime
dependency without silently expanding agent authorization.

## Item properties and provenance

Create properties must be JSON values, include a valid `name`, and use fields and
field types present in the deployed Item catalog. Graph identity fields (`id`,
`in`, `out`) are forbidden inside properties. The node ID must use the `item`
table, and destinations must exist and satisfy the declared `located_in`
signature.

`source` is the smallest extensible provenance hook: a bounded source type and
optional session ID. No audit/provenance node, edge, or property exists in the
current schema, so provenance is returned in `MutationResult` and is not persisted
as an invented graph fact. A future audit design can consume the hook without
changing the mutation intents.

## Existing repository constraint

`ingest_directory` remains the canonical bulk file-to-database synchronization
path for deployment/bootstrap data. It prunes database records that are absent
from its JSON source. Running full ingestion after runtime item mutations can
therefore remove or overwrite those mutations. Making runtime writes durable
across a later source-of-truth reingestion requires an explicit persistence policy;
silently editing JSON files would break the SurrealDB transaction boundary and is
not part of this API.

There are consequently two different scopes, not competing semantic APIs:

```text
deployment data synchronization -> ingest_directory
runtime item intent             -> ItemWritingService
```

Future runtime modules must use the latter rather than calling SurrealDB directly.

