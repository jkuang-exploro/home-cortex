# Item attribute tools

## Architecture

The named write_item contract now supports update_attributes in addition to
create, update_location, and delete. Create accepts semantic attributes and
always stores an item_type (unknown when omitted). Per the user's preference,
the interpreter may infer clear categories from item nouns. Other attributes
are intended to come from explicit user input. No household-specific lookup or
utterance repair was introduced.

Ontology item_writable declarations own writable types and write_hint metadata.
The runtime catalog includes declared fields even before source data has values.
Supported attributes: item_type, brand, model, color, quantity, unit, description,
expiration_date. Model output schemas close the attribute key set and constrain
value types. The name adapter maps semantic names to physical fields and resolves
items in the configured household; model output cannot set IDs, metadata or edges.

Canonical updates use a transaction that verifies the validated entity snapshot
before MERGE. Unspecified attributes, names and locations are preserved. Preview
and no-op do not mutate. Creation with conflicting explicit attributes fails and
requires an explicit update. Existing untyped records can be explicitly updated;
no batch classification or production data rewrite is part of deployment.

Generic inspect is a deterministic read operation over one entity's declared
semantic properties, showing unrecorded values explicitly. Single-property reads
continue to use select(property). Responses are rendered without an answering LLM.

## Deterministic verification

681 tests passed locally. Added coverage includes typed and closed output schemas,
create and update attributes, unknown fallback, legacy backfill, preview, repeat,
name/location/other-property preservation, invalid and non-writable fields,
household scope, semantic read-after-write, all-attribute inspection, rollback,
and rejection of a concurrent stale snapshot.

## GPU verification

Runs used qwen3.5:9b on an isolated package with invented embedded SurrealDB data.
Chinese sausage creation inferred food. Explicit brand/quantity/unit updates
applied, inspection returned them, preview did not change quantity, and a food
filter found the item. English tool creation and brand update used the complete
literal item name. An unidentified item received unknown. Quotes and negated
writes made no mutations; their ordinary conversational answers were not scored.
No broad language-model accuracy claim is made.

The final candidate schemas prevent unknown attribute keys. Earlier candidates
were rejected for copying a command into description and generating default
quantity/unit fields. Final targeted checks omitted default quantity/unit, but
still sometimes duplicate the user's literal item phrase into description.
That remaining interpretation limitation is recorded, not counted as perfect
attribute extraction. Exact names are intentional: red screwdriver is not
silently resolved from screwdriver.

Final package SHA-256 (sorted relative Python paths+contents):
26349a15c821c70721f78810df62e5592b7b5705257b06780c2d7837bc836c89
Schema SHA-256: 4c8512142043b3b2e1060420a4b6fe438797feb187f67a8b544d805b11e8a7b0
Prompt markdown SHA-256: adae0e8759ca4975f3839efc28a9303f227b73523214d31d9d6ad8001fae9321
Invented fixture SHA-256: a0b65592d9d431f28c16f15710e263b85110ad2b09b3014a526169cbf41622e7

## Evaluation compatibility

The compact offline codec moves from V2 to V3 for its changed field dictionary;
old bytes are rejected instead of reinterpreted. Production remains expanded JSON.
Composition fingerprints changed for the ontology; frozen cases/scoring did not.

Final deployment package hash after a conflict-message-only correction: bce2f0deb0e3972058c21cb67771531ec9047c4f02cf581b038494410305b9bc. The correction directs existing-item requests to attribute or location updates.
