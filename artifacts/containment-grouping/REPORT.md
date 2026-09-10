# Collapsed contents grouped by space

## Architecture

Collapsed `contents` traversal keeps its flat entity result and now carries
`content_groups` with the authoritative containing spaces. Group membership is
restricted to the final selected entities after filters and exclusions. Counts,
subsequent traversal, direct subspace queries, evidence endpoints, cycle detection,
and completeness checks retain their existing semantics. The renderer uses stored
space names and emits only nonempty groups; it does not read graph storage or the
utterance.

The interpreter preserves compound names. Named storage references may use the
existing nullable type contract instead of guessing item versus space. Exact-name
resolution remains responsible for grounding; explicit types are still binding
and same-name cross-type records remain ambiguous. Three redundant demonstrations
were replaced with generic container, subspace, and named-room examples, retaining
the 42-example budget. No reported household wording or identity mappings were
added to the prompt.

## Runtime correction

The GPU API already had the earlier direction fix, but its database fridge record
lacked `collapse: true`. That single field was updated. The mounted legacy item
source was backed up; after new item shards appeared, all legacy IDs were verified
to exist in the shards and the duplicate `nodes/item.json` was moved to `/tmp`.
No full graph ingestion or location-edge rewrite was performed.

## Validation and limits

Local deterministic suite: 655 passed. Added coverage uses invented data through
real embedded SurrealDB, ingestion, the internal dispatcher, resolver, executor,
and renderer, including nullable typing, ambiguity, explicit wrong types, nested
spaces, grouping, exclusions, counts, and subsequent location traversal.

An isolated package on the GPU host with qwen3.5:9b and PYTHONHASHSEED=0 was checked
against the reported four-turn sequence. It returned grouped fridge contents,
only the fridge for the kitchen, and only cheese for the door shelf. Seven
additional single-turn probes preserved compound storage names, an independently
named room, or the requested object-location relation. These are bounded debugging
checks, not a production accuracy benchmark. Earlier prompt candidates introduced
scope/type regressions and were not deployed.

Final isolated diagnostic fingerprints (SHA-256, sorted relative paths followed
by file contents):

- Package (`*.py`): `377ebbf288e460a7cfe1c0028e5ed28e0f96d4c9fb5eb610ef49b5ef95138c93`
- Schemas (`*.yaml`): `b7a6fd4aa1ecefed58cb546146210c53354cc08bb0cd4e40b1f4a8664a86ccf6`
- Mounted source data (`*.json`): `aed59824250d3e1572166f32acc40ba9d09716d3d7f8755d5d040851eab1ee55`

The source-data fingerprint is not a database snapshot: the targeted database
metadata correction was independently read back. No full accuracy claim is made.

## Deployed HTTP verification

The patch was applied to the GPU checkout and the API image rebuilt. Only the
API service was recreated (`up -d --no-deps cortex-api`). `/health` returned 200.
The actual `/v1/chat/completions` endpoint was then queried through the configured
project-owner mapping, with prior user and assistant turns retained. All four
reported turns returned the intended scope and grouping:

- Fridge: door shelf → cheese; interior → bread and milk.
- Kitchen: fridge only.
- Fridge again: the same two nonempty space groups.
- Door shelf: cheese only.

HTTP request IDs, respectively:
`b7f0464f9e21489cb02da90cc5a10be7`,
`8035afc88f1746ea83ce22d31900c55d`,
`d4af014dddcc40a5897542bd69ba76ad`,
`8d42a7be9e1d4ced9c6cdb1af1989e8e`.

This verifies the reported flow on the deployed service; it does not establish
accuracy for all household questions. No synthetic food was added to production.
