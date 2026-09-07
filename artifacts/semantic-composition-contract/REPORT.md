# Composition implementation and historical production review

Source baseline: `b0e145c`. Review date: 2026-09-07.

## Architecture changes

The [contract](../../docs/semantic-composition-contract.md) was written before
implementation. Changes were built incrementally: collection projection and
reference exclusion; generic calendar addition; trusted discourse and service/API
wiring; result serialization and explicit row scoring.

`projection=each` reuses scalar operations, retaining per-entity values, units,
missing statuses, and edge evidence. Relationship properties yield per-edge rows;
edge conditions must hold on that same edge. Exclusions resolve authoritatively.
Singularity, property ownership, predicates, and complete ontology expansion
remain validated. No question routing, household mappings, or specialized age or
birthday operations were added.

`date_add` works on both property owners and returns past as well as future
anniversaries. Invalid month days become the following month's first day,
consistent with existing date-only completed-period boundaries. Datetime offsets
preserve household wall time and explicitly reject DST gaps/folds.

Both history truncation sites now preserve user discourse. Session bindings are
produced only by deterministic execution and reloaded through the resolver.
Explicit sessions check ownership and serialize turns. Stateless antecedents are
interpreted and re-grounded on demand in request-local context, with no shared
focus or assistant-prose facts. Topic changes, failed turns, eviction, and
absent/ambiguous antecedents cannot silently fall back to an earlier person.

The first deployed UI integration stored the Cortex conversation ID on the
greeting message but did not put it on completion requests. Open WebUI therefore
always selected the stateless replay path. The follow-up fix persists the ID in
`params.custom_params.conversation_id`; the pinned Open WebUI OpenAI payload
builder promotes custom parameters to the upstream request body. The API then
authorizes that ID against agent and speaker before using session focus.

## Deterministic validation

Local command: `UV_CACHE_DIR=/tmp/home-cortex-uv-cache uv run pytest -q` —
**439 passed**, including 53 new compositional checks. Only the interpreter
transport is mocked in service and HTTP tests; the planner, validation, resolver,
graph dispatcher, executor, renderer and conversation coordinator run normally
against invented graph data.
JSON and streaming HTTP requests pass the named-person/anniversary follow-up;
cross-owner conversation access is rejected. Tests cover partial/empty collections,
edge-specific filters and multiple residence edges, ambiguity, topic changes,
stateless replay, session eviction, concurrency, deleted/changed authoritative
records, property ownership, past anniversaries, leap/month boundaries and DST.
No production household data was copied into these tests.

## Review of Grok's findings in git history

Reviewed `7802132:artifacts/tier1-baseline/REPORT.md` (the commit named “grok
probe and benchmark”) and the latest production report and summary committed in
`b0e145c`. Large per-case JSONs were not loaded.

The original Grok report used qwen3:8b on CPU: 75/100 measured probe plans and
answers, 88/119 suite plans, with all 119 suite answers unscored. Its failures
included relationship property ownership, missing comparison operands/ordering,
incomplete son filters, and elapsed time substituted for birthday recurrence.
These findings support preserving strict layer boundaries and distinct temporal
operations; they are not measurements of this change.

The latest [date-interval report](../date-interval-contract/REPORT.md) and
[suite summary](../date-interval-contract/suite-final-summary.json) used
qwen3.5:9b, 100% GPU, the isolated `/tmp/hc-date-intervals` package, JSON graph
backend, temperature 0, seed 0, think=false, 384 output tokens and one warm-up.
Package/data/schema/model/contract fingerprints are present, although container
git revision is recorded as unavailable. The suite clock was September 3; the
16-case date probe used September 7. Those runs are not a same-configuration
before/after comparison with the original CPU report.

The latest suite records **113/119 plan matches**, with **119/119 answers
unscored**. All six remaining mismatches were reviewed from its failure table:

- Four unspecified-unit marriage-duration cases use `years`; the declared default
  and existing expectations require `days`. These stay failures even though
  execution returned a valid interval with units.
- `self_identity::你知道我叫什么吗` adds an incompatible property to identity
  resolution and fails `INVALID_PLAN` before execution.
- `son_identity::哪个男孩是我儿子` chooses a collection selection instead of singular
  identity resolution. The new projection primitive does not legitimize this error.

The 16 explicit date-interval cases and 24 prior age/filter cases passed in those
historical GPU runs. They do not cover the new projection or discourse contract.
The historical fixed suite also has two `entity_not_found` and one
`property_unavailable` statuses, independent of its six plan mismatches; `found`
is not an answer-correctness oracle. No old failures were relabeled or repaired.

## Evaluation changes and limits

Scoring revision `2026-09-07.2-composition-shapes` serializes result shape, rows,
row evidence and focus IDs. `expected_rows` requires identity, value, unit and
status for every row, with optional evidence/missing requirements. Row order is
irrelevant, multiplicity is preserved, and absent row expectations remain unscored.
The old display-name equivalence is restricted to ordinary scalar selection;
collection projection and exclusion cannot be normalized into singular identity.
Historical artifacts were not rescored and benchmark wording was not copied into
interpreter examples.

No real-LLM benchmark or deployment was performed for this implementation. The
new contract still needs Grok's isolated GPU run before any production accuracy
claim: both observed chains through the normal service, unseen compositions,
warm-up 1/repeat 5, then the fixed 119, age/filter 24 and date-interval 16 suites.
Record all fingerprints and score plans, bindings, membership, row values/units/
statuses, and presentation separately. Measure demand-driven antecedent latency as
well as explicit-session latency; a chain may add eight interpreter calls. Process-local
session state requires consistent worker routing or a future shared store.
