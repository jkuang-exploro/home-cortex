# Bounded semantic composition contract (2026-09-07)

This extends the existing semantic IR; property ownership and singular resolution
remain explicit. No utterance is inspected after interpretation.

- **Shapes:** `shape=entity` identifies one resolved entity, `entities` an entity
  collection, `scalar` a property/computation value, and `rows` associated results.
  Ordinary scalar requests consume one resolved entity (multiple
  matches remain ambiguous). `select(property=null)` returns an entity collection.
  `projection="each"` consumes an explicit collection and applies one scalar
  operation or property selection independently, returning `shape="rows"` and
  typed rows containing entity, value, unit, status, evidence, missing requirements.
  No nested projections, reductions over computed rows, or pairwise mapping.
  Entity properties produce one row per distinct entity. Relationship properties
  produce one row per final edge, associated with its target entity and edge
  evidence; multiple edges are never arbitrarily collapsed to one scalar. For
  relationship rows, all source=relation filters bind the same final edge.
- **Exclusion:** `exclude` holds at most eight semantic references. Each resolves
  singularly unless explicitly declared a discourse collection. Resolve exclusions
  before filtering/projection; subtract canonical identities and their final edges.
  Unresolved/ambiguous exclusions fail the whole request, including empty inputs.
  `other` retains its comparison meaning. `kind="unresolved"` explicitly requests
  clarification when interpretation cannot select a reference. The interpreter must express whether
  “other” excludes self or a discourse referent; missing meaning requires clarification.
- **Missing data:** a missing projected property affects only its row. Empty
  collections return successful empty rows. A rows result's `found` status means
  collection evaluation completed, not that every row succeeded. Missing filter
  evidence continues to fail the query; exclusion/identity failures are not hidden.
- **Calendar offset:** `date_add(property, amount, mode)` accepts a strict signed
  integer amount bounded to +/-120000 and modes years/months/days. Calendar overflow
  uses the following month's first day if the target day does not exist:
  Feb 29 + one year and Jan 31 + one month both land on March 1. This matches
  existing date-only completed-period boundaries. Offsets are applied once, not iterated or inverted.
  Dates remain dates. Datetimes use the trusted household timezone, preserve wall
  time, and return ISO datetimes with offset. Naive stored datetimes retain the
  existing UTC interpretation. Nonexistent or ambiguous target wall times fail
  explicitly. Out-of-range dates fail. Past dates are returned unchanged by the
  clock; `annual_occurrence` retains its separate next-valid-recurrence semantics.
- **Discourse:** `kind="discourse"`, `turn_offset=1..8`, `cardinality=single|collection`
  addresses the resolved subject focus of that preceding user turn, not an ID or
  a fact from assistant prose. The interpreter specifies the earlier turn and
  cardinality. A singular reference to a multi-entity focus is ambiguous; no gender,
  recency, or first-candidate guessing. Paths compose from that resolved focus.
  Focus consists only of graph-resolved identities from successful execution;
  pairwise queries retain both operands; collection focus retains all result
  entities. Non-fact turns, failures, and empty results have no antecedent.
  Relationship-property follow-ups compose a relationship from an entity focus;
  bare edge references are not yet part of the reference algebra.
- **Conversation boundary:** explicit conversation IDs use bounded process-local
  state keyed by conversation, authenticated speaker, household, and agent.
  API IDs must belong to the caller/agent. State keeps eight user turns and expires
  on eviction/restart; missing state clarifies. Calls without an ID are request-local:
  up to eight supplied prior user turns are reinterpreted and re-grounded in order.
  They never reuse another request's focus. Assistant prose is not forwarded as
  evidence or replayed. The model sees user discourse but no bindings or canonical
  IDs. The resolver reloads bound entities from authoritative storage each turn.
  Explicit sessions serialize turn evaluation and use server history; caller
  supplied earlier history cannot overwrite stored focus. Unauthenticated calls
  cannot persist discourse across requests.

Implementation and deterministic acceptance are separate from real-model accuracy.
New production acceptance requires Grok's isolated GPU report and fingerprints.
