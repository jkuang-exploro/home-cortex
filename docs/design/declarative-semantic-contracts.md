# Declarative semantic type and value contracts

Status: **proposed design; not implemented**. Owner: Codex. Priority: P0.
Dependency: [Ticket 1 layer traces](../../artifacts/layer-failure-trace/REPORT.md).
This is the design deliverable for Ticket 2, not a description of deployed behavior.

Follow-on status: Ticket 3 implements the display-only slice through optional V1
`label`/`value_labels` metadata and compositional rendering. See its
[implementation report](../../artifacts/condition-rendering/REPORT.md).
An opt-in V2 type/domain/compiler candidate is now implemented; the default
ontology remains V1. See the [compatibility and evaluation handoff](../../artifacts/generic-contracts/REPORT.md)
for implemented boundaries and remaining provider/production checks. This design
is the proposal baseline, not a claim that every migration phase has completed.

## 1. Decision

Extend the existing ontology and `SemanticSchemaRegistry`; retain
`SemanticFactRequest`, `SemanticReference`, `SemanticFilter`, concept expansion,
and the deterministic executor. Do not introduce a second meaning IR, question
router, text-matching repair pass, new model call, or household answer cache.

Add four reusable declarations:

1. Property value types and optional closed value domains with scoped aliases.
2. Explicit semantic applicability and allowed filter operators.
3. Localized display labels for properties, domain values, predicates and concepts.
4. Optional, explicit disjointness between declared collection predicates.

Compile these into one immutable resolved contract per schema instance. Its
model-facing capabilities, supported JSON Schema subset, and authoritative
runtime checks must share the same type/domain/operator definitions. Existing
operator implementations and relation signatures remain their own authoritative
registries; the ontology restricts their use rather than reimplementing them.

This addresses illegal combinations and makes semantic distinctions more visible.
It does **not** establish that a legal plan faithfully interprets the utterance.
The interpreter can still choose `adult` when the user meant `gender=male`.
That requires the separate compositional evaluation/model work in Tickets 4–7.

## 2. Evidence and existing implementation

Ticket 1 used isolated synthetic data with resident qwen3.5:9b on Ollama 0.32.15,
three repetitions, and recorded fingerprints. It did not read production graph
records or private conversations. Its results supersede speculation based only
on the user's rendered transcript.

| Finding | Existing defense / behavior | Design implication |
|---|---|---|
| `我家里都有谁` becomes `select self -> member` | Traversal validation already rejects the person/address mismatch | Advertise concept input/output types clearly; retain rejection. Never change `self` into `current_household` downstream. |
| Gender counts become `adult AND minor`, with no gender filter | Both predicates are individually valid; conjunction is accepted | Advertise gender's value domain; optionally reject declared same-scope predicate contradictions. Neither measure guarantees gender is chosen. |
| After an adult question, gender count retains `adult` | The plan is executable but means the wrong thing | Do not infer intent from type correctness or successful execution. Record as a semantic interpretation failure. |
| In-law query after father query emits `father -> spouse` | Both hops can be structurally valid | Types cannot distinguish all legitimate relationship meanings. Preserve paths, including valid nested in-law paths. |
| Birthday wording becomes stored-date selection | Both `select(birth_date)` and `annual_occurrence(birth_date)` are valid | Operation intent remains a model decision. Do not rewrite one into the other. |
| Correct gender-filtered gold plan loses gender in rendered text | `_count_noun()` selects coarse nouns; correct numerical execution | Provide display metadata and a structured condition-description interface for Ticket 3. |

Relevant source audit:

- `semantic_ontology.py`: `OntologyProperty` currently contains `fields`, `aliases`
  and `ordering`; version 1 rejects unknown property keys. Domain/type/display
  fields are genuinely new. Concepts already own complete paths and filters.
- `SemanticSchemaRegistry.physical_property()` / `relation_property()` map semantic
  properties to physical fields. `_property_candidates()` can also expose an
  unaliased physical field under its own name. Version 2 must not silently inherit
  that fallback into the model-facing vocabulary.
- `schema_catalog.py`: property types can be inferred from records; unknown or
  sparsely populated data can limit the available type information. Storage
  observations are not a semantic value-domain definition.
- `planner_output_schema()` already separates field filters from predicates,
  collection filters from traversal anchor comparisons, and contextual from named
  references. Property names are enumerated globally, while filter values still
  use a broad scalar/tuple union. This is not a missing field-versus-predicate IR.
- `validation_code()` already checks relationship endpoints, property ownership,
  operation inputs, projection, pairwise operands, exclusions and predicate entity
  applicability. `_valid_predicate()` checks operator field kinds and some value
  shapes, but does not bind literal values to a property's declared domain.
- `_semantic_kind()` collects kinds for available bindings; it does not by itself
  prove a property is valid for **every** possible endpoint type. Version 2 checks
  applicability universally over reachable endpoint types.
- `operator_registry.py` already defines `ValueKind`, input/output shapes,
  required parameters and legal field kinds. Equality and membership accept
  `any` field kind; this is not permission for arbitrary literal types.
- Collection filter execution already reports `filter_input_missing` for missing
  inputs. Preserve this behavior instead of inventing missing-value-to-zero rules.
  Relation filters currently use any-matching associated edge; this proposal does
  not redefine edge association or quantification.
- `SemanticConversationService` already scopes trusted state by conversation,
  speaker, household and agent. This design does not alter that state or its
  history selection. Existing utterance-specific identity hints are outside this
  migration; do not extend that pattern to new query classes.

## 3. Concrete proposed ontology format

Use `version: 2`. The following is a **YAML fragment to merge into a complete
ontology**, not a runnable replacement file. Existing physical fields remain in
the internal adapter declarations and never appear in model-facing capabilities.

```yaml
version: 2
properties:
  gender:
    fields: [gender, sex]          # Existing internal adapter field candidates.
    aliases: [性别, gender, sex]   # Property-name aliases, not question patterns.
    type: {kind: string}
    applies_to:
      entity: [person]
      relationship: []
    filter_operators: [eq, ne, in, exists]
    values:
      male:
        aliases: [男, 男性, male]
        label: {en: male, zh: 男性}
      female:
        aliases: [女, 女性, female]
        label: {en: female, zh: 女性}
    label: {en: gender, zh: 性别}

  birth_date:
    fields: [birth_date, birthday, dob, date_of_birth]
    aliases: [出生日期, birth date, date of birth]
    type: {kind: date}
    applies_to: {entity: [person], relationship: []}
    filter_operators: [eq, ne, gt, gte, lt, lte, in, exists, date_range]
    label: {en: birth date, zh: 出生日期}
    ordering:
      minimum: [年长, older, oldest]
      maximum: [年幼, younger, youngest]

  start_date:
    fields: [start, since, start_date]
    aliases: [开始日期, start date, since]
    type: {kind: date}
    applies_to:
      entity: []
      relationship: [spouse, residence, member]  # Semantic relations, not tables.
    filter_operators: [eq, ne, gt, gte, lt, lte, in, exists, date_range]
    label: {en: start date, zh: 开始日期}

  display_name:
    fields: [display_name, name, full_name]
    aliases: [姓名, name, display name]
    type:
      any_of:
        - {kind: string}
        - {kind: collection, items: {kind: string}}
    applies_to: {entity: [person], relationship: []}
    filter_operators: [exists]
    label: {en: name, zh: 姓名}
```

The date fragment intentionally specifies date-only values. Before migrating
`start_date`, audit adapters: if a supported binding contains datetimes, declare
an explicit supported union or a separate datetime semantic property. Do not
silently truncate timestamps to dates. Likewise, the `display_name` union
preserves existing scalar/list-valued name representation; it is not an invitation
to enable unspecified list containment or ordering operations.

Example additive metadata on an existing predicate (all existing role/fallback
fields remain required and unchanged):

```yaml
collection_predicates:
  adult:
    # Existing aliases, entity_types, role policy and fallback remain here.
    label: {en: adult, zh: 成年}
    disjoint_with: [minor]
  minor:
    # Existing definition remains here.
    label: {en: minor, zh: 未成年}
    disjoint_with: [adult]
```

Add an optional `label: {en: ..., zh: ...}` to each existing reference concept.
Do not add another `emit` field or another copy of its path. Concept input/output
type signatures are **derived** by composing the existing semantic relation
signatures; they are not manually declared a second time.

### Type and value rules

- Supported initial atomic kinds: string, boolean, integer, number, date, datetime.
  Reuse their operator-registry meanings. Add only the shallow collection/union
  descriptor needed for existing stored representations. Arbitrary JSON objects,
  recursive types and new list operators are out of scope for model-originated
  field filtering in this migration.
- Integer and number are distinct from boolean even though Python `bool` is an
  `int` subclass. Numbers must be finite. Integer literals are valid for a number
  property; floats are not silently accepted for an integer property.
- Date literals must be valid ISO dates. Datetimes must include a timezone offset.
  JSON Schema `format` is advisory unless the selected decoder enforces it;
  authoritative parsing and range checks always occur in runtime validation.
- `values`, when present, is a closed domain of canonical **string** values, not
  the set of values observed in a household. It is supported only for string
  properties in this phase. Non-string domains can be a later explicit extension.
- The example gender domain is illustrative. Migration must audit supported
  source values and include every intentionally supported category. Missing data
  and an explicitly stored domain value such as `unknown` are different; neither
  is automatically mapped to male or female.
- Language aliases help the model select canonical values. A model-produced
  `value: "女性"` is rejected if the canonical value is `female`; do not repair it
  by rescanning the question. Literal names remain `named_entity` references.
- Data-source encodings, if different from canonical values, belong in explicit
  resolver/schema adapters, not the linguistic alias map. No new identity or
  gender inference from names, role, or relatives is permitted.
- Alias collisions are checked within each property/value domain after the
  existing case-fold convention. Cross-property aliases need not be globally
  unique (the existing ontology already has context-dependent aliases). Do not
  build a global first-match keyword router. Ambiguous lexical terms may remain
  unmapped rather than force a gender or adulthood interpretation.
- Display fallback: exact locale, base language, English, then the semantic key.
  Labels are plain text and must be escaped by the presentation layer. Missing
  translations do not remove a condition from the answer.

## 4. Resolve once, generate three views

Introduce an internal immutable `ResolvedSemanticContract` owned by the existing
`SemanticSchemaRegistry`, not a new API or parallel planner. It contains typed
property/domain/operator descriptors, predicate metadata, existing relation and
concept signatures, and internal adapter bindings. Its public view excludes
physical field names, graph values, identity bindings and private records.

Compilation sequence:

1. Load and validate the versioned ontology: key shapes, referenced vocabulary,
   types, operator subsets, domain aliases, labels and predicate disjointness.
2. Bind declared entity types/semantic relations through the catalog and adapters.
   Check all reachable endpoint types, rather than accepting any one matching
   physical property. Preserve source ownership explicitly.
3. Intersect declared filter operators with the existing operator registry's
   supported types and operand shapes. An incompatible declaration is a startup
   configuration error, not a model retry.
4. Validate ontology concept filters and predicate dependencies through the same
   descriptor checks. Preserve their complete paths and anchor positions.
5. Produce deterministic capabilities, JSON Schema fragments, validator lookups
   and display descriptors. Cache by the resolved ontology/catalog/operator
   fingerprint; return immutable data or defensive copies.

A supported binding with unknown observed kind may use an explicit declared
semantic type and validate values when read. A known incompatible kind fails
binding validation. A binding absent from the catalog is unavailable: omit it
from model capabilities, and reject structured requests that use it. Dependent
concepts must be disabled as complete concepts, never shortened. A malformed
reference to a nonexistent vocabulary item is a configuration error, distinct
from an unavailable deployment binding.

Empty storage does not invent a binding. Supporting a new property requires its
ontology declaration **and** a catalog/schema-adapter binding, which may be
explicit metadata rather than inferred data. No question handler is required.
New operations still require a generic operator implementation.

### Model-facing capability example

```json
{
  "property_contracts": {
    "gender": {
      "type": {"kind": "string"},
      "applies_to": {"entity": ["person"], "relationship": []},
      "filter_operators": ["eq", "ne", "in", "exists"],
      "values": {
        "male": {"aliases": ["男", "男性", "male"]},
        "female": {"aliases": ["女", "女性", "female"]}
      }
    }
  }
}
```

Keep field values separate from `collection_predicates`. Preserve the existing
predicate policy and `definition_only` explanations; do not ask the model to
compute adulthood or encode storage-role tests. Do not duplicate the same full
property descriptions in old and new capability sections indefinitely: migrate
consumers and retain only the compact references needed for ownership/signatures.

## 5. Preserve the model IR and compile stricter schema fragments

The valid model request for a female household-member count remains:

```json
{
  "requires_fact": true,
  "request": {
    "operation": "count",
    "subject": {
      "kind": "current_household",
      "entity_type": "address",
      "path": [{"concept": "member"}]
    },
    "property": null,
    "property_source": "entity",
    "filters": [
      {"property": "gender", "source": "entity", "operator": "eq", "value": "female"}
    ]
  }
}
```

Adding `{"predicate":"adult"}` means adult **and** female. It is a separate
condition, not a synonym or replacement. No new operation or second LLM call is
needed to express this combination.

Generate filter schema branches by property, owner and operator/operand shape.
For example, a collection equality branch is:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "property": {"enum": ["gender"]},
    "source": {"enum": ["entity"]},
    "operator": {"enum": ["eq", "ne"]},
    "value": {"type": "string", "enum": ["male", "female"]}
  },
  "required": ["property", "operator", "value"]
}
```

Omitted `source` continues to mean entity under the existing IR default.
Relationship branches require an explicit `source: relation`. Notice the
intentional existing naming difference: field-filter `source=relation` versus
projection `property_source=relationship`; do not rename either wire field.

| Operator form | Operand schema / authoritative check |
|---|---|
| `eq`, `ne` | A non-null scalar of the property's declared type/domain. |
| `gt`, `gte`, `lt`, `lte` | Scalar of an ordered permitted type; unordered closed domains cannot enable them in this phase. |
| `in` | Nonempty JSON array of homogeneous valid literals; converts to the existing immutable tuple IR. |
| `date_range` | Exactly two valid temporal endpoints of the same declared temporal kind, start < end, half-open interval. |
| `exists` | Existing boolean-or-null operand convention only; this migration does not change the existing missing-input execution policy. |
| Traversal `value_from=anchor` | Separate branch with no literal operand; validate both endpoints and compatible types. `value_property` must be declared and applicable to the existing traversal anchor. |
| Declared predicate | Exactly `{"predicate": name}`; no field value, source or comparison options. |

Collection filters continue to forbid dynamic anchor operands. Preserve the
current definition of the anchor used by traversal comparisons; do not rebase
it to an intermediate relative as part of this work.

Use `$defs` to share literal/domain fragments and `anyOf`/single-value `enum`
for the practical decoder subset. Avoid unbounded enumeration of every legal
relationship path. Keep the full standards-compatible schema as a validation
artifact, and explicitly list constraints omitted from the provider-compatible
projection. Runtime validation enforces the full contract regardless of provider.

Do not claim a supported keyword is enforced merely because the provider accepts
the schema. Ticket 6 must test decoded outputs and schema acceptance for the
actual pinned Ollama/model combination. Unknown keyword support falls back to
runtime rejection, never weaker semantic acceptance.

## 6. Runtime validation, data checks and disjoint predicates

Validation has two phases:

- **Before graph access:** check literals, applicability, ownership, operator
  shapes, operation input/output requirements and complete concept paths. Check
  both pairwise operands and every exclusion independently. The property must
  apply to every possible endpoint type, not merely one member of a type union.
- **After adapter reads:** validate actual non-null values against the declared
  type/domain before comparisons or computation. Known invalid values must not
  silently compare unequal and masquerade as a zero count. Keep adapter mappings
  explicit; do not infer values from private identity data.

Retain existing missing-input behavior and status envelopes. Missing/null filter
inputs continue to produce `filter_input_missing`; malformed present filter
values produce `filter_unsupported` with a typed diagnostic. Invalid projected
values use `property_unavailable` / `relation_property_unavailable`; invalid
computation inputs use the existing computation failure status as appropriate.
Details identify semantic property, stage and expected type, never raw private
values. This tightens some previously accepted data comparisons and requires an
explicit compatibility audit; no three-valued algebra, partial counts, new null
semantics or list-membership semantics are introduced here.

Top-level planner rejection codes remain compatible (`INVALID_PLAN`,
`UNKNOWN_PROPERTY`, etc.). Add structured internal detail such as
`INVALID_LITERAL_TYPE`, `VALUE_OUT_OF_DOMAIN`, `PROPERTY_NOT_APPLICABLE`, or
`CONTRADICTORY_PREDICATES`; do not change the HTTP response contract in this ticket.
Use the current bounded validation retry with a generic diagnostic if applicable.
It must not provide a guessed corrected plan or substitute one predicate for another.

`disjoint_with` is explicit, symmetric, refers only to known predicates, and
contains no self references. Reject `adult AND minor` only when applied to the
**same collection subject at the same filter site**. Different relatives,
intermediate versus final entities, exclusions, and pairwise operands are not
the same site. Do not drop or deduplicate conditions to make a plan executable.

Before declaring adult/minor disjoint, validate the existing policy: recognized
roles and date fallback must not assign both to the same complete input.
Incomplete/conflicting evidence remains a missing-input outcome, not proof of
one predicate. Disjointness asserts incompatibility, not exhaustiveness: never
infer `adult` from the absence of `minor`. This is a narrow contradiction check,
not a general satisfiability solver. Conflicting ordinary field filters retain
their existing semantics unless a later design explicitly changes them.

## 7. What schema and validation cannot prove

| Error | Schema projection | Authoritative runtime | Still needs semantic evaluation |
|---|---|---|---|
| `gender=17` or undeclared domain value | Rejectable with generated literal branches | Reject before reads | No |
| `gender > female` when ordering is not allowed | Rejectable | Reject before reads | No |
| `predicate=adult` combined with field-filter fields | Already structurally rejectable | Reject | No |
| Entity birth date requested as relationship property | Some owner/branch constraints expressible | Resolve owner/type and reject; never repair | No |
| `self -> member` | Complete path typing intentionally not enumerated | Already rejected by signatures | Choosing household scope remains an interpreter problem |
| Same-site `adult AND minor` | Cross-array constraint not required of decoder | Reject only with declared disjointness | Inferring gender instead remains an interpreter problem |
| Gender utterance compiled as `adult` alone | Legal | Legal | **Yes** |
| Legal `father -> spouse` instead of `spouse -> father` | Legal | Legal when signatures permit | **Yes** |
| `select(birth_date)` versus `annual_occurrence` | Both legal | Both legal | **Yes** |
| Omitted gender in response text | Not an interpreter schema concern | Ticket 3 uses executed condition descriptors | Renderer fidelity tests |

The scope distinction between current household and the speaker's residence must
remain explicit. Do not force every possessive phrase into one scope with text
matching. Similarly, do not coarsen clock semantics, consume prior assistant prose
as evidence, or overwrite current explicit references with prior focus.

## 8. Display contract for Ticket 3

The renderer receives the validated expanded request, result and resolved public
labels. It can build a condition descriptor containing the referenced scope,
ordered relationship path, each filter's attachment site, property/predicate,
operator, canonical value and labels. This is a view of existing semantic IR,
not a second semantic interpretation or persisted answer cache.

Every effective condition must appear in the composed description, including
filters inside concepts and relationship-edge filters. Preserve intermediate
attachment sites and exclusion/comparison distinctions. Do not flatten all gender
filters into a final-entity adjective. Existing array association and edge
quantification are unchanged. If natural prose is ambiguous, use explicit generic
scope/condition wording rather than a misleading kinship noun.

Examples of intended descriptions, with synthetic numbers:

- Household members where gender is female: 3 people.
- Household members who are adult and whose gender is female: 2 people.

The display labels introduced here enable this work; replacing `_count_noun()`
and implementing the renderer are Ticket 3, not part of this design ticket.

## 9. Compatibility and migration

### Version behavior

- Existing readers accept only ontology version 1 and reject new keys. Therefore
  new code supporting version 2 must ship **before** a version-2 ontology is activated.
- A new reader keeps version-1 behavior through an explicit legacy contract path;
  it must not pretend inferred types are explicit closed value domains.
- Version 2 requires explicit contracts for every model-advertised property.
  Undeclared raw catalog fields are excluded from that vocabulary. Internal
  resolver metadata remains internal. V1 fallback behavior stays confined to V1.
- `SemanticFactRequest` and its wire names/defaults stay stable. V2 can reject
  previously accepted ill-typed requests; this is intentional validation tightening,
  not full behavioral backward compatibility. Canonical valid requests should
  produce identical normalized plans, results and evidence.
- A version-2 property addition of an already implemented type needs ontology and
  adapter/catalog metadata, plus regenerated contracts; no prompt example, router
  branch, renderer sentence or operator code. A new semantic operation still needs
  its generic implementation and tests. Storage-schema evolution is not automatic.

### Phased implementation plan

1. Freeze Ticket 1 evidence and a V1 baseline. Audit all current properties, actual
   supported storage representations, undeclared exposed fields and custom
   ontology users. Use synthetic fixtures or separately authorized data access.
2. Implement the V2 loader/resolved contract and diagnostics behind explicit version
   selection. Verify V1 golden outputs unchanged. Generate capabilities/schema in
   offline tests before invoking a model.
3. Migrate the bundled ontology as a complete file. Validate all bindings, domain
   values, concept filters and predicate fallbacks. Compare V1/V2 capabilities and
   list every vocabulary restriction and stricter rejection.
4. Run deterministic compatibility tests and synthetic malformed-data probes.
   Never silently rewrite existing household records to pass the contract.
5. Hand frozen packages to Grok for matched real-model evaluation (Tickets 4/6).
   Record prompt/schema/token-size changes, first-pass versus retry outcomes,
   rendering separately, and latency. The ontology alone is not an accuracy claim.
6. Activate V2 only after the agreed gates pass. Retain a complete V1 package/config
   for rollback; do not roll back just the ontology while leaving incompatible
   bindings. Restart/reload follows existing process-local conversation behavior;
   no attempt to reconstruct lost trusted discourse is added.

No live deployment, graph migration, prompt edit or model call is authorized by
this design document itself. Production work uses `jkuang@home-cortex-0` over
Tailscale under the session's applicable authorization.

## 10. Change boundaries and owners

| File / component | Follow-on change | Explicitly unchanged |
|---|---|---|
| `schemas/semantic/ontology.yaml` | V2 property contracts and labels; audited predicate disjointness | Household records, instance identities, question lists |
| `semantic_ontology.py` | Versioned parsing, immutable descriptors, declaration validation | Concept meaning and complete declared paths |
| `semantic_facts.py` / `SemanticSchemaRegistry` | Resolved descriptors, capability/schema generation, shared validation lookups | IR field names, ownership rules, semantic repair prohibition |
| `schema_catalog.py` and adapters | Explicit binding/type compatibility where needed | Physical fields never exposed as a substitute for resolution |
| `operator_registry.py` | Reuse signatures; generic strict operand validation if required | New question-specific operators or duplicated policy definitions |
| Filter/read execution | Validate present values; preserve existing missing-input and edge association behavior | Automatic data correction, guessed values, partial counts |
| `ollama.py` | Ticket 5 may replace redundant capability guidance with concise grammar instructions | No sentence handlers or new default LLM call chain |
| Renderer | Ticket 3 consumes condition/display descriptors | LLM factual rendering or re-reading utterance text |
| Benchmarks/tests | Compatibility, malformed data, composition and decoder conformance | Loosened scoring, acceptance cases copied into examples |

Codex owns the loader/compiler/validation design and implementation. Grok owns
bounded evidence gathering and independent GPU evaluation. Tickets 3–6 consume
this design; no simultaneous prompt/ontology mutation during a frozen evaluation.

## 11. Test matrix and acceptance gates

| Area | Positive/control | Negative or counterexample | Expected gate |
|---|---|---|---|
| V1 compatibility | Existing ontology/requests and normalized results | Unknown V1 extension key | Old behavior preserved; key rejected |
| V2 loader | Complete type/domain/labels | Unknown kind/operator, missing applicability, domain alias collision | Deterministic configuration failure |
| Binding | Declared date with unknown observed kind and known binding | Known string-only binding contradicts date; undeclared physical field | Fail conflicting binding; do not advertise undeclared field |
| Literals | Female enum, integer, finite number | `17` for gender; `true` as integer; NaN; malformed date | Reject before graph reads |
| Membership/range | Homogeneous `in`; ordered two-date interval | Empty/mixed `in`; reversed range; invalid leap date | Schema where expressible plus runtime rejection |
| Missing/invalid data | Valid present field | Null/absent versus malformed present value | Existing missing status; explicit invalid-value failure, never zero by accident |
| Ownership | Entity birth date; spouse-edge start date | Swapped `property_source` or wrong filter source | Reject, no fallback to other owner |
| Applicability | Property valid for all reachable types | Mixed endpoint types where only one owns property | Reject unsupported reference/property combination |
| Scope | `current_household -> member`; explicit residence traversal | `self -> member` | Preserve correct plans; reject wrong path |
| Concept composition | Atomic in-law; explicit spouse + in-law | Removing a duplicate-looking hop changes meaning | Expand verbatim; never collapse legitimate nested paths |
| Intermediate filters | Gender filter at the intended relative | Moving it to the final relative | Distinct plans and counterexample outputs |
| Anchor comparison | Existing sibling anchor semantics | Collection anchor operand; incompatible anchor property | Preserve valid anchor; reject unsupported forms |
| Predicate distinction | Gender-only; adult-only; adult AND female | Gender replaced by adult | Contract accepts individually legal plans; semantic benchmark must catch substitution |
| Disjointness | Adult at one site, minor at another | Adult AND minor at same collection site | Allow different sites; reject declared contradiction |
| Missing predicate inputs | Complete policy/date evidence | Conflicting roles or absent fallback | Preserve existing unresolved-input outcome |
| Shape/operations | Count set; select set; projection each; pairwise comparison | Scalar/collection misuse, absent pairwise operand | Existing structural gates retained |
| Display | All filters, exclusions and edge ownership described | Adult label hides gender; final label hides intermediate constraint | Ticket 3 fidelity tests, including zero results |
| Conversation | Same request across different histories; two speakers/homes/agents | Prior predicates leak; lost discourse reused | Isolation/clarification unchanged; semantic errors scored separately |
| New property extension | Synthetic `preferred_language` string/domain on person, with adapter metadata | Same request without its declaration | Declarative addition only; absent contract rejected |
| Serialization | Equivalent maps under multiple hash seeds | Shared object mutated by caller | Stable fingerprints; no cache leakage |
| Provider conformance | Generated schema accepted by pinned provider | Deliberately ill-typed generated output | Full runtime rejects regardless of provider behavior |
| Held-out semantics | Unseen combinations and paraphrases, multiple synthetic households | Correct count from wrong population; birthday wrong operation | Full-plan scoring, not `found`/number equality |
| Cost | Baseline/candidate same model/data/clock, bounded schema size | Extra retries or decoder/schema overhead | Pre-agreed latency/token gates in Ticket 6; no promised speedup |

Completion of Ticket 2 means this design, example, boundaries, compatibility plan
and matrix are reviewable. It does not mean implementation or real-model accuracy
has passed. Implementation must not claim to solve Ticket 1's legal-but-wrong
plans merely by adding types. Any remaining such failures are returned to the
semantic evaluation/model workstream rather than repaired downstream.
