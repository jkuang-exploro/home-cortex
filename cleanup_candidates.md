# Remaining semantic duplication audit

Date: 2026-09-04

Scope: `semantic_facts.py`, `agent_service.py`, `edge_schema.py`, `db.py`,
`display.py`, `calculate.py`, `calendar.py`, `fact_benchmark.py`, plus
imports/callers required to classify those symbols. No redesign.

One production deletion was applied: unused
`SemanticSchemaRegistry.semantic_property_kinds`. No other code was changed.

## Counts

| Surface | Count | What was counted |
|---|---|---|
| Semantic entry points | **4** | `SemanticFactService.try_answer` (Tier 0 + planner), `HouseholdFactEngine.execute`, `RetrievalService.retrieve` (`POST /v1/retrieve`) |
| Entity-resolution paths | **5** | `EntityResolver` contextual IDs, `resolve_entity_alias`, `search_entities`, `get_entity`, Open WebUI identity map |
| Compatibility / fallback paths | **12** | listed under P0–P2 |
| Estimated removable production LOC | **~180–250** | P0 + high-confidence P1, not including renderer copy that must be replaced rather than dropped |

`calculate.py`, `calendar.py`, and `db.py` have no remaining semantic/profile
duplication. They were inspected and are omitted from the candidate list.

`facts.py`, `fallbacks.py`, `request_analysis.py`, and `memorable_dates.py`
are already gone from source (pycache only).

---

## P0 — clearly obsolete / duplicated

### 1. `RuntimeSchemaCatalog.entity_aliases` — `schema_catalog.py`

- **Production callers:** none. `from_data_dir` writes the set; nothing reads it.
- **Test-only callers:** none.
- **Overlaps:** `record_aliases()` / `resolve_entity_alias`, which already
  resolve stored names at query time.
- **Action:** DELETE field and the collection loop in `from_data_dir`.
- **Uncertainty:** low. Tests construct the catalog with two arguments and
  rely on the default empty set.

### 2. `_latest_user_text` — `semantic_facts.py`

- **Production callers:** `SemanticFactService.try_answer`.
- **Test-only callers:** none directly.
- **Overlaps:** `text.latest_user_message` (identical contract).
- **Action:** MIGRATE+DELETE (`try_answer` should call `latest_user_message`).
- **Uncertainty:** low.

### 3. `SpeakerContext` — `grounding.py`

- **Production callers:** `AgentRequestContext.speaker`, then
  `EntityResolver._resolve` (`speaker_id`, `household_id` only).
- **Test-only callers:** none.
- **Overlaps:** `AgentRequestContext.caller_entity_id` /
  `household_id`. `SpeakerContext.locale` and `.timezone` are never read.
- **Action:** MIGRATE+DELETE. Read the context fields directly.
- **Uncertainty:** low. `grounding.py` is already only request context; the
  extra type is leftover from the old grounding executor.

### 4. Contextual type map, three copies — `semantic_facts.py`

`{"self": "person", "assistant": "person", "current_household": "address"}`
appears in `SemanticSchemaRegistry.validates`,
`_base_entity_types`, and
`SemanticFactPlanner._normalize_context_reference`.

- **Production callers:** planner normalize + schema validation + type
  inference.
- **Test-only callers:** indirect via planner/engine tests.
- **Overlaps:** itself. Ontology does not own these context kinds.
- **Action:** MIGRATE+DELETE the extra copies; keep one module-level constant.
  Do not invent a new abstraction.
- **Uncertainty:** low.

### 5. Inverse-relation lookup, three copies

| Implementation | File |
|---|---|
| `EdgeSchemaRegistry.resolve` | `edge_schema.py` — **authoritative** |
| `SemanticSchemaRegistry.physical_relation` | `semantic_facts.py` |
| `RuntimeSchemaCatalog.relation_field_type` | `schema_catalog.py` |

Ontology `base_relations` names the inverse (`child_of`, `has_resident`),
then `physical_relation` re-finds the stored edge (`parent_of`, `lives_in`).
`relation_field_type` does the same scan when given an inverse name.

- **Production callers:** executor, validator, catalog field typing.
- **Test-only callers:** `test_semantic_ontology.py` asserts the derived
  `(parent_of, "in")` / `(lives_in, "in")` pairs.
- **Overlaps:** `EdgeSchemaRegistry.resolve`.
- **Action:** MIGRATE+DELETE the two reimplementations; call `resolve`.
- **Uncertainty:** low for the catalog helper. Medium for
  `physical_relation`, which also applies ontology `base_relations` and
  direction. That composition should stay; only the inverse scan should go.

### 6. JSON-graph dispatchers (duplicate of retrieval)

| Symbol | File | Role |
|---|---|---|
| `_JsonGraphDispatcher` | `fact_benchmark.py` | CLI JSON backend |
| `FixtureGraphDispatcher` | `tests/test_semantic_facts.py` | tests |

Both reimplement `get_entity` / `resolve_entity_alias` / `get_relationships`,
and both treat `search_entities` as an alias of `resolve_entity_alias`.

- **Production callers:** none (`fact_benchmark` is a CLI).
- **Test-only callers:** semantic-fact tests; planner benchmark imports
  `_JsonGraphDispatcher`.
- **Overlaps:** `RetrievalService` + `ToolDispatcher`.
- **Action:** MIGRATE+DELETE `FixtureGraphDispatcher` in favor of
  `_JsonGraphDispatcher`. Keep one JSON adapter for offline benchmarks.
- **Uncertainty:** low. `_summary` field sets already differ slightly
  (`gender` vs `ENTITY_SUMMARY_FIELDS`).

---

## P1 — likely removable after small caller migration

### 7. `RetrievalService.retrieve` + `RetrievedContext` + `POST /v1/retrieve`

- **Production callers:** `api.retrieve`.
- **Test-only callers:** `tests/test_retrieval.py`, `tests/test_api.py`.
- **Overlaps:** `SemanticFactService.try_answer` is the household-fact
  answer path. `retrieve()` is a keyword dump of nodes/edges via
  `search_entities` + `get_relationships`.
- **Action:** MIGRATE+DELETE if `/v1/retrieve` has no remaining client.
  That is the main leftover semantic *entry point*.
- **Uncertainty:** medium. It is a live HTTP route. Confirm no external
  caller before deleting.

### 8. `RetrievalService.search_entities`

- **Production callers:** `retrieve()`, `ToolDispatcher._search_entities`.
  `ModelLoop` strips `GRAPH_TOOL_NAMES`, so the conversation model never
  calls it. `EntityResolver` uses `resolve_entity_alias`, not search.
- **Test-only callers:** retrieval, hosted_by, tools, api tests.
- **Overlaps:** `resolve_entity_alias` (exact alias/appellation). Search is
  substring-on-id/name.
- **Action:** MIGRATE+DELETE with `/v1/retrieve`, or KEEP if a non-semantic
  lookup API is still required. Do not keep it as a second named-entity
  resolver for facts.
- **Uncertainty:** medium. Same client question as #7.

### 9. `include_residents` / `_attach_residence_rosters` — `retrieval.py`

- **Production callers:** `get_relationships` default `True`; tool schema
  default `True`. `EntityResolver` always passes `False`.
- **Test-only callers:** `test_retrieval.py`, `test_tools.py`.
- **Overlaps:** semantic `member` (`lives_in` inverse) traversal. The
  roster-on-`lives_in` attachment is a pre-IR household special case.
- **Action:** MIGRATE+DELETE once graph tools are no longer a public
  household-roster API. Semantic facts already disable it.
- **Uncertainty:** low if #7/#8 retire; otherwise KEEP as tool behavior.

### 10. `schema_catalog.record_aliases` extra physical name fields

Collects `preferred_name`, `nickname`, `english_name`, `chinese_name`,
`first_name`, `last_name` in addition to `name` / `aliases` /
`display_name`.

- **Production callers:** `resolve_entity_alias`, catalog construction.
- **Test-only callers:** retrieval + semantic-fact alias tests.
- **Overlaps:** ontology `display_name` / `given_name` / `family_name`
  field lists. Alias resolution bypasses the semantic property map.
- **Action:** MIGRATE+DELETE the extra physical-name special cases after
  confirming deployed nodes only need ontology fields + `aliases`.
- **Uncertainty:** medium. Deployed `person.json` still has `first_name` /
  `last_name`.

### 11. `FactRenderer._relation_noun` / `_relation_label` / `_counts_children`

Hardcoded kinship labels: spouse/wife/husband, child/son/daughter,
parent/father/mother, member, residence; plus “孩子” via `child` or
`household_role=minor_dependent`.

- **Production callers:** Chinese renderer only.
- **Test-only callers:** indirect via answer-string tests.
- **Overlaps:** `schemas/semantic/ontology.yaml` `reference_concepts` and
  `collection_predicates.minor` (already the planner vocabulary).
- **Action:** MIGRATE+DELETE the parallel kinship tables. Renderer can
  keep a small display-label map keyed by the same semantic relation
  names; do not grow a second kinship graph.
- **Uncertainty:** low that they duplicate the ontology. Medium on the
  smallest replacement (labels vs concept aliases).

### 12. Renderer special cases — `FactRenderer._zh`

| Special case | Trigger |
|---|---|
| Marriage date | `spouse` + `start_date` |
| Adult/minor missing evidence | `"adult"` / `"minor"` in `missing_requirements` |
| Age extrema | `argmin`/`argmax` + `birth_date` |
| Birthday countdown | `annual_occurrence` + `mode=days` |
| Address | `full_address` |

- **Production callers:** fact rendering after a generic executor result.
- **Test-only callers:** semantic-fact answer tests.
- **Overlaps:** generic `_zh` / `_en` value formatting. Marriage/adult/minor
  copy is domain-specific on top of already-generic IR.
- **Action:** MIGRATE+DELETE marriage and adult/minor branches first; they
  are the leftover profile special cases. KEEP birthday/address only if
  measured copy quality requires them.
- **Uncertainty:** medium. These are user-visible strings, not query paths.

### 13. `_property_label` — `semantic_facts.py`

Chinese labels for `birth_date`, `display_name`, `given_name`,
`family_name`, `form_of_address`, `gender`, `household_role`,
`start_date`, `end_date`, `full_address`, `adult`, `minor`.

- **Production callers:** Chinese missing-property messages.
- **Test-only callers:** none directly.
- **Overlaps:** ontology property `aliases` (already Chinese+English).
- **Action:** MIGRATE+DELETE in favor of `ontology.properties[name].aliases[0]`
  plus predicate names. No new helper type.
- **Uncertainty:** low.

### 14. `SemanticFactRequest` default `completed_years` → `birth_date`

- **Production callers:** request validation mutates missing property to
  `birth_date`.
- **Test-only callers:** age-plan tests.
- **Overlaps:** ontology `birth_date` and operator `completed_years`
  (already generic). This is the last age-specific IR default.
- **Action:** MIGRATE+DELETE; require `property` like every other
  transform. Planner prompt already says “date property”.
- **Uncertainty:** low if planner eval still passes without the default.

### 15. `SemanticSchemaRegistry._property_candidates` physical pass-through

Unknown semantic names that are not ontology aliases are accepted as
physical field names (`favorite_color` test).

- **Production callers:** `physical_property` / `relation_property`.
- **Test-only callers:** `test_semantic_facts.py` open-world field test.
- **Overlaps:** ontology `properties.fields` is supposed to be the alias
  map. Pass-through reopens physical names to the planner.
- **Action:** MIGRATE+DELETE the `(semantic,)` fallback once advertised
  properties are only ontology + unaliased catalog fields that are not
  in `_aliased_physical_properties`. Keep the unaliased-catalog branch.
- **Uncertainty:** medium. Removing it tightens protocol; confirm planner
  eval does not rely on emitting storage names.

### 16. `_FactExecution.records` `hasattr(..., "dispatch_internal")`

- **Production callers:** live `ToolDispatcher` has `dispatch_internal`.
- **Test-only / CLI callers:** JSON dispatchers have only `dispatch`, so
  this branch is for them.
- **Overlaps:** `ToolDispatcher.dispatch_internal` (the real internal
  gate for `resolve_entity_alias`).
- **Action:** MIGRATE+DELETE the `hasattr` fallback after JSON adapters
  grow a no-op/public `dispatch_internal`, or after facts always go
  through `ToolDispatcher`.
- **Uncertainty:** low.

### 17. `canonical_utterances` — `TierZeroSemanticParser`

- **Production callers:** none.
- **Test-only callers:** `tests/test_semantic_planner_benchmark.py`.
- **Overlaps:** the four exact strings already in `parse()`.
- **Action:** KEEP as the documented Tier-0 surface, or inlined into the
  test. Not worth a helper unless a second caller appears.
- **Uncertainty:** low. Authority report retains this method on purpose.

### 18. `steward` `ALLOWED_TOOLS` graph names vs `ModelLoop` strip

Steward allowlist still includes `get_entity`, `search_entities`,
`get_relationships`. `ModelLoop` removes `GRAPH_TOOL_NAMES` and rejects
those calls. Semantic facts use the same dispatcher internally.

- **Production callers:** `ToolDispatcher` construction; model never sees
  the tools.
- **Test-only callers:** `test_agents.py` asserts the allowlist.
- **Overlaps:** `GRAPH_TOOL_NAMES` + internal `resolve_entity_alias`.
- **Action:** KEEP the dispatcher grants (facts need them). MIGRATE the
  steward allowlist / tool JSON so graph tools are not described as
  model-facing. Error text still says “require a grounding plan”.
- **Uncertainty:** low.

---

## P2 — uncertain, judgment required

### 19. `validation_code` vs `validates` — `SemanticSchemaRegistry`

- **Production callers:** planner uses `validation_code`; engine and
  `try_answer` call `validates` again.
- **Test-only callers:** both, separately.
- **Overlaps:** each other. `validation_code` ends with
  `"VALID" if self.validates(request)`.
- **Action:** KEEP both for now. Folding them is a behavior-sensitive
  refactor, not a cleanup.
- **Uncertainty:** high.

### 20. `EntityResolver._filter_entities` vs `HouseholdFactEngine._filter_collection`

Traversal-step filters vs collection filters. Same
`evaluate_predicate` / `physical_property` work; different statuses
(`property_unavailable` vs `filter_input_missing`) and collection
predicates only on the engine side.

- **Production callers:** resolver + engine.
- **Test-only callers:** filter/kinship tests.
- **Overlaps:** each other.
- **Action:** KEEP. Merging would be a redesign.
- **Uncertainty:** high.

### 21. `EntityResolver` injectable on `HouseholdFactEngine`

- **Production callers:** never passed; default-constructed.
- **Test-only callers:** none construct a custom resolver.
- **Overlaps:** the default `EntityResolver`.
- **Action:** KEEP the class. DELETE the unused `resolver=` parameter
  if no test needs it.
- **Uncertainty:** low for the parameter, high for touching the class.

### 22. `resolve_display_name` / `resolve_person_reference` — `display.py`

- **Production callers:** `semantic_facts._name`, greetings, `model_loop`.
- **Test-only callers:** `test_display.py`.
- **Overlaps:** `DisplayNameResolver.resolve`. The functions are
  pass-through wrappers (empty resolver for object-shaped records).
- **Action:** KEEP. Wrappers are the public display API; not semantic
  duplication.
- **Uncertainty:** low.

### 23. `EdgeSchemaRegistry.load_default` / `SemanticOntology.load_default`

Path fallbacks: package `schemas/`, `/app/schemas/`, optional `data_dir`.

- **Production callers:** retrieval/ingestion (edge, when registry omitted);
  `SemanticSchemaRegistry` (ontology).
- **Test-only callers:** `FixtureGraphDispatcher` uses
  `EdgeSchemaRegistry.load_default`.
- **Overlaps:** `from_directory` / `from_file`, which API and benchmarks
  already call with explicit paths.
- **Action:** KEEP. Deployment path search, not a second schema.
- **Uncertainty:** low.

### 24. Collection-predicate `fallback` (age from `birth_date`)

- **Production callers:** `HouseholdFactEngine._semantic_predicate_matches`.
- **Test-only callers:** adult/minor count tests.
- **Overlaps:** none. This is declared ontology policy, not a second
  parser.
- **Action:** KEEP.
- **Uncertainty:** low.

### 25. Planner predicate-as-property + `default_scope_relation` normalize

`SemanticFactPlanner._normalize_collection_predicates` rewrites
`property=adult` into `predicate=adult` and may inject `member` on
`current_household`.

- **Production callers:** every planner plan.
- **Test-only callers:** planner tests.
- **Overlaps:** ontology `collection_predicates` aliases / scope.
- **Action:** KEEP. Compatibility for model output, not a second executor.
- **Uncertainty:** medium if planner accuracy no longer needs it.

### 26. `ingestion._validate_node_name` “legacy alias lists”

Accepts `name: [str]` and localized objects.

- **Production callers:** ingest.
- **Test-only callers:** ingestion tests.
- **Overlaps:** `record_aliases` / display localized-name handling.
- **Action:** KEEP until node documents are one shape.
- **Uncertainty:** medium.

### 27. `fact_benchmark.QUESTIONS` vs `benchmarks/semantic_planner_eval.yaml`

- **Production callers:** none (CLI).
- **Test-only callers:** fact-benchmark tests use the module.
- **Overlaps:** planner eval dataset is the authoritative utterance set.
- **Action:** MIGRATE+DELETE the hardcoded `QUESTIONS` / `SPEAKER_CASES`
  once the CLI can point at the eval YAML. Not production LOC.
- **Uncertainty:** low.

### 28. `agent_service` re-exports of `model_loop` types

`MAX_AGENT_STEPS`, `AgentLimitError`, `AgentResult`, `AgentStreamingError`.

- **Production callers:** `api.py` imports them from `agent_service`.
- **Test-only callers:** `test_agent_service.py`.
- **Overlaps:** `model_loop`.
- **Action:** KEEP. Compatibility façade, not semantic duplication.
- **Uncertainty:** low.

### 29. `AgentService.answer` wrapping `answer_messages`

- **Production callers:** tests / simpler clients.
- **Overlaps:** `answer_messages`.
- **Action:** KEEP.
- **Uncertainty:** low.

### 30. `TierZeroSemanticParser`

Exact-match latency path for six utterances. Authority report retains it.

- **Production callers:** `SemanticFactService.try_answer` when
  `tier_zero_enabled`.
- **Test-only callers:** many; planner benchmark compares Tier 0 to planner.
- **Overlaps:** `SemanticFactPlanner` covers the same six plans.
- **Action:** KEEP until a measured removal. Not leftover phrase-table
  code; the specialized handlers are already gone.
- **Uncertainty:** low.

### 31. `identity.resolve_user_entity_id` vs `agent_service._normalized_identity`

- **Production callers:** `api._resolve_identity` →
  `AgentService._normalized_identity`.
- **Test-only callers:** `test_identity.py`, agent-service tests.
- **Overlaps:** both produce a person id, at different layers (headers vs
  already-loaded record).
- **Action:** KEEP. Not duplicate resolvers.
- **Uncertainty:** low.

---

## Not candidates

| Symbol | Why |
|---|---|
| `SemanticFactService` / `HouseholdFactEngine` / `SemanticFactPlanner` | Authoritative pipeline |
| `EntityResolver` | Authoritative graph resolution |
| `SemanticOntology` + `ontology.yaml` | Authoritative kinship / property / predicate vocabulary |
| `EdgeSchemaRegistry` | Authoritative stored-edge schema |
| `resolve_entity_alias` | Authoritative named-entity path used by facts |
| `DisplayNameResolver` | Presentation, not fact resolution |
| `operator_registry` vs `calculate.evaluate_expression` | Collection IR vs allowlisted arithmetic |
| `calendar.parse_calendar_datetime` | Provider window parsing, not fact dates |
| `Database` | Surreal client wrapper |

---

## Suggested order if cleanup continues

1. Delete `entity_aliases` and switch `_latest_user_text` / `SpeakerContext`.
2. Point inverse lookup at `EdgeSchemaRegistry.resolve`.
3. Collapse the two JSON-graph dispatchers.
4. Decide whether `/v1/retrieve` + `search_entities` + `include_residents`
   are still a product surface. That is the remaining second semantic
   entry point and the remaining second named-entity path.
5. Point renderer kinship/property labels at ontology names.
6. Drop `completed_years` → `birth_date` default and the physical-name
   pass-through only after planner eval stays green.
