# Test coverage and behavioral gap audit

Date: 2026-09-04

Scope: architectural invariants of `SemanticFactService` / `HouseholdFactEngine` /
`EntityResolver` / `SemanticFactPlanner`. No production changes. No new coverage
framework: `pytest-cov` / `coverage.py` are not installed, and `pyproject.toml`
does not configure them. Branch conclusions below are from reading production
code against the existing suite.

Existing mass: `tests/test_semantic_facts.py` collects **105** cases, mostly
oracle-planned happy paths against production `data/` (the Kuang household).
The planner eval oracle (`test_planner_only_oracle_executes_full_eval_and_checks_tier0_parity`)
already executes **111** accepted plans with Tier 0 off and asserts
`result.status == "found"`. That suite does **not** protect failure statuses,
canonical IDs, or the production graph stack.

---

## Invariant coverage

| Invariant | Coverage | Why |
|---|---|---|
| speaker-relative self | well covered | Tier-0 `我是谁` / `Who am I?` through `AgentService`; planner-disabled `self` in `test_tier_zero_disabled_preserves_semantic_correctness`; per-speaker `我儿子是谁` with entity IDs |
| direct alias resolution | well covered | `匡德伦` / `德伦` / `Dylan` / `Dylan Kuang` converge on `person:dylan_kuang`; scoped appellation; `RetrievalService.resolve_entity_alias` separately |
| relational entity resolution | well covered | named-entity + path (`巴璞的儿子`) and `self` + path share canonical IDs |
| multi-hop kinship | well covered | 岳父/岳母, 孙子/外孙, 哥哥 (`value_from=anchor`); in-law missing path |
| semantic property lookup | well covered | `birth_date` via `dob`/`birthday`, `full_address`, open-world catalog field |
| relationship property lookup | well covered | spouse `start_date`, duration, missing start, speaker-relative marriage |
| Tier-0 independence | well covered | `tier_zero_enabled=False` + `DisabledParser`; eval 6/6 Tier-0/Planner parity |
| planner fallback | partially covered | Tier-0 miss → planner is exercised; `requires_fact=false` only at `AgentService`; planner crash / exhausted `MALFORMED_OUTPUT` / `INVALID_PLAN` untested through `try_answer` |
| filter/count | partially covered | adult/minor/gender counts and role-then-age fallback exist; adult **renderer** for missing evidence (`semantic_facts.py:1929`) is not asserted; minor missing and generic property-filter missing are absent |
| argmin/argmax | partially covered | household extrema and generic `fixture_score`; pairwise age compare only via rendered Chinese; equal-age branch untested |
| date calculations | partially covered | `annual_occurrence` / `completed_years` through the engine; leap-day only in `operator_registry`; `date_difference` never executed; duration asserts `value > 4000` |
| missing evidence | partially covered | property / computation / relation-property / empty-count are good; `caller_context_missing` only for Tier-0 `我是谁`; no missing `household_id`; adult/minor copy untested |
| ambiguity | partially covered | grandson asserts candidate **IDs**; duplicate name and multiple sons assert **display strings** only |
| malformed planner output | partially covered | retry-then-success, `UNSUPPORTED_OPERATION`, `UNKNOWN_PROPERTY` (no retry), `UNKNOWN_RELATION`, `MODEL_ORIGINATED_ENTITY_ID`; no exhausted malformed, no `INVALID_PLAN` via planner, no `NOT_A_FACT` at `try_answer` |
| database failure | not covered | `_FactExecution.records` maps `ok is not True` to `computation_impossible`; `_JsonGraphDispatcher` always returns `ok: True` |

Duplicatively covered: speaker-relative Dylan, household extrema, marriage start,
birthdays, and address lookup appear in the 41-case string parametrize, the
Tier-0-disabled ID suite, the 6-case “open world” string parametrize, the 111-case
eval oracle, and `test_agent_service_reported_queries_use_semantic_planner`.

---

## P0 missing tests

These are holes in the architecture, not missing paraphrases. Prefer engine or
`try_answer` assertions on **status + canonical entity IDs + diagnostics**, not
rendered copy, except where the copy *is* the contract (adult/minor missing
evidence).

1. **Database / dispatcher failure through the fact engine**
   - Production: `_FactExecution.records` (`semantic_facts.py` ~2288–2294) raises
     `_FactFailure("computation_impossible")` when `dispatch_internal` returns
     `ok is not True`.
   - Gap: every semantic-fact test uses `_JsonGraphDispatcher`, which always
     returns `ok: True` (unknown tools become an empty list, which looks like
     `entity_not_found`).
   - Assert: `result.status == "computation_impossible"`; rendered text has no
     exception type, SQL, or dispatcher payload.

2. **`INVALID_PLAN` through the planner, with no structural retry**
   - Production: `validation_code` returns `INVALID_PLAN` when named checks pass
     but `validates()` is false (e.g. `argmin` on `display_name`). Semantic
     failures must not retry (`test_semantic_validation_failure_is_classified_without_retry`
     only covers `UNKNOWN_PROPERTY`).
   - Current tests only call `schema.validates(...) is False`.
   - Assert: `SemanticPlannerFailure.diagnostics.validation_result == "INVALID_PLAN"`
     and `attempt_count == 1`.

3. **Exhausted malformed planner output**
   - Production: two attempts, then `MALFORMED_OUTPUT` (or `UNSUPPORTED_OPERATION`
     if the loc is `operation`).
   - Current: first payload is malformed, second succeeds
     (`test_planner_retries_once_for_structural_failure`).
   - Assert: both attempts fail; `try_answer` returns
     `semantic_plan_unsupported`; `attempt_count == 2`; no graph calls.

4. **`requires_fact=false` at `SemanticFactService.try_answer`**
   - Production: `try_answer` returns `None` and must not execute the graph
     (`semantic_facts.py` ~2169).
   - Current: only `AgentService` fakes (`plan_semantic_fact` →
     `requires_fact: False`). Diagnostics `NOT_A_FACT` are never asserted.
   - Assert: `try_answer(...) is None`; dispatcher `calls == []`;
     planner `validation_result == "NOT_A_FACT"`.

5. **`caller_context_missing` on a speaker-relative relation, not just `我是谁`**
   - Production: `EntityResolver` raises when `kind == "self"` and
     `caller_entity_id is None`.
   - Current: `test_speaker_identity_without_authentication_fails_clearly` hits
     **Tier 0** `我是谁？` only. A planner-built `self → child` with no caller
     is untested (and would be the path if Tier 0 is disabled).
   - Assert: status `caller_context_missing`; no `get_relationships` call.

6. **Adult / minor `filter_input_missing` renderer (the cited branch)**
   - Production: `FactRenderer._zh` lines 1927–1931. Engine status for **adult**
     is already asserted in
     `test_status_predicate_prefers_authoritative_role_then_falls_back_to_age`
     (`missing_requirements[:1] == ("adult",)`), but the Chinese copy is not,
     and **minor** missing evidence is never executed.
   - Assert:
     - adult → `"目前缺少足够的年龄或家庭角色资料，无法确定成年人数量。"`
     - minor → `"目前缺少足够的年龄或家庭角色资料，无法确定未成年人数。"`
     - status `filter_input_missing` in both cases.

7. **Ambiguity must pin candidate IDs**
   - `test_multi_match_grandson_is_ambiguous` already does this.
   - `test_duplicate_exact_name_is_ambiguous` and
     `test_multiple_matching_children_are_ambiguous` only check substrings
     (`匡德伦`, `次子`, “找到多个符合条件”).
   - Strengthen those two (do not add new paraphrases):
     `result.status == "ambiguous"` and candidate IDs
     `{person:dylan_kuang, person:other_dylan}` /
     `{person:dylan_kuang, person:second_son}`.

8. **One engine path through `RetrievalService`, not `_JsonGraphDispatcher`**
   - Production graph access is `ToolDispatcher.dispatch_internal` →
     `resolve_entity_alias` / `get_relationships` / `get_entity`.
   - Semantic-fact tests reimplement that contract in the JSON adapter
     (`cleanup_candidates.md` item 6). Alias and kinship can pass in JSON and
     fail against retrieval (or the reverse).
   - One in-memory case is enough: ingest `tests/static_test_data`, resolve a
     named alias to a canonical ID, then traverse one relation to a canonical
     ID. Do not replay the Kuang household.

---

## P1 useful tests

Still high-signal, but they close partial coverage rather than a silent
architectural hole. Cap this list; do not grow a second phrase table.

9. **Missing `household_id` on `current_household`**
   - Resolver raises `entity_not_found` (`semantic_facts.py` ~1159–1161).
   - Untested. Assert status and zero useful roster output.

10. **Planner unexpected exception**
    - `try_answer` `except Exception` (`semantic_facts.py` ~2153) returns
      `_failure_answer` without diagnostics.
    - Untested. Interpreter `raise RuntimeError("boom")` →
      `semantic_plan_unsupported`; answer text must not include `"boom"`.

11. **Equal-age `argmin`/`argmax` with `other`**
    - Renderer branches on `result.value["equal"]` (zh ~1993, en ~2074).
    - Pairwise compare is only covered by “巴璞年龄比匡健大”. Give both people
      the same `dob` and assert `equal is True` plus entity IDs.

12. **`date_difference` through the engine**
    - Advertised in capabilities; operator registry lists it; no engine test
      and no eval plan uses it. One `property_source=relationship` case next
      to the existing duration test is enough.

13. **Duration with a frozen clock, exact integer**
    - Replace `assert duration_result.value > 4000` (household-specific and
      clock-relative) with exact days from `context.current_time` and
      `2014-05-04`, still asserting `evidence.entity_ids`.

14. **Generic `filter_input_missing` (not adult/minor)**
    - `_entity_filter_matches` raises when a physical field is absent.
    - Count household members with `gender` missing on one person →
      `filter_input_missing` and the generic renderer
      (`semantic_facts.py:1933`), not the adult/minor sentences.

15. **Conflicting household roles**
    - `_semantic_predicate_matches` raises `filter_input_missing` when
      recognized roles disagree (`semantic_facts.py` ~1758). Untested.

16. **Birthday-today through the engine + renderer**
    - Operator registry already returns `0` days. Renderer “生日就是今天” /
      “The birthday is today.” is untested. One `annual_occurrence` with
      `mode=days` and `current_time` on the stored date.

17. **English locale is not accidentally Chinese**
    - Every `test_semantic_facts` context uses `locale="zh"`, including
      `"when was Dylan Kuang born"` / `"what is my date of birth"`.
    - One found + one missing-property case with `locale="en"` asserting
      English renderer keys, not 家庭资料 copy.

18. **Empty household `count` is zero, not missing entity**
    - Empty **list** is tested (`test_empty_household_list_has_a_clear_response`).
    - Empty **count** is not. Mirror the empty-child-count test on
      `current_household → member`.

19. **Anchor property missing on older-sibling resolution**
    - `_anchor_property` raises `property_unavailable` when the speaker has
      no `birth_date`. `我哥哥是谁` is only tested on the happy path.

20. **Type-incompatible extreme is `INVALID_PLAN` at `validation_code`**
    - If P0 item 2 goes through the planner, this is redundant. Otherwise
      change `test_semantic_validator_rejects_type_incompatible_extreme` to
      assert `validation_code == "INVALID_PLAN"` rather than only `validates is False`.

Do **not** add: more birthday paraphrases, more address paraphrases, more
“谁最年长” variants, `unit_conversion` / `first` / `last` / `latest` unless a
real planner eval case needs them, or English copies of every Chinese failure
string.

---

## Redundant tests that can be deleted or consolidated

Delete or fold these. They do not protect an extra invariant.

| Test | Problem | Action |
|---|---|---|
| `test_canonical_and_planner_composed_facts_execute_deterministically` (41 rows) | Asserts rendered substrings (`匡健`, `五个人`, `9岁`, `58天`, `12745 Droxford St`). Mixes Tier-0 exact forms with planner paraphrases. Pins production household values. Duplicates the 111-case eval executor. | Delete the parametrize. Keep at most 2–3 **ID** cases if they are not already in `test_all_static_and_relational_references_converge_on_canonical_person` / extrema / marriage tests. |
| `test_semantic_birth_date_plan_never_contains_physical_alias` | Asserts `_fixture_plan`, a **test-only** phrase table, never production planner or executor. | Delete. Eval YAML already has `birth_date` not `dob`. |
| `test_birthday_intents_have_distinct_generic_plans` | Same: tests `_fixture_plan` branches. | Delete. Distinct operations are already in eval plans + `test_missing_birth_date_is_computation_input_missing_for_transform`. |
| `test_in_law_is_composed_from_existing_generic_relations` | Tests `_fixture_plan` path `["spouse","parent"]`. Ontology already asserts `father_in_law`. | Delete. |
| `test_marriage_date_is_a_planner_owned_relationship_property` | Tests `_fixture_plan` plus “Tier 0 does not parse 我们什么时候结婚的”. | Delete. Eval + `test_relationship_properties_and_duration_use_the_spouse_edge` cover it. Keep the one-liner `TierZeroSemanticParser().parse(...) is None` on extrema if still wanted. |
| `test_open_world_plans_execute_with_tier_zero_disabled` (6 string rows) | Oracle interpreter + `expected in answer.text` (`匡悠然`, `三位成年人`, `2014-05-04`). Overlaps eval filtering / relationship_property_lookup. | Delete. |
| `test_tier_zero_disabled_preserves_semantic_correctness` **and** `test_tier_zero_disabled_preserves_multi_speaker_kinship` | Second suite repeats Dylan IDs already in `test_speaker_relative_kinship_converges_on_dylan`. First suite repeats alias + in-law + count. Unique value is `DisabledParser` + `timings.tier == 1`. | Keep **one** test: disable Tier 0, assert parser is not called, assert a small ID set (self, named alias, one multi-hop, one count). |
| `test_agent_service_reported_queries_use_semantic_planner` | Name claims planner; `"我是谁"` is Tier 0 (`llm_call_count` would be 0). Address asked three ways; only strings. Overlaps identity + residence tests. | Delete. If agent wiring must stay tested, one question that is **not** a Tier-0 utterance is enough. |
| `test_filter_and_relationship_capabilities_are_semantic_and_allowlisted` vs `test_capabilities_are_semantic_and_new_fields_require_no_handler` | Overlapping `capability_payload()` snapshots. | Keep one capabilities test; drop the duplicate keys. |
| `_fixture_plan` paraphrase clusters | `我儿子几岁了`/`我儿子几岁`; `谁最年长`×3; 匡德伦 birthday×3; address×3; `我老婆的爸爸`/`父亲`. | Goes away with the 41-case parametrize. Do not rebuild them. |
| `test_disabled_mode_benchmark_does_not_require_tier_zero` | Entire `SemanticFactService` is faked; asserts the benchmark **harness** mode flag, not the pipeline. | Keep only if the CLI contract matters; it does not protect fact behavior. |
| `test_minor_status_is_distinct_from_a_persons_child_relation` | On this household both counts are `2`, so the test cannot fail if the two plans were swapped. It only asserts IR shape. | Keep the IR-shape assertions; drop the coincidental `== 2` pair or use a fixture where they diverge. |

Quality issues to fix in tests you **keep** (not extra cases):

- Prefer `result.evidence.entity_ids` over `expected in answer.text`.
- Do not put Tier-0 exact utterances in tests whose names say “planner” /
  “Tier 1”. Current offenders: the 41-case parametrize, `你是谁` in
  `test_assistant_identity_never_queries_household_graph` (acceptable if
  renamed as a Tier-0 test), and `我是谁` in
  `test_agent_service_reported_queries_use_semantic_planner`.
- Stop using production `data/` as the default semantic-fact fixture.
  `tests/static_test_data` already exists (Alex / Blair / Casey) and is unused
  here. Household-specific values (`匡德伦`, `2014-05-04`, `9岁`, `58天`,
  `12745 Droxford St`, five members / three adults) make generic IR tests
  brittle.

`_fixture_plan` / `FixtureSemanticInterpreter` should shrink to a handful of
oracle plans for executor tests, or be replaced by the eval YAML plans. It is
currently a second natural-language parser living in the test suite.

---

## Production APIs retained only because tests depend on them

| Symbol | File | Why it is still here | Test dependents |
|---|---|---|---|
| `TierZeroSemanticParser.canonical_utterances` | `semantic_facts.py` | Production `parse()` already lists the six strings. No runtime caller. | `test_semantic_planner_evaluation_is_large_and_adversarial` |
| `SemanticFactService(..., parser=)` | `semantic_facts.py` | Production (`AgentService`, benchmarks) never passes `parser`; they use the default + `tier_zero_enabled`. | `test_tier_zero_disabled_preserves_semantic_correctness` (`DisabledParser`) |
| `SemanticReference.kind == "entity_id"` | IR + `EntityResolver` | Planner output schema **strips** `entity_id`. Resolver still executes it. Kept so tests can spoof and so the planner can reject `MODEL_ORIGINATED_ENTITY_ID`. | `test_semantic_validator_rejects_context_reference_type_spoofing`, `test_semantic_planner_rejects_model_originated_entity_id` |
| `_JsonGraphDispatcher.calls` | `fact_benchmark.py` | CLI adapter plus test spy. Production answering uses `ToolDispatcher`. | query-count / no-graph-on-assistant / no-graph-on-unknown-property tests |
| `SemanticSchemaRegistry._property_candidates` physical pass-through `(semantic,)` | `semantic_facts.py` | Lets tests advertise `favorite_color` / `fixture_score` as the storage name. Production also uses this for unaliased catalog fields; the **tests** are what keep the “add a field, no handler” story alive. | `test_tier_one_can_select_a_new_schema_field_without_a_fact_handler`, `test_new_numeric_field_immediately_supports_generic_argmax`, `test_capabilities_are_semantic_and_new_fields_require_no_handler` |

Not test-only (do not delete for this audit): `tier_zero_enabled` (wired from
config), `validation_code` (planner), `validates` (engine + `try_answer`),
`capability_payload` (planner prompt).

Already gone, so not retained by tests: `RuntimeSchemaCatalog.entity_aliases`,
`_latest_user_text`. `HouseholdFactEngine(resolver=)` is unused even by tests.

---

## Recommended set (22 cases, not hundreds)

Keep the existing **good** ID / status tests (speaker-relative kinship, alias
convergence, scoped appellation, missing property vs computation, empty child
count, role-then-age fallback, marriage evidence, planner retry/unknown
property/entity_id, Tier-0 independence, eval oracle). Add or reshape only:

**P0 (8)**

1. Dispatcher `ok=False` → `computation_impossible`, no leak.
2. Planner `INVALID_PLAN` (argmin on `display_name`), `attempt_count == 1`.
3. Two malformed payloads → `MALFORMED_OUTPUT`, no graph access.
4. `requires_fact=false` → `try_answer is None`, `NOT_A_FACT`.
5. `self → child` with `caller_entity_id=None` → `caller_context_missing`.
6. Adult **and** minor `filter_input_missing` renderer (lines 1929–1931).
7. Duplicate-name and two-son ambiguity assert candidate IDs (reshape, not new).
8. Engine + in-memory `RetrievalService` on `static_test_data`: one alias, one hop, IDs.

**P1 (12, pick freely inside this list)**

9. Missing `household_id`.
10. Planner `RuntimeError` → `semantic_plan_unsupported`, no exception text.
11. Equal-age compare (`equal is True` + IDs).
12. `date_difference` on relationship `start_date`.
13. Duration exact days at frozen `current_time`.
14. Generic property `filter_input_missing`.
15. Conflicting `household_role` values.
16. Birthday today (`mode=days` → 0) through renderer.
17. One English found + one English missing-property path.
18. Empty household `count == 0`.
19. Older-sibling path with speaker `birth_date` missing.
20. `validation_code == "INVALID_PLAN"` if not folded into item 2.

That is **20** explicit new/reshaped cases plus the two ambiguity
strengthenings in item 7. Do not add a 41st household-string parametrize.
