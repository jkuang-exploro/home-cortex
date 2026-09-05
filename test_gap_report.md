# Test Coverage and Behavioral Gap Audit

Date: 2026-09-04

Scope: tests that protect household-fact architecture (`semantic_facts.py`,
`EntityResolver`, `HouseholdFactEngine`, `SemanticFactPlanner`,
`SemanticFactService`, `AgentService` fact routing, `RetrievalService` alias
path). Adjacent suites (greetings, calendar, ingest, calculate) were inspected
only when they touch the listed invariants.

Method: full inventory of `tests/` against production branches. No coverage
package is installed (`pytest` only; `pyproject.toml` has no `coverage` /
`pytest-cov`). This audit did not add one. Branch claims below are from
reading production control flow against test assertions, not measured
line-coverage percentages.

The current suite is large in `tests/test_semantic_facts.py` (~50 test
functions, one parametrize of ~40 utterances). Size is not the problem.
Assertions, oracles, and fixture choice are.

---

## Invariant map

| Invariant | Verdict | What actually protects it | Gap |
|---|---|---|---|
| Speaker-relative self | Partially covered; duplicative on the default speaker | `我是谁` via Tier 0 for `person:jian_kuang`; agent-service identity tests; planner-only oracle runs the self plan | Other speakers are not asserted for **self** (only for kinship). Most asserts are the string `匡健`. |
| Direct alias resolution | Well covered at two layers | Semantic: canonical IDs for 匡德伦/德伦/Dylan. Retrieval: exact alias, ambiguity, scoped appellation, embedded Surreal | Semantic path uses `_JsonGraphDispatcher`, not `RetrievalService`. |
| Relational entity resolution | Well covered | Named-entity + path (`巴璞的儿子`, `巴志刚的女儿`) with canonical IDs | Same JSON-dispatcher isolation as alias. |
| Multi-hop kinship | Well covered; duplicative | In-law, 孙子/外孙/哥哥; IDs; ontology `father_in_law` composition | Two speaker-kinship tests do the same five cases. Ended edges and missing-anchor sibling are untested. |
| Semantic property lookup | Partially covered | `birth_date`, `full_address`, open-world `favorite_color`; storage alias `dob`→`birthday` | `given_name` / `display_name` execute in the oracle eval but are not locked to IDs or values. English locale almost unused. |
| Relationship property lookup | Well covered | Spouse `start_date`, duration, speaker-relative extra couple, missing start | Duration asserts `> 4000` rather than a clock-relative exact value. |
| Tier-0 independence | Partially covered; some tests force Tier 0 | Planner-only eval (111 cases, Tier 0 off); `test_tier_zero_disabled_preserves_semantic_correctness`; config flag | Exact utterances `我是谁` / `你是谁` / `家里都有谁` / `家里有几个人` hit Tier 0 in the big parametrize and in `AgentService` tests. |
| Planner fallback | Partially covered | Structural retry; unknown property without retry; `requires_fact: false` → conversation at agent layer | `MALFORMED_OUTPUT`, `NOT_A_FACT`, `INVALID_PLAN`, planner exception, and “invalid plan must not fall through to chat” are missing or only half-tested. |
| Filter / count | Well covered | Adult/minor/gender counts, empty child count = 0, role-then-age fallback, date-range on relation | `filter_unsupported` never executed. Counts in the big parametrize assert Chinese copy (`五个人`) not IDs. |
| Argmin / argmax | Well covered | Household extrema IDs, pairwise age, partial evidence, generic new numeric field | Equal-age branch untested. Big parametrize re-tests extrema as `巴志刚` / `匡悠然` strings. |
| Date calculations | Well covered at operator + engine | `completed_years`, `annual_occurrence` (incl. leap day), `duration`, invalid date → `computation_impossible` | `date_difference` / `unit_conversion` / `latest` / `earliest` are registered, not executed through the engine. Birthday-today (`days == 0`) untested. |
| Missing evidence | Well covered | Distinct statuses: `property_unavailable`, `computation_input_missing`, `relation_property_unavailable`, `entity_not_found`, `relationship_not_found` | Several still also pin Chinese renderer copy. `caller_context_missing` is only covered via Tier-0 `AgentService`, not the engine. |
| Ambiguity | Partially covered | Duplicate name, two sons, two grandsons | Duplicate-name and two-son tests assert rendered names, not candidate IDs (grandson test does assert IDs). |
| Malformed planner output | Partially covered | Incomplete request retries once; invented operation → `UNSUPPORTED_OPERATION`; model `entity_id` → `MODEL_ORIGINATED_ENTITY_ID`; unknown property/relation | No assertion of `MALFORMED_OUTPUT`. No non-object / non-JSON payload. No `INVALID_PLAN` through the planner. |
| Database failure | Not covered for facts | `/health` → 503; `ToolDispatcher` swallows `RuntimeError` | Fact engine/service never see a failed graph call. Uncaught dispatcher exceptions would escape `try_answer`. |

---

## P0 missing tests

Add these. They protect architectural fail-closed behavior that is currently
unasserted. Do not grow the 40-row string parametrize to cover them.

### 1. Graph / database failure through the fact pipeline

`ToolDispatcher` maps unexpected errors to `{ok: false, code: tool_execution_failed}`.
`_FactExecution.records` turns `ok is not True` into `computation_impossible`.
`HouseholdFactEngine.execute` / `SemanticFactService.try_answer` do **not**
catch a raised dispatcher exception.

Missing:

- dispatcher returns `{ok: false}` during `get_entity` / `get_relationships` /
  `resolve_entity_alias` → `computation_impossible`, no secret in `answer.text`
- dispatcher **raises** → request does not 500 unhandled (today it would)
- `AgentService.answer` still returns a fact failure, does not call
  `chat_with_tools`

`/health` 503 does not cover this invariant.

### 2. Malformed planner output is `MALFORMED_OUTPUT`, then stops

`_structural_validation_code` returns `MALFORMED_OUTPUT` unless the error is
an operation-field `ValidationError`. Existing tests retry a missing `subject`
until success, or classify `UNSUPPORTED_OPERATION`. Nothing asserts the
malformed code.

Missing:

- non-object payload (string / `None` / `[]`) twice → `MALFORMED_OUTPUT`,
  `attempt_count == 2`
- `try_answer` returns `semantic_plan_unsupported`, `dispatcher.calls == []`
- `AgentService` does **not** invoke the conversation model (invalid plans
  must not fall through to chat)

### 3. `NOT_A_FACT` is conversation fallback, not a fake household answer

When `requires_fact is false`, `try_answer` returns `None` and
`AgentService` runs the model loop. Agent tests cover this only with a
stub that always returns `requires_fact: false` for every utterance,
including capability questions.

Missing a fact-service-level test:

- planner returns `{requires_fact: false, request: null}`
- `try_answer(...) is None`
- zero graph calls
- `AgentService` then calls `chat_with_tools` once

This is the only legitimate planner → conversation fallback.

### 4. `INVALID_PLAN` is classified without graph access

`validation_code` returns `INVALID_PLAN` when properties/relations are known
but `validates()` is false (e.g. `argmin` on `display_name`, `count` on bare
`self`). Tests check `validates() is False` only.

Missing: planner emits that plan → `SemanticPlannerFailure.validation_result == "INVALID_PLAN"`, `attempt_count == 1`, no dispatcher calls. Semantic
validation must not retry and must not fall back to Tier 0 (Tier 0 already
ran first; lock that by using a non-canonical utterance).

### 5. Speaker-relative **self** for two speakers, by canonical ID

Kinship is speaker-relative (`test_speaker_relative_kinship_converges_on_dylan`).
Self identity is not: `我是谁` is always `person:jian_kuang` / `"匡健"`, and
agent-service tests force Tier 0 (`semantic_plan_calls == 0`).

Missing, both with Tier 0 on and `tier_zero_enabled=False`:

- speaker `person:alex_example` → evidence `("person:alex_example",)`
- speaker `person:blair_example` → evidence `("person:blair_example",)`
- missing caller → `caller_context_missing` through the engine, not only
  through `AgentService` + Tier 0

Prefer `tests/static_test_data` (Alex/Blair) over `data/` (匡健 household).

### 6. Semantic resolver on the real retrieval path

Every semantic-fact test uses `_JsonGraphDispatcher` over `data/`. Alias
normalization, appellations, and inverse edges are reimplemented there.
`RetrievalService` is well tested in isolation (`test_retrieval.py` already
uses generic fixtures + MemoryDatabase).

Missing one integration test:

- `HouseholdFactEngine(ToolDispatcher(RetrievalService(MemoryDatabase)))`
- ingest `tests/static_test_data`
- named alias `艾力克斯` and one relational hop (e.g. Casey’s parent)
- assert canonical IDs

Without this, CI can pass while Surreal alias SQL and the JSON adapter
diverge.

### 7. Ended relationships are invisible to facts

Retrieval tests `include_ended=False`. The engine always passes
`include_ended: False`. There is no semantic-fact test that a spouse (or
parent) edge with `end` set is `relationship_not_found` rather than a live
answer.

### 8. Planner transport failure

`try_answer` has `except Exception` around `planner.plan` and returns
`_failure_answer`. Untested. An Ollama timeout/crash must become
`semantic_plan_unsupported`, must not query the graph, and must not run
conversation as if the question were chit-chat.

---

## P1 useful tests

High value, not blockers. Prefer these over more utterance paraphrases.

1. **`given_name` / `display_name` select** on a resolved person: assert
   `entity_ids` and the stored value. The 111-case oracle runs these plans
   but only checks `final_answer` truthy.
2. **Ambiguity candidate IDs** for duplicate exact name and two matching
   children. Grandson already does this; the other two assert Chinese copy.
3. **English locale** (`locale="en"`) for self, missing property, count, and
   `semantic_plan_unsupported`. Renderer `_en` is almost unexecuted;
   English utterances in the big parametrize still render with `locale="zh"`.
4. **`filter_unsupported`**: unknown predicate that somehow reaches the
   engine, or a relation filter on a non-relation property. Renderer has a
   branch; engine can raise it; no test.
5. **Equal-age `argmin`/`argmax`** with `other`: `result.value["equal"] is True`.
6. **`annual_occurrence` today** (`mode=days` → `0`) and leap-day through the
   engine (operator registry already covers leap day in isolation).
7. **`date_difference`** on a relationship `start_date` (distinct from
   `duration`). Registry-listed, unused by engine tests.
8. **Missing `household_id`** on `current_household` → `entity_not_found`,
   not a fabricated roster.
9. **Older-brother with missing speaker `birth_date`**: `value_from=anchor`
   → `property_unavailable` / invalid reference, not a guessed sibling.
10. **Clock-relative duration**: freeze `current_time`, assert exact day
    count instead of `> 4000`.
11. **Gender filter collection** through `try_answer` asserting member IDs,
    not `两位` copy.
12. **Generic-fixture kinship** on Alex/Blair/Casey so tests do not encode
    Fort Cerritos names. Retrieval already has this graph.

Do **not** add: more Chinese paraphrases, more `favorite_color` open-world
fields, more renderer copy variants, real-LLM planner tests in CI (the
oracle eval is the right split).

---

## Recommended cases (22)

Keep the existing ID-based kinship/alias tests. Add or replace with the
following; delete the redundant string rows listed in the next section.

| # | Pri | Case | Assert |
|---|---|---|---|
| 1 | P0 | Graph call returns `ok: false` | `computation_impossible`; no secret in text |
| 2 | P0 | Graph call raises | no unhandled exception; same safe status |
| 3 | P0 | Cases 1–2 through `AgentService` | `chat_with_tools` not called |
| 4 | P0 | Planner returns a string twice | `MALFORMED_OUTPUT`, attempts=2, no graph |
| 5 | P0 | Case 4 through `AgentService` | no conversation loop |
| 6 | P0 | `requires_fact: false` | `try_answer is None`; then exactly one chat call |
| 7 | P0 | `argmin` + `display_name` from planner | `INVALID_PLAN`, attempts=1, no graph |
| 8 | P0 | Self identity, speaker A and speaker B, Tier 0 on | evidence IDs match each speaker |
| 9 | P0 | Same as 8, `tier_zero_enabled=False` | `timings.tier == 1`; same IDs |
| 10 | P0 | Self with no `caller_entity_id` through engine | `caller_context_missing` |
| 11 | P0 | Engine + `RetrievalService` + MemoryDB alias | `person:alex_example` |
| 12 | P0 | Engine + MemoryDB parent hop | Casey → parent IDs |
| 13 | P0 | Ended `spouse_of` | `relationship_not_found` |
| 14 | P0 | Planner raises `RuntimeError` | `semantic_plan_unsupported`; no graph; no chat |
| 15 | P1 | `select given_name` on named entity | ID + value |
| 16 | P1 | Duplicate name / two sons | `ambiguous` + candidate IDs |
| 17 | P1 | `locale="en"` missing property and count | English renderer, same status/IDs |
| 18 | P1 | Equal-age pairwise extrema | `equal is True` |
| 19 | P1 | Birthday today | `annual_occurrence` value `0` |
| 20 | P1 | `date_difference` on marriage start | numeric value vs frozen clock |
| 21 | P1 | Missing household id | `entity_not_found` for member list/count |
| 22 | P1 | Brother hop, speaker dob missing | not_found / property_unavailable, not a name |

---

## Redundant tests that can be deleted or consolidated

Do not delete the ID-based tests. Delete or shrink the string-oracle
duplicates.

### Delete or shrink

| Test | Why | Keep instead |
|---|---|---|
| `test_canonical_and_planner_composed_facts_execute_deterministically` (~40 rows) | Almost every row asserts a Fort Cerritos **rendered substring**. Four exact rows **force Tier 0**. Duplicates specialized tests and the 111-case oracle. | A handful of ID/status cases, or rely on the oracle + targeted tests. |
| `test_open_world_plans_execute_with_tier_zero_disabled` | Same extrema/count/marriage answers as the parametrize, still strings. | Oracle eval + engine extrema/count tests. |
| `test_tier_zero_disabled_preserves_multi_speaker_kinship` | Duplicate of `test_speaker_relative_kinship_converges_on_dylan` (same five speakers/questions, same ID). | Keep the original. |
| `test_agent_service_reported_queries_use_semantic_planner` | Name is false: `我是谁` hits Tier 0. Address rows re-assert `12745 Droxford St`. | One AgentService test that a **non-canonical** fact uses the planner (`llm`/plan call count). |
| `_fixture_plan` phrase table (~200 lines) | Third copy of `benchmarks/semantic_planner_eval.yaml` plus extra household questions. Tests the test oracle, not production language parsing. | YAML oracle (`test_planner_only_oracle_executes_full_eval_and_checks_tier0_parity`). |
| Duplicate parametrize rows | `我儿子几岁了`/`我儿子几岁`; `我老婆的爸爸`/`父亲`; three 匡德伦 birthday strings; three address strings; three 最年长 strings | One per IR plan. |
| `test_birthday_intents_have_distinct_generic_plans` + `test_in_law_is_composed_from_existing_generic_relations` + `test_marriage_date_is_a_planner_owned_relationship_property` | Assert shapes of `_fixture_plan`, not production planner/ontology | Ontology tests already own in-law composition; eval YAML owns marriage IR. |
| `fact_benchmark.QUESTIONS` / `SPEAKER_CASES` overlap with eval YAML | Second utterance list. Benchmark tests patch Ollama and re-run household questions. | Point the CLI at the eval YAML; keep benchmark tests as report-shape tests with a tiny fake service (the disabled-mode test already does this). |

### Consolidate, do not delete both

| Pair | Action |
|---|---|
| `test_all_static_and_relational_references_converge_on_canonical_person` vs `test_tier_one_open_world_paraphrases_use_resolver_not_entity_ids` vs dylan rows in `test_tier_zero_disabled_preserves_semantic_correctness` | Keep **one** alias+relational ID matrix. Keep the Tier-0-disabled test only for the six canonical utterances + one multi-hop. |
| `test_argmin_and_argmax_share_the_household_extrema_path` vs canonical 最年长/最年幼 rows vs `test_tier_one_uses_one_semantic_call_then_deterministic_execution` | Keep the ID-based extrema test. Keep the one-call Tier-1 test. Drop string extrema rows. |
| `test_capabilities_are_semantic_and_new_fields_require_no_handler` vs `test_filter_and_relationship_capabilities_are_semantic_and_allowlisted` | Merge into one capability-payload contract test. |
| `test_operator_registry.test_argmin_is_generic_over_new_type_compatible_fields` vs `test_new_numeric_field_immediately_supports_generic_argmax` | Complementary (registry vs engine). Keep both; they are not duplicates. |
| Retrieval scoped appellation vs semantic `test_scoped_appellation_is_grounded_by_resolver_context` | Complementary layers. Keep both. After P0 #11 they should share `static_test_data`. |

### Tests that mock away the pipeline (fix assertions, or fold into P0)

- Default `SemanticFactPlanner(FixtureSemanticInterpreter)`: production
  language parsing is never under test in unit tests. That is acceptable
  **if** CI keeps the YAML oracle and does not grow `_fixture_plan`.
- `test_disabled_mode_benchmark_does_not_require_tier_zero` uses a fake
  `try_answer` that always returns Dylan. It tests the benchmark harness,
  not facts. Keep as a harness test; do not count it as Tier-0 independence.
- `AgentService` identity tests (`我是谁？`, `你是谁？`) intentionally force
  Tier 0. Keep one each; do not treat them as planner coverage.

### Household-specific values to stop spreading

`tests/test_semantic_facts.py` loads production `data/` (`person:jian_kuang`,
`address:fort_cerritos`, `12745 Droxford St`, 匡健/巴璞/匡德伦). Retrieval and
ingest already use generic `tests/static_test_data` (Alex/Blair/Casey). New
tests should use the generic graph unless the case is specifically about
deployed ontology field aliases present only in `data/`.

---

## Production APIs retained only because tests depend on them

Live product surfaces (`try_answer`, `execute`, `resolve_entity_alias`,
`dispatch_internal`, `AgentService.answer` used by `api.py`) are out of
scope here.

| Symbol | Callers | Notes |
|---|---|---|
| `TierZeroSemanticParser.canonical_utterances` | `tests/test_semantic_planner_benchmark.py` only | Production `parse()` already lists the six strings. Inline the tuple in the test, or read it from the eval YAML. Safe to delete from production. |
| `SemanticFactService(parser=...)` | `test_tier_zero_disabled_preserves_semantic_correctness` injects `DisabledParser` | Production never passes `parser`. `tier_zero_enabled=False` already bypasses parse. The injection exists to fail if parse is called; keep that as a test double **or** assert `timings.tier == 1` without a custom parser. |
| `fact_benchmark.QUESTIONS` and `SPEAKER_CASES` | CLI defaults; overlap eval YAML | Not required by tests once the CLI reads the eval file. Tests that need utterances should import the YAML. |
| `_JsonGraphDispatcher` as the semantic-fact test backend | tests + offline CLI | Not test-only, but tests are the reason the adapter must mimic `resolve_entity_alias`. After P0 integration with `RetrievalService`, tests should not be the justification for a second alias implementation. CLI may keep a thin JSON backend. |

Already gone (cleanup doc is stale; do not keep tests for them):
`RuntimeSchemaCatalog.entity_aliases`, `RetrievalService.retrieve`,
`search_entities`, `POST /v1/retrieve` (`test_legacy_retrieve_route_is_removed`
is the right lock).

Not test-only (do not delete for this reason): `AgentService.answer` (API
uses it), `validation_code` / `validates` (planner + engine),
`dispatch_internal` (engine production path).

---

## Weak production branches (no new coverage framework)

Unexecuted or only renderer-covered, besides the P0/P1 list:

- `FactStatus.operator_unsupported` is rendered but never raised by the engine
- `EntityResolver` unknown `kind` → `computation_impossible`
- `kind == "entity_id"` execution (planner is forbidden from emitting it;
  engine still accepts a hand-built request)
- `_filter_entities` `needs_load` when the traversal payload lacks the
  filter field (JSON entities are full records, so this rarely runs)
- `load_if_unnamed` when `display_name`/`name`/`full_name` are absent
- `_anchor_property` with `len(anchors) != 1`
- English renderer success paths (`_en` `found` branches)
- `try_answer` `except Exception` around the planner (P0 #14)

Do not add tests for `operator_unsupported` until something produces it.
)
