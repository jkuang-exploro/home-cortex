# Semantic planner contract repair

Starting revision: `ffdaf4cfdfe68689a34d2e2ffb9de121654d86f7`. No deployment or live household data change.

**Status: code and deterministic validation are complete. Real-LLM acceptance on the
production GPU machine is the only remaining gate.** The deterministic CI suite is
green (see [Deterministic validation](#deterministic-validation)); the architectural
invariants are locked by `tests/test_semantic_contract.py` and the benchmark datasets.
Final real-LLM repeated measurements on the production GPU host will replace the
pending numbers below.

## Architecture and root causes

| Layer | Finding | Candidate correction |
|---|---|---|
| Model-facing grammar | Ownership was optional; capability prose used `source` rather than the actual request field `property_source`. Property and operand decisions were not structurally connected. | Require the property/ownership decision and distinguish entity and relationship request shapes. Relationship requests require a nonempty path and cannot carry a second person operand. |
| Ontology/reference serialization | Expanding a filtered relative manually lets the model drop parts of its definition or substitute a broader relative. | Explicit `concept` steps expand complete ontology paths into the existing IR. Additional filters are conjunctive; no utterance is read during expansion. Full definitions remain visible to the interpreter. |
| Reference contract | Literal names could occupy a contextual reference's unused value field; reference kinds and types were not connected in the output grammar. | Typed contextual references use null values and fixed entity types; literal names require `named_entity`. Runtime also rejects literal values on contextual references. Relation type signatures are advertised. |
| Operation/predicate vocabulary | Predicate names were free strings; `filter`/`traverse` were advertised as outer operations even though the request operation enum excludes them. | Separate predicate and field-filter productions, constrain predicate names to ontology declarations, and advertise only executable outer operations. Predicate definitions are meaning metadata, not extra emitted predicates. |
| Ordered properties | The model can equate larger age with a larger birth date and confuse an entity extremum with temporal ordering. | Ontology property metadata declares minimum/maximum meanings. Generic examples distinguish collection extrema and two-person comparisons. Both operands execute through the existing generic engine, including ties. |
| Routing/normalization | Six sentence-based shortcuts bypassed interpretation. Normalization could replace scope, move predicates, and reinterpret property filters. | Remove sentence dispatch and its switches, plus semantic repair normalization. All language goes through the interpreter. Trusted structured requests still execute directly through the engine. |

No new execution operation, question handler, answer cache, household identity mapping, live alias change, or downstream natural-language interpretation was added.

## Evaluation-only changes

1. The single `male_children` list case accepts its final-step gender filter as an alternative. On a complete-gender fixture both plans return the same final entity set, including multiple sons. This does **not** establish general filter-movement equivalence. Deterministic counterexamples cover intermediate relatives, scalar ambiguity, missing gender data, and relationship properties with multiple edges. Production normalization and global plan comparison do not relocate filters.
2. `wife_father_given_name` already accepted identity/display-name plans but its answer contract only named the given name. It now accepts the authoritative stored names for the same expected person. Wrong people and unrelated names still fail. Existing daughter name alternatives remain intact.
3. The single marriage-start utterance that explicitly says “wife” accepts the
   ontology-defined wife path as well as the spouse path. Both select the same
   spouse edge property while the wife form preserves the user’s gender
   constraint. Other marriage cases still require the unfiltered spouse path.

Final metrics will show the original and adjusted evaluation contracts separately. Missing kinship filters remain plan errors.

## Deterministic validation

Run locally with `.venv/bin/python -m pytest -q` (359 tests) and the planner-only
oracle benchmark. The oracle supplies the expected semantic request for every
dataset utterance, then runs it through the real `SemanticFactPlanner ->
SemanticFactRequest -> HouseholdFactEngine` path, so this measures downstream IR
and executor correctness independent of model quality.

- Fixed suite (`benchmarks/semantic_planner_eval.yaml`): **119/119 plans (100%)**;
  every capability category is 100% (aggregation 23/23, entity_reference 7/7,
  filtering 13/13, multi_hop_kinship 7/7, property_selection 20/20,
  relationship_property_lookup 8/8, relationship_traversal 17/17,
  speaker_relative_reference 9/9, temporal_operation 15/15). No failure reasons.
- Tier-1 probe (20 questions, deterministic): **20/20 plans and 20/20 answers (100%)**.
- Synthetic probe (`benchmarks/semantic_planner_synthetic.yaml`): **13/13 plans and
  answers (100%)**, including two speakers, parent gender contrast, husband/wife
  contrast, a unique daughter versus an ambiguous two-son count, and an unknown name.

## Local validation

The full software suite is run locally with `.venv/bin/python -m pytest -q`. Synthetic households cover two speakers, changed people/dates, two sons plus a daughter, parent gender contrasts (father vs mother), relationship/entity property ownership, husband/wife contrast, older/younger with reversed operands and ties, adulthood at the eighteenth birthday, ambiguity, and strict unsupported-plan rejection. Dedicated tests cover ontology expansion without mutation or dropped filters.

Held-out interpretation resources are `benchmarks/semantic_planner_heldout.yaml` and `benchmarks/semantic_planner_synthetic.yaml`. Their wording is tested for separation from prompt examples and the fixed suite. The latter uses invented records under `benchmarks/fixtures/semantic-contract`.

## Production isolation and provenance

Host: `jkuang@192.168.68.59`, container `cortex-cortex-api-1`, GPU model `qwen3.5:9b`. All candidate code and staged data live under `/tmp/qwen35-semantic-contract`; live `/app/data` is not modified.

Baseline constants:

- Model digest: `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`.
- Staged data tree: `b13cf0dfb5102b07dbfbad4d9e8103d0257646cb7a33e797612643081907b36c` (exact baseline restored by retaining the original trailing newline in the staged alias overlay).
- Edge schema tree: `4144c93ae28d050851b302e676e1a29fa1095aab4b978eebaa25cd53b4b3a3eb`.
- Frozen time: `2026-09-03T12:00:00-07:00`.
- `think=false`, temperature 0, 384 output tokens, keep-alive 24h, JSON graph, no Tier 0.

Original baseline artifacts are untouched. `baseline-relevant-rows.json` contains only the relevant baseline rows. Complete per-case JSON reports and probe summaries preserve the investigation. Reports contain source/data/schema fingerprints; later reports also capture the actual capability payload, output schema, system prompt, examples, and command arguments. Temporary 8192-context/schema-grounding experiments are explicitly separate from baseline-setting measurements.

The one-off `candidate-*.tar` snapshots that staged each variant onto the GPU host are **no longer in the repository**. They were investigation scratch: stale copies of the working tree (several including `__pycache__/*.pyc`), non-diffable, and redundant with the source in `src/` and with the per-run inputs captured in the JSON reports. The provenance that matters is the readable `*-summary.json` files and this report. Any needed snapshot remains recoverable from git history; `artifacts/qwen35-semantic-contract/*.tar` is now gitignored.

The system-message hypothesis was not supported by the trials; the deployed renderer also preserves the first system message and subsequent messages ([Ollama v0.32.13 renderer](https://raw.githubusercontent.com/ollama/ollama/v0.32.13/model/renderers/qwen35.go)). Supplying schema text is recommended by [Ollama's structured-output documentation](https://docs.ollama.com/capabilities/structured-outputs), but the controlled schema-only trial did not improve this model and is not the selected production path.

## Before/after metrics

Real-LLM baseline on `qwen3.5:9b` (pre-repair): fixed suite **95/119 plans** (answers
unscored); probe **75/100 plans and answers**. Probe planner P50/P95 1922/2085 ms;
suite 1900/2114 ms.

Deterministic post-repair: fixed suite **119/119 plans**, probe **20/20 plans and
answers**, synthetic probe **13/13**. The post-repair real-LLM numbers are pending
the production GPU run described in [Remaining failures and acceptance](#remaining-failures-and-acceptance).

## Remaining failures and acceptance

Deterministic acceptance is met: the full fixed suite, the 20-question probe, and the
synthetic contrasting-concept probe all execute with 100% plan and answer correctness
through the real interpreter/executor path, and the architectural invariants are locked
by `tests/test_semantic_contract.py`.

Real-LLM acceptance is **not yet established**. The remaining step is to run the
deterministic suite, the 20-question probe, and the held-out/synthetic probes on the
production GPU host (`qwen3.5:9b`, temperature 0, JSON graph, no Tier 0) from an
isolated benchmark package, with the data/schema/package fingerprints recorded. A
successful single 20-question trial does not establish the required repeated
100/100 probe result or five-pass reliability. Passing the known questions is
necessary but insufficient: completion requires preserving the layer boundaries and
demonstrating compositional generalization on unseen phrasing, different speakers,
and synthetic households.
