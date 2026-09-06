# Semantic planner contract repair — validation in progress

Starting revision: `ffdaf4cfdfe68689a34d2e2ffb9de121654d86f7`. No deployment or live household data change.

Acceptance has not yet been established. Final repeated measurements will replace this status.

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

## Local validation

The full software suite is run locally with `.venv/bin/python -m pytest -q`. Synthetic households cover two speakers, changed people/dates, two sons plus a daughter, parent gender contrasts, relationship/entity property ownership, older/younger with reversed operands and ties, adulthood at the eighteenth birthday, ambiguity, and strict unsupported-plan rejection. Dedicated tests cover ontology expansion without mutation or dropped filters.

Held-out interpretation resources are `benchmarks/semantic_planner_heldout.yaml` and `benchmarks/semantic_planner_synthetic.yaml`. Their wording is tested for separation from prompt examples and the fixed suite. The latter uses invented records under `benchmarks/fixtures/semantic-contract`.

## Production isolation and provenance

Host: `jkuang@192.168.68.59`, container `cortex-cortex-api-1`, GPU model `qwen3.5:9b`. All candidate code and staged data live under `/tmp/qwen35-semantic-contract`; live `/app/data` is not modified.

Baseline constants:

- Model digest: `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`.
- Staged data tree: `b13cf0dfb5102b07dbfbad4d9e8103d0257646cb7a33e797612643081907b36c` (exact baseline restored by retaining the original trailing newline in the staged alias overlay).
- Edge schema tree: `4144c93ae28d050851b302e676e1a29fa1095aab4b978eebaa25cd53b4b3a3eb`.
- Frozen time: `2026-09-03T12:00:00-07:00`.
- `think=false`, temperature 0, 384 output tokens, keep-alive 24h, JSON graph, no Tier 0.

Original baseline artifacts are untouched. `baseline-relevant-rows.json` contains only the relevant baseline rows. Candidate tar snapshots and complete per-case JSON reports preserve the investigation. Reports contain source/data/schema fingerprints; later reports also capture the actual capability payload, output schema, system prompt, examples, and command arguments. Temporary 8192-context/schema-grounding experiments are explicitly separate from baseline-setting measurements.

The system-message hypothesis was not supported by the trials; the deployed renderer also preserves the first system message and subsequent messages ([Ollama v0.32.13 renderer](https://raw.githubusercontent.com/ollama/ollama/v0.32.13/model/renderers/qwen35.go)). Supplying schema text is recommended by [Ollama's structured-output documentation](https://docs.ollama.com/capabilities/structured-outputs), but the controlled schema-only trial did not improve this model and is not the selected production path.

## Before/after metrics

Pending final measurements. Baseline: fixed suite 95/119 plans (answers unscored); probe 75/100 plans and answers. Probe planner P50/P95 1922/2085 ms; suite 1900/2114 ms.

## Remaining failures and acceptance

Pending final measurements. A successful single 20-question trial does not establish the required 100/100 probe result or five-pass reliability.
