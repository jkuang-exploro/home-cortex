# Architectural convergence for coding-agent efficiency

Baseline: `379f1e7` (Grok cleanup). This pass targets development context, not runtime tokens. The working tree was clean at the start; preceding changes are not credited here.

## Canonical architecture

```text
AgentService: utterance + trusted AgentRequestContext
  → SemanticFactService
    → SemanticFactPlanner → expand concepts / validate → SemanticFactRequest
    → HouseholdFactEngine → EntityResolver → RetrievalService → SurrealDB
    ← FactResult
    → FactRenderer → FactAnswer
```

The engine has no dependency on interpretation or rendering. The service owns sequencing, diagnostics, and failure presentation. Structured callers can execute the engine without importing the service.

| Concept | Canonical owner | Responsibility / input → output |
|---|---|---|
| Semantic request | `semantic_ir.SemanticFactRequest` | Validated semantic operations, references, ownership, conditions |
| Speaker context | `semantic_ir.AgentRequestContext` | Created by `AgentService`; conversation state supplies scoped focus |
| Interpretation | `semantic_planner.SemanticFactPlanner` | User language + semantic capabilities → `SemanticPlan` |
| Schema binding | `semantic_schema.SemanticSchemaRegistry` | Ontology concepts + deployed catalog → expanded/validated request and physical bindings |
| Entity resolution | `entity_resolver.EntityResolver` | Reference + context → successful `ResolvedEntities` or canonical failure `FactResult` |
| Fact execution | `household_fact_engine.HouseholdFactEngine` | Request + grounded entities → computed, evidence-checked `FactResult` |
| Persistence | `retrieval.RetrievalService` / SurrealDB | Bound graph operations → stored records; database owns factual values |
| Fact result | `semantic_ir.FactResult` | Values, rows, evidence, ambiguity, missing requirements |
| Rendering | `fact_renderer.FactRenderer` | Executed request + result + ontology → text |

## Findings verified against Grok's resulting code

The latest Grok commit already split the monolith into six focused implementation modules. A separate P0/P1 review-queue document was not present in the tracked repository; the commit and live call sites were audited directly. The earlier runtime-token report is not evidence for development-context savings.

There was already one factual request model and one deterministic execution path. `FactRequest`, `QueryRequest`, and a live Tier-0 fact router were not found. They were not invented or renamed to claim consolidation. Names/aliases and speaker references already converge through `EntityResolver`; exact-ID authentication, display-name rendering, and greeting selection serve distinct purposes.

The actual remaining duplication was a re-export facade, failure-status translations, and inverted dependencies between semantic types, persistence, execution, and orchestration.

## Architecture removed

* Removed the broad `semantic_facts.py` re-export implementation. Production callers, tests, and maintained probe scripts now import symbols directly from their owners. The file now contains the existing `SemanticFactService` and its diagnostics, moved out of the deterministic engine. No compatibility aliases were added.
* Removed `ResolutionStatus`, its ten-value vocabulary, and both conversion dictionaries. Resolution failures previously went from fact status → resolver status → fact status → reconstructed result. They now produce a `FactResult` that execution forwards unchanged, preserving evidence and candidates.
* Retired the mixed success/failure `ResolutionResult`. `ResolvedEntities`, colocated with its producer, contains only successful grounding data: entities, IDs, traversal records, evidence, and containment groups. Three failure-related fields and the duplicate failure contract are gone. The count of dataclass models is unchanged; this is a smaller contract, not a claim that grounding and computation are the same thing.
* Removed the engine's imports of planner, renderer, and answer orchestration. Its file shrank from 1,062 to 743 lines.
* Moved the existing `WriteMode` type authority to `mutation_ir.py`. The storage writer consumes that declaration. Semantic models no longer import the writer, ingestion, DB client, or record-ID machinery merely to describe preview/commit mode.

Canonical factual execution paths: **1 → 1**. Canonical factual request models: **1 → 1**. Failure-status vocabularies: **2 → 1**. Failure mapping dictionaries: **2 → 0**. Re-export facade layers: **1 → 0**. Production files removed: **0**; the existing facade file was repurposed, with no new production file.

## Repository-context reduction

These are conservative, reproducible **static import-reachability estimates**, including deferred/type-only imports and full source-file LOC. They estimate the source surface a dependency-following agent could encounter, not the minimum snippets a human needs or measured LLM token consumption. The same semantic entry modules are used before and after; all reached filenames are recorded in `context-summary.json`.

| Development task | Files before | Files after | Relevant LOC before | Relevant LOC after |
|---|---:|---:|---:|---:|
| Understand complete factual-query architecture | 22 | 18 | 8,826 | 7,385 |
| Change entity resolution | 16 | 12 | 6,169 | 4,815 |
| Change speaker behavior | 23 | 19 | 9,006 | 7,565 |
| Change containment semantics | 16 | 12 | 6,169 | 4,815 |
| Add a factual property | 15 | 11 | 5,560 | 4,206 |

LOC surface falls approximately **16–24%** across these tasks. These improvements are relative to the already-split Grok baseline, not the older monolith. File count in the package remains **45 → 45**. Production Python LOC: **17,228 → 17,113**, with **366 additions / 481 deletions, net −115**. Counts include package benchmark modules and exclude tests, scripts, YAML, and reports. Some import-formatting changes affect LOC; the import-graph reductions are the stronger architectural evidence.

Reproduce using an isolated archive of the baseline:

```sh
python scripts/context_surface_audit.py \
  --before /tmp/hc-agent-context-baseline/src/home_cortex \
  --output /tmp/context-summary.json
```

## Validation

* Baseline full suite: **704 passed**.
* Focused ownership, facts, composition, containment, transport, and evaluation suite: **198 passed**.
* Final full suite: **713 passed in 12.49 seconds**.
* New architecture tests verify that semantic-model imports do not load persistence, engine imports do not load its callers, and seven resolver failure categories preserve the original `FactResult` and evidence.
* Interpreter capabilities, output schema, semantic-plan schema, instructions, and examples have identical baseline/candidate SHA-256 fingerprints (`boundary-summary.json`). No prompt or model-behavior optimization was attempted.
* `git diff --check` passes.

The composition manifest's **test-file fingerprint only** was refreshed because imports in `test_composition_eval.py` changed. Benchmark wording, fixture data, gold plans, scoring revision, and all other recorded file fingerprints are unchanged. This is an evaluation bookkeeping update, not an accuracy correction. No production inference or accuracy claim is made.

## Remaining intentional complexity

`SemanticPlan` selects fact/non-fact/mutation intent; it is not another factual request. Concept expansion supplies complete ontology paths and filters before strict validation, so that transition adds semantics. `ResolvedEntities` retains intermediate edges required by relationship-property computation. `FactAnswer` adds text and diagnostics to the executed request/result. Merging these would hide distinctions a maintainer needs.

`SemanticConversationService` owns bounded, isolated discourse state; `SemanticFactService` owns one-turn interpretation/execution/rendering. Their multiple real consumers include the API and benchmark harness. Stateless replay and persistent focus share the same factual implementation. The graph dispatch adapter enforces operation boundaries and supports both SurrealDB and synthetic execution; it is not an alternate fact engine.

Ontology YAML owns reusable domain meaning; edge YAML owns graph direction and endpoints; runtime catalog discovers available fields; semantic schema binds them. The bounded IR and generic operator registry intentionally describe different levels of computation.

## Future-agent guidance

The short authoritative guide is in `AGENTS.md`.

* New relationship behavior: start with ontology/edge YAML; inspect `semantic_schema.py` only for a new binding rule and `entity_resolver.py` for a new traversal rule.
* Speaker-relative bug: `entity_resolver.py`, `semantic_ir.AgentRequestContext`; inspect `semantic_conversation.py` only when discourse scope is involved.
* Nested containment: `entity_resolver.py` and `tests/test_collapsed_containment.py`.
* Existing-kind factual property: ontology YAML and deployed schema/data; `semantic_schema.py` owns discovery/binding, with no new question handler.
* New deterministic computation: `operator_registry.py` and `household_fact_engine.py`.
* Answer wording: `fact_renderer.py` / `semantic_display.py`; no resolver or storage changes.
