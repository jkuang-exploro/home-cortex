# Home Cortex architectural convergence and context-surface cleanup

## Outcome

The production package now contains 41 Python files and 14,416 raw Python lines,
down from 42 files and 14,805 lines. The natural size of the current production
core is therefore about 14.4k lines. This is **some remaining justified
complexity**: the package is cohesive at its current capability level, while the
opt-in V2 contract candidate remains a deliberate parallel schema constraint
until its evaluation is resolved.

The full deterministic suite passes. No real-model benchmark was run locally,
and this report makes no production accuracy claim.

## Canonical request path

The active path is:

```text
API / AgentService
  -> AgentRequestContext + DiscourseContext
  -> SemanticFactPlanner
  -> SemanticPlan / SemanticFactRequest
  -> EntityResolver
  -> HouseholdFactEngine
  -> RetrievalService / Database
  -> FactResult
  -> FactRenderer
```

`SemanticFactService` coordinates the planner, engine, renderer, and diagnostics
without translating the request or result into another DTO. The detailed owner,
type, input, output, schema, rendering, provider, and speaker-context map is in
`src/README.md` under **Semantic ownership and convergence**.

The audit confirmed these distinct authorities:

- `semantic_ontology.py` validates reusable semantic declarations loaded from
  ontology YAML.
- `edge_schema.py` validates physical relationship tables and directions.
- `schema_catalog.py` describes fields present in the deployed graph and owns
  shared name/appellation matching.
- `semantic_schema.py` binds ontology names to that catalog, validates plans,
  and derives planner capabilities.
- `operator_registry.py` owns the public operation type, operation/predicate
  definitions, validation, and deterministic implementations.
- `semantic_ir.py` owns canonical requests, context, plans, results, and evidence.
- `mutation_ir.py` remains separate because preview/apply and mutation constraints
  have no equivalent in factual reads.

## Changes

### Removed production-looking offline architecture

`semantic_transport.py` was never imported by the serving package. Its only
consumers were an offline profiling command and codec-specific tests. The codec
now lives at `scripts/profiling/semantic_transport.py` beside that profiler, so a
coding agent no longer encounters a second apparent serving wire format in the
runtime package. Ollama and OpenRouter continue to serve the canonical expanded
JSON schema.

### Converged operation ownership

The public `FactOperation` type moved beside the registry definitions. Derived
`FACT_OPERATORS` is now the source for schema capabilities, validation, and fact
execution, with an import-time invariant that its definitions exactly match the
typed operation vocabulary.

The following inactive registry entries were removed:

- `completed_years` and `duration`, structured-call compatibility aliases for
  `date_difference` that were already hidden from the planner;
- `traverse` and `filter`, names used only as benchmark trace labels rather than
  executable semantic operations;
- `sort`, `subtract`, and `divide`, unreferenced generic entries never emitted or
  executed by production;
- the unused `TRANSFORM_OPERATORS` export, output-kind metadata, binary-field
  input fields, and their validation branches.

Relationship tenure and age scenarios now use `date_difference` directly. The
benchmark trace can still report traversal/filter stages without registering
them as semantic requests.

### Preserved canonical context and boundaries

`AgentService` remains the only source of authenticated speaker identity.
Conversation focus extends the same `AgentRequestContext`; `EntityResolver`
alone interprets self, named entities, discourse, kinship, household scope, and
containment. Tool `ContextVar` state only carries trusted context through async
calendar/write handlers and does not independently derive identity.

`HouseholdFactEngine` remains the deterministic authority for fact execution.
Rendering modules retain separate roles: fact sentence rendering, semantic scope
description, generic tool-record display, greetings, and shared text cleanup.
Provider-specific request and streaming code remains in the two adapters because
their wire/error contracts differ; their planner prompt and schema are shared.

## Production LOC by batch

| Batch | Added | Removed | Net |
|---|---:|---:|---:|
| Relocate inactive compact transport to profiling | 0 | 266 | -266 |
| Operation/IR convergence and provider wording | 61 | 184 | -123 |
| **Total production** | **61** | **450** | **-389** |

README, tests, scripts, YAML, prompts, and generated artifacts are excluded from
this production total.

## Coverage and tests

| Metric | Before | After |
|---|---:|---:|
| Executable statements | 7,171 | 6,931 |
| Statement coverage | 87.62% | 88.37% |
| Branches | 2,958 | 2,846 |
| Branch coverage | 77.11% | 77.90% |
| Passing tests | 730 | 757 |
| Raw test Python LOC | 14,472 | 14,572 |

| Core module | Statement before | Statement after | Branch before | Branch after |
|---|---:|---:|---:|---:|
| `operator_registry.py` | 72.05% | 90.16% | 58.11% | 79.71% |
| `semantic_schema.py` | 89.58% | 89.62% | 83.84% | 83.94% |
| `semantic_ontology.py` | 83.76% | 83.76% | 65.54% | 65.54% |
| `entity_resolver.py` | 94.41% | 94.41% | 88.19% | 88.19% |
| `household_fact_engine.py` | 89.40% | 89.67% | 79.86% | 80.28% |
| `fact_renderer.py` | 77.11% | 77.11% | 74.06% | 74.06% |
| `semantic_ir.py` | 92.18% | 92.77% | 61.54% | 62.50% |

Focused operator tests now cover active shape validation, incomplete numeric
collections, predicate dispatch, unsupported symbols, unit conversion, and a
DST-ambiguous calendar offset. Remaining uncovered operator lines are primarily
defensive invalid-value branches; they were retained where they fail closed.

`db.py` remains low at 43.33% statement coverage because the uncovered code is
the thin live SurrealDB lifecycle/driver boundary (`connect`, `close`, `version`,
`query`, and `upsert`). Its behavior is active in API startup, health checks,
retrieval, ingestion, export, and writing. Most deterministic tests substitute a
memory or fake database, so duplicating the driver in unit tests would add little
architectural confidence.

The uncovered renderer paths are supported combinations of status, locale,
projection, and relationship wording. Ontology gaps are chiefly fail-closed YAML
validation branches. These remain explicit because deleting them would weaken
input validation; coverage was not inflated with implementation-mirroring tests.

## Context Surface Area

The maintenance audit measures conservative transitive package import
reachability, including deferred and type-only imports. Values are navigation
surface estimates rather than runtime module loading or literal agent token use.

| Task | Files Before | Files After | Relevant LOC Before | Relevant LOC After |
|---|---:|---:|---:|---:|
| Speaker-relative resolution | 32 | 32 | 12,733 | 12,610 |
| Add factual property | 13 | 13 | 5,558 | 5,435 |
| Nested containment | 12 | 12 | 4,815 | 4,694 |
| Relationship operation | 13 | 13 | 5,558 | 5,435 |
| DB fact retrieval | 16 | 16 | 6,090 | 5,967 |
| Planner interpretation | 14 | 14 | 5,465 | 5,344 |

The file counts stay level because the retained dependency boundaries are real.
Relevant transitive source falls by 121–123 lines for every representative task.
The wider production inventory drops by one file and 389 lines. Planner work also
has one fewer competing candidate implementation: the withdrawn compact codec is
now visibly engineering-only.

## Final sizing

| Inventory | Final |
|---|---:|
| Production Python LOC | 14,416 |
| Production Python files | 41 |
| Executable statements | 6,931 |
| Statement coverage | 88.37% |
| Branch coverage | 77.90% |
| Tests passing | 757 |
| Test Python LOC | 14,572 |
| Scripts Python LOC | 6,384 |

The current 14.4k-line production package is a natural size for the supported
API, identity, calendar, mutation, graph-schema, semantic planning, deterministic
execution, localization, and provider surfaces. Further merging would force
agents to reason across unrelated invariants inside larger files.

One concrete conditional cleanup remains: once the V2 contract evaluation is
accepted or rejected, remove either the V1 compatibility path or the opt-in V2
candidate. Deleting either before that decision would erase an active evaluation
boundary rather than converge completed architecture.

## Verification

```text
COVERAGE_FILE=/tmp/home-cortex-coverage-final.data \
  python -m pytest --cov=home_cortex --cov-branch \
  --cov-report=term-missing \
  --cov-report=json:/tmp/home-cortex-coverage-final.json -q

757 passed
```

Static compilation and `git diff --check` also pass. Real-LLM accuracy must still
be evaluated on the isolated production GPU host with recorded fingerprints.
