# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this project is

home-cortex is an experimental stationary cognition layer for a home humanoid
robot. It is a local API that grounds natural-language household questions in a
SurrealDB graph using a **semantic interpreter + deterministic executor**. The
LLM only interprets language into semantic meaning; it never computes answers and
never reads physical storage. The source code implements the *grammar of
reasoning*, not a catalog of expected questions.

## Active source (edit these)

- `src/home_cortex/` — the Python package. Main pieces:
  - `semantic/facts.py` — `SemanticFactService`, the utterance-to-answer coordinator.
    Import each concept directly from its owner:
    - `semantic/ir.py` — request/result types (`SemanticFactRequest`, `FactResult`)
    - `semantic/schema.py` — `SemanticSchemaRegistry` (vocabulary → catalog)
    - `semantic/planner.py` — LLM interpreter (`SemanticFactPlanner`)
    - `facts/resolver.py` — `EntityResolver` (self/name/path grounding)
    - `facts/engine.py` — deterministic executor, independent of planner/rendering
    - `facts/renderer.py` — answer text (`FactRenderer`)
  - `semantic/ontology.py` — declarative ontology model + validation.
  - `semantic/prompt.py` — interpreter system prompt and reusable examples.
  - `providers/` — provider contract plus Ollama and OpenRouter adapters.
  - `facts/operators.py` — deterministic operator/predicate registry.
  - `common/tracing.py` — reusable, opt-in runtime request/LLM timing.
  - `api/`, `runtime/`, `agents/` — HTTP, application coordination, and named agents.
  - `mutation/` — mutation intent, preview/commit service, and transactional writes.
  - `persistence/` — SurrealDB, retrieval, graph schemas, ingestion, and export.
- `schemas/` — declarative ontology (`semantic/ontology.yaml`) and edge schemas
  (`edge/*.yaml`). Reusable domain concepts belong here, not in prompts.
- `scripts/` — importable engineering utilities grouped as `benchmarks/`,
  `profiling/`, `probes/`, and `maintenance/`; see `scripts/README.md`. Runtime
  modules must not import scripts. Benchmark inputs remain in `benchmarks/`.
- `tests/` — pytest suite. Run with `python -m pytest -q`.
- `pyproject.toml` — package metadata and dependencies.

## Fast ownership guide

`AgentService` creates `AgentRequestContext` (`semantic/ir.py`); conversation
state adds scoped focus. `SemanticFactService` calls the planner, engine, and
renderer. `EntityResolver` grounds references; `HouseholdFactEngine` computes
facts. Both report failures as `FactResult`. `ResolvedEntities` is successful
grounding with traversal edges, not another answer format. SurrealDB owns facts;
`RetrievalService` owns graph reads. There is no Tier-0 factual route.

For relationship meaning or a new property, start in `schemas/semantic/ontology.yaml`
and the relevant edge YAML; `semantic/schema.py` binds these to deployed fields.
For speaker-relative or containment bugs, start in `facts/resolver.py` and the
matching `test_entity_alias_resolution.py`, `test_semantic_composition.py`, or
`test_collapsed_containment.py`. Add a new computation in `facts/operators.py`
and its executor integration. Do not restore the old cross-module re-export facade.

## Data / generated — do NOT treat as source, do NOT read wholesale

- `artifacts/` — **generated benchmark result JSONs** plus prose reports. Read only
  the `*-summary.json` files and `REPORT.md` / `REPORT-qwen35-9b.md`. The large
  per-case JSONs (`probe-iteration*.json`, `probe.json`, `suite.json`,
  `baseline-relevant-rows.json`) are derived output, **not in the working tree**
  (gitignored and removed), and should **not** be read in full. If you need one,
  restore it from git history: `git log --all -- <path>` to find a commit, then
  `git show <commit>:<path>` (e.g. `git show 45b8761:artifacts/tier1-baseline/probe.json`).
- `benchmarks/` — **benchmark dataset inputs** (YAML + fixtures). Data consumed at
  fixed paths by the benchmark harness and tests; keep those paths stable. Never
  copy benchmark wording into the interpreter examples.
- `data/` — **runtime household graph JSON** (nodes/edges are gitignored; only
  `Readme.md` is tracked). Not source; do not treat as a fixture for new tests.
- `docs/` — historical / long-form reports. Not the active architecture; see
  `src/README.md` for current design.

## Key architectural invariants (do not break)

- Do not add question-specific routing, implicit semantic repair, or
  household-specific behavior in exchange for accuracy.
- LLM interprets language; ontology defines reusable concepts; context sets the
  current speaker/household/time; resolvers and schema adapters ground references
  and map semantic properties/relations to storage; the deterministic executor runs
  generic operations; the renderer formats results.
- Property ownership is explicit: `property_source` ∈ {`entity`, `relationship`}.
  Do not silently turn an invalid entity-property request into a relationship
  property request, or vice-versa.
- Pairwise comparisons carry both operands (`subject` + `other`) and the ordered
  property through generic IR.
- Declared predicates (`adult`, `minor`) are distinct from operations. Do not
  remove an invalid predicate merely to make a plan executable.
- Ontology `concept` steps expand complete paths/filters during composition. Do not
  inspect utterance text downstream to restore "son"/"wife"/"father" filters, and
  do not substitute a broader relative for a declared concept.
- Accept plan equivalence only where execution semantics justify it; preserve
  distinctions involving intermediate entities, relationship properties, and
  contextual scope.
- The model must not receive household identity mappings or physical storage fields
  as a substitute for resolution. Literal names become `named_entity` references;
  contextual phrases become speaker-relative relation compositions.

## Workflow

- Before substantial work, inspect recent relevant entries in `.llm/` and verify
  their material assumptions against the current source.
- Every meaningful coding, debugging, refactoring, investigation, benchmark,
  architecture, or review cycle must end with a concise `.llm/` work log. Follow
  [the repository work-log skill](.agents/skills/persistent-llm-work-log/SKILL.md).
  The log is part of completion, alongside testing and reporting.
- Run the deterministic suite locally: `python -m pytest -q`.
- Real-LLM benchmark runs happen on the production GPU host from an isolated
  package with recorded data/schema/package fingerprints. Do not fabricate LLM
  accuracy numbers here.
- Report architecture changes, production accuracy changes, and evaluation
  corrections separately. Passing known questions is necessary but insufficient;
  completion requires preserving the layer boundaries and demonstrating
  compositional generalization.
