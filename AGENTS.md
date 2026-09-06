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
  - `semantic_facts.py` — semantic IR, schema registry, planner, deterministic
    executor, renderer, and the regression-locked invariants.
  - `semantic_ontology.py` — declarative ontology model + validation.
  - `ollama.py` — LLM client, interpreter system prompt, and reusable examples.
  - `operator_registry.py` — deterministic operator/predicate registry.
  - `semantic_planner_benchmark.py` / `fact_benchmark.py` — benchmark harness.
  - `api.py`, `agent_service.py`, `agents/` — HTTP and agent layer.
- `schemas/` — declarative ontology (`semantic/ontology.yaml`) and edge schemas
  (`edge/*.yaml`). Reusable domain concepts belong here, not in prompts.
- `tests/` — pytest suite. Run with `python -m pytest -q`.
- `pyproject.toml` — package metadata and dependencies.

## Data / generated — do NOT treat as source, do NOT read wholesale

- `artifacts/` — **generated benchmark result JSONs** plus prose reports. Read only
  the `*-summary.json` files and `REPORT.md` / `REPORT-qwen35-9b.md`. The large
  per-case JSONs (`probe-iteration*.json`, `probe.json`, `suite.json`,
  `baseline-relevant-rows.json`) are derived output, gitignored, and should **not**
  be read in full; they are kept on disk / in history only for on-demand inspection.
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

- Run the deterministic suite locally: `python -m pytest -q`.
- Real-LLM benchmark runs happen on the production GPU host from an isolated
  package with recorded data/schema/package fingerprints. Do not fabricate LLM
  accuracy numbers here.
- Report architecture changes, production accuracy changes, and evaluation
  corrections separately. Passing known questions is necessary but insufficient;
  completion requires preserving the layer boundaries and demonstrating
  compositional generalization.
