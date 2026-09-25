# Home Cortex — Repository Cleanup / Standardization Audit

**Date:** 2026-09-24
**Scope:** whole tracked repository (518 files)
**Kind:** review only. No source file was modified by this audit.
**Verification baseline:** `python -m pytest -q` → **983 passed in 19.33s** (clean tree, `master` @ `e689819`).

Every finding below cites a real file and line. Nothing here proposes a generic pattern; where
a "standard" practice would not fix an observed problem it is not recommended.

---

## 1. Executive Summary

The repository is in **materially better shape than its size suggests**. An earlier
reorganization (recorded in `.llm/2026-09-19_2210_functional-package-reorganization.md`) already
collapsed 43 root modules to 1, removed the re-export facades, and added architecture tests that
enforce an acyclic production import graph. 983 deterministic tests pass in 19 seconds with no
network, no live model, and no skips. That is a strong base, and this audit does **not** recommend
re-doing any of it.

The problems that remain are specific, local, and mostly traceable to one of two causes: (a) the
provider layer was extended (Ollama → OpenRouter → llama.cpp) without re-cutting the seam between
*transport* and *planning*; (b) the engineering tree (`scripts/`, `benchmarks/`, `artifacts/`) grew
by accretion of one-off experiments, each of which left behind a runner, a corpus, a report, and
sometimes a test that pins that experiment's numbers.

The ten most important findings, ordered by how much they cost:

1. **The provider contract embeds planning.** `providers/base.py:28-67` declares
   `plan_item_mutation`, `plan_semantic_fact`, and `plan_unified_semantic` on the "provider-neutral"
   `ModelProvider` protocol, and both adapters import `..semantic.prompt` and `..mutation.ir`
   (`providers/ollama.py:8-16`, `providers/openai_compatible.py`). Every adapter must therefore know
   the planner's prompt, seed, `num_predict`, and JSON-schema surgery. This is the single largest
   structural defect and the root cause of findings 2 and 3.

2. **The two live adapters duplicate the entire planner method surface.** `ollama.py` and
   `openai_compatible.py` each define `chat`, `stream_chat`, `chat_with_tools`,
   `stream_chat_with_tools`, `plan_item_mutation`, `plan_semantic_fact`, `plan_unified_semantic`,
   and `close`. The bodies differ only in how the response is decoded — and the two decode into
   *different types* (third-party `ChatResponse` vs. the local `providers/ir.py` protocol), which is
   why they were never merged. `ollama.py:12-16` additionally imports four symbols it never uses.

3. **`semantic/unified_planner.py` monkey-patches `semantic/planner.py`'s prompt.**
   `unified_chat_messages()` rebuilds `prompts.planner_chat_messages(...)`, string-replaces element
   `[0]["content"]`, guards the assumption with `raise RuntimeError("semantic planner opening
   changed")` (`unified_planner.py:372`), and computes a positional index
   `reminder_index = 1 + len(prompts._semantic_planner_examples())` (`:456`). It also imports five
   private symbols from `planner.py` (`:21-27`). Two planners that must stay byte-compatible are
   coupled by positional arithmetic instead of by a shared builder.

4. **`semantic/planner.py` still contains utterance-regex repair guards.** `_SECOND_PERSON_IDENTITY`,
   `_FIRST_PERSON_IDENTITY`, `_NAMED_OBJECT_LOCATION` (`planner.py:203-216`) feed
   `_identity_person_mismatch` / `_object_location_mismatch`, which re-read the raw utterance after
   planning. This is in tension with the stated invariant in `src/README.md:52-54` and
   `AGENTS.md`. It is deliberately scoped (an IR-based `_invalid_plan_retry_hint` at `:243` is
   documented as *not* matching on text), so this is a **boundary erosion to document, not to rip
   out**.

5. **The test suite has no `tests/conftest.py` and exactly 6 fixtures for 18,286 LOC.** Reuse is
   achieved by importing test modules (`test_semantic_composition.py:521`,
   `test_calendar.py:21-28`, `test_entity_alias_resolution.py:17`). `MemoryDatabase` is
   copy-pasted verbatim into 7 files; four independent fake-Ollama implementations exist.

6. **The engineering tree carries a real dead-code tail.** `scripts/benchmarks/tier1_latency_bench.py`
   (153 LOC) is fully superseded by `hc_suites.py:468-489` (`latency` suite);
   `scripts/maintenance/copy_tier1_bench_into_api.sh` (59 LOC) has a default path
   (`docker/cortex`, line 8) that no longer exists. Six more scripts are referenced *only* by the
   `--help` sweep in `tests/test_engineering_entrypoints.py:21-38`.

7. **`tests/` has no structure and no markers.** 56 test files sit flat in one directory; the 17-subprocess
   `test_engineering_entrypoints.py` runs alongside the 9-LOC `test_gui_session.py`. There is no
   `unit`/`integration`/`architecture` mark and no `-m` expression in `pyproject.toml`, so "run the
   fast tests" is not expressible.

8. **Benchmark corpora have no single source of truth** — 11+ independent case definitions, one of
   them (`fact_benchmark.py:35-95`, 33 `QUESTIONS` + 5 `SPEAKER_CASES`) a bare Python literal with
   no YAML backing. `benchmarks/composition/frozen/MANIFEST.yaml` re-lists all 36 frozen cases and
   is **absent from the hash map** in `benchmarks/composition/fingerprints.json`, so it can drift
   undetected.

9. **Configuration has one provably dead variable and no documented env surface.**
   `HOME_CORTEX_DISABLE_TIER0` is set in both compose files (`docker-compose.yml:83`,
   `docker-compose.llamacpp.yml:106`) and has no `Settings` field; `extra="ignore"`
   (`config.py:35`) makes it silent. The two compose files also duplicate the `home-gui`,
   `home-media`, `proxy`, and `surrealdb` service blocks verbatim.

10. **Small but concrete duplication in the two sibling packages.** The GUI duplicates its fetch
    wrapper and error envelope across `lib/api.ts:3-22` and `lib/media.ts:38-93` with two
    incompatible error types; `home_media` keeps the path-containment rule in both
    `content.py:21-30` and `scanner.py:61-74`, and the thumbnail cache path
    `thumbnail_root/{id}.jpg` in three places. Both packages are otherwise clean, independent, and
    free of tracked build artifacts.

**Overall recommendation:** four focused stages (Section 19), of which **Stages 1 and 2 are safe
and high-value**, Stage 3 is mechanical, and Stage 4 touches semantics and must be gated on the
real-model benchmark. Total *deletion* realistically reachable: **≈1,300–1,900 LOC**, plus a much
larger amount of *de-duplication* (Section 20 distinguishes the two).

---

## 2. Current Repository Map

```
home-cortex/                        518 tracked files
├── src/
│   ├── home_cortex/       148 files   21,626 LOC    the Python service (the active product)
│   ├── home_gui/          25 files    1,554 LOC     Svelte 5 + Vite client (own Dockerfile)
│   └── home_media/        19 files    1,227 LOC     separate package (own pyproject + lock + tests)
├── tests/                 65 files   18,503 LOC     56 test files + 9 static fixtures, flat, no conftest
├── scripts/               37 files    8,800 LOC     engineering tree: benchmarks/ probes/ profiling/ maintenance/
├── benchmarks/            49 files    6,223 LOC     dataset inputs (YAML + fixtures + composition corpus)
├── artifacts/            156 files   76,765 LOC     GENERATED results + prose reports (not source)
├── schemas/                6 files      434 LOC     ontology.yaml (389) + 5 edge YAMLs
├── docker/                 6 files      361 LOC     4 compose files + nginx.conf + monitoring/
├── docs/                   5 files      810 LOC     historical design docs
├── .llm/                  33 files    3,573 LOC     work logs (33 entries + this audit)
├── .agents/                1 file        58 LOC     the work-log skill
├── data/                   1 file       131 LOC     runtime graph (nodes/edges gitignored)
├── Readme.md, AGENTS.md, setup_v0.2.md, pyproject.toml, Dockerfile, uv.lock
└── dist/  tmp/  .venv/  .pytest_cache/            (untracked; see §11)
```

**`src/home_cortex` package ownership** (this is the layer map that matters):

| Package | Files | LOC | Owns |
|---|---:|---:|---|
| `semantic/` | 11 | 4,310 | IR, ontology, schema binding, prompts, both planners |
| `facts/` | 5 | 2,652 | grounding, deterministic engine, operators, rendering |
| `benchmark/` | 14 | 2,502 | CLI harness — **zero production importers** (§12) |
| `persistence/` | 8 | 1,820 | SurrealDB, retrieval, edge/schema catalogs, ingestion, export |
| `spatial/` | 10 | 1,610 | contracts (production) + localization math (tests only) |
| `api/` | 14 | 1,571 | FastAPI app, deps, errors, SSE, execution, 5 route modules |
| `capabilities/` | 5 | 1,554 | calculation, calendar, model-facing catalog, dispatcher |
| `mutation/` | 5 | 1,098 | mutation IR, semantic intent, service, transactional writes |
| `vision/` | 4 | 1,069 | **paused** — contracts + ports only, no production importer |
| `runtime/` | 3 | 1,028 | `agent.py`, `model_loop.py` |
| `providers/` | 7 | 844 | base protocol + ollama / openai_compatible / openrouter / llamacpp |
| `conversation/` | 4 | 641 | transcript store, GUI sessions, greetings |
| `common/` | 5 | 603 | identity, display, text, tracing |
| `agents/` | 4 | 144 | named-agent registry + `steward/` definition |
| `config.py` | 1 | 176 | process settings |

---

## 3. Complexity Metrics

Measured on `git ls-files`, excluding `node_modules/`, `dist/`, `uv.lock`, `package-lock.json`.

| Tree | Files | LOC | Nature |
|---|---:|---:|---|
| `artifacts/` | 156 | 76,765 | generated JSON (145 `.json` files repo-wide, mostly here) |
| `src/` | 148 | 25,135 | product source |
| `tests/` | 65 | 18,503 | deterministic suite |
| `scripts/` | 37 | 8,800 | engineering utilities |
| `benchmarks/` | 49 | 6,223 | dataset inputs |
| `.llm/` | 33 | 3,573 | work logs |
| `docs/` + `schemas/` + `docker/` | 17 | 1,605 | reference + deployment |
| **total tracked text** | **518** | **141,266** | |

The headline number is misleading and worth stating plainly: **`artifacts/` is 54% of the tracked
LOC and 100% of it is generated output, not source.** Excluding it, the repository is ~64,500 LOC
of human-authored text, of which the Python product is 21,626 LOC.

### 3a. Modules ≥ 500 LOC (12 in `src/home_cortex`)

| LOC | Module | Classification |
|---:|---|---|
| 914 | `semantic/schema.py` | **A** — cohesive; vocabulary→catalog binding, one concern |
| 837 | `vision/contracts.py` | **A** for now — paused, contract-only, no production importer |
| 828 | `facts/engine.py` | **A** — the deterministic executor; splitting risks the invariant |
| 763 | `capabilities/calendar.py` | **B** — Google OAuth + token refresh + event normalization + tz handling + authorization in one file |
| 697 | `mutation/writing.py` | **A** — transactional writes; "fixed SurrealQL, no caller queries" is the point |
| 648 | `runtime/model_loop.py` | **B** — `run()` and `stream()` are parallel implementations of one loop (§8.3) |
| 642 | `semantic/unified_planner.py` | **B/C** — see §9.1; contains large inline example tuples |
| 638 | `facts/operators.py` | **A** — the operator/predicate registry; declarative, meant to be long |
| 628 | `semantic/ontology.py` | **D** — ontology model + validation; data-shaped, acceptable |
| 618 | `benchmark/environment.py` | **B + duplicated** — provenance/hashing; see §5.4 |
| 609 | `facts/resolver.py` | **A** — `EntityResolver`; the grounding boundary |
| 576 | `facts/renderer.py` | **A** — answer text; bilingual by nature |

Only 3 modules exceed 800 LOC. There is **no module over 1,000 LOC in `src/`** — the largest is
914. This is a healthy profile and argues against any "split the big files" program.

### 3b. Modules < 40 LOC (27 in `src/home_cortex`)

Of these, 13 are 1-line package docstrings (the deliberately-empty initializers documented in
`src/README.md:84-90`). The genuinely small *code* modules:

| LOC | Module | Verdict |
|---:|---|---|
| 7 | `providers/llamacpp.py` | **Pure pass-through** (whole file is a docstring + `class LlamaCppService(OpenAICompatibleService): """..."""`). Keep: it is the config-selectable adapter (`providers/base.py:77-83`) and is tested (`tests/test_llamacpp.py`). |
| 8 | `agents/steward/tools.py` | **Keep** — a policy constant, dynamically imported by `agents/registry.py:60`. Not dead (§6 caveat). |
| 10 | `spatial/primitives.py` | **Keep** — leaf constants/error breaking the only import SCC (per `.llm/2026-09-19_2210`) |
| 21 | `providers/openrouter.py` | **Keep** — validates key, sets `provider_parameters=True`; real logic |
| 30 | `common/text.py` | **Keep** — shared `latest_user_message` |
| 31 | `benchmark/plugins.py` | **Keep, but see §4.2** — contains the inverted `scripts` import |
| 31 | `mutation/service.py` | **Thin, justified** — the preview/commit boundary; validates then delegates |
| 33 | `common/identity.py`, `33 providers/ir.py` | **Keep** |

**Wrapper count is low.** The re-export facades were already removed (`.llm/2026-09-19_2210:24-25`)
and are guarded by `tests/test_backend_architecture.py:140`. The only remaining pass-through is
`providers/llamacpp.py`, and it is load-bearing.

---

## 4. Dependency Findings

### 4.1 Simplified production dependency diagram

Edges are real imports observed by AST scan of `src/home_cortex`. `→` reads "imports".

```
                          config.py  ◄─────────── (api, persistence, providers, runtime wire it)
                              │
  api/ ──► agents/ ──► capabilities/ ──► facts/ ──► semantic/
   │           │             │               │            ▲
   │           │             └──► mutation/ ─┘            │
   │           │                    │                     │
   ├──► conversation/ ──────────────┼─────────────────────┘
   ├──► persistence/ ───────────────┘        persistence ──► spatial.contracts
   ├──► providers/ ──► ⚠ semantic.prompt, mutation.ir      (production, via ingestion)
   └──► runtime/ ──► semantic/, mutation/, capabilities/, providers/
                       │
                       └──► common/  ◄── everything
  vision/ ──► spatial/          (no production importer — tests only)
  benchmark/ ──► semantic/, providers/, config ──► ⚠ scripts.benchmarks.hc_suites
```

The graph is **acyclic** — enforced by `tests/test_backend_architecture.py:79`
(`test_production_import_graph_is_acyclic`, real AST + `TopologicalSorter`). Two edges are wrong:

- **`providers/ → semantic/` + `providers/ → mutation/`** (§4.2) — the layering inversion.
- **`benchmark/ → scripts/`** (§4.3) — a documented "must not" that the guard cannot see.

### 4.2 The provider layering inversion

`providers/base.py:10` says *"Provider-neutral language-model contract"*. Its protocol body
disagrees:

```python
# providers/base.py:28-67  (abridged)
class ModelProvider(Protocol):
    model: str
    last_planner_runtime: dict[str, Any]          # planner state on a transport object
    async def plan_item_mutation(self, messages): ...
    async def plan_semantic_fact(self, messages, capabilities, output_schema, *, household_now): ...
    async def plan_unified_semantic(self, messages, output_schema): ...
```

And both concrete adapters import across the boundary:

- `providers/ollama.py:8` → `from ..mutation.ir import MutationDecision, mutation_messages, read_plan_schema, attribute_output_schema`
- `providers/ollama.py:9-16` → `from ..semantic.prompt import PLANNER_NUM_PREDICT, PLANNER_SEED, planner_chat_messages, ...`
- `providers/openai_compatible.py` imports the same two modules.

**Consequence:** any change to the interpreter prompt, the plan JSON schema, or the mutation
decision schema is a provider-layer change. This directly produced §5.1 (the duplicated method
surface) and §8.1 (Ollama-shaped constants in a "generic" adapter).

**Evidence that this is not theoretical:** `tests/test_backend_architecture.py:91-104` contains
three separate guard tests whose entire purpose is to stop this from getting *worse*
(`test_provider_contract_import_does_not_choose_or_load_an_adapter` and friends). The guards
protect the factory's laziness; they do not address the contract's content.

### 4.3 `benchmark/ → scripts/` — an invariant the guard cannot see

- `src/home_cortex/benchmark/plugins.py:30` — `importlib.import_module("scripts.benchmarks.hc_suites")`
- `pyproject.toml` — `[project.entry-points."home_cortex.benchmark_suites"] builtin = "scripts.benchmarks.hc_suites:register"`
- `scripts/README.md:3-4` — *"Runtime code lives in `src/home_cortex/` … runtime modules never
  import them."*
- `AGENTS.md:35-37` — *"Runtime modules must not import scripts."*

`tests/test_engineering_entrypoints.py:13-18` walks `ast.Import`/`ast.ImportFrom` and therefore
**cannot see a string module name**. This is deliberate and documented in `plugins.py`'s own
docstring ("Benchmark commands are the exception"), but it means the rule is enforced by convention
only on this path. Note the blast radius: `benchmark/` is only reachable from the CLI, so the
architectural cost is low — but the *statement* in `AGENTS.md` and the *test* now disagree.

### 4.4 Other observed edges

- `capabilities/dispatcher.py:31-35` constructs `MutationService` and `HouseholdFactEngine`
  *inside `ToolDispatcher.__init__`*, using `writing.catalog` / `writing.ontology`. A transport-
  facing dispatcher reaching into the write service for the engine's schema registry is an
  assembly concern that belongs in `api/app.py`'s composition root.
- `persistence/ingestion.py:19-24` → `spatial.contracts` — **the only production consumer of
  `spatial/`**, and it uses `contracts.py` only.
- `vision/` has no production importer at all (§6.3).

---

## 5. Duplication Findings

### 5.1 The entire planner method surface, twice (the largest single duplication)

| Method | `providers/ollama.py` | `providers/openai_compatible.py` |
|---|---|---|
| `chat` | :46 | :~150 |
| `stream_chat` | :60 | :~170 |
| `chat_with_tools` | :82 | :~200 |
| `stream_chat_with_tools` | :215 | :~260 |
| `plan_item_mutation` | :96 | :~290 |
| `plan_semantic_fact` | :108 | :~320 |
| `plan_unified_semantic` | :139 | :~370 |
| `close` | :235 | :~430 |

Both files build `options={"temperature": 0, "num_ctx": …, "num_predict": PLANNER_NUM_PREDICT,
"seed": PLANNER_SEED}`, call `read_plan_schema(...)` / `attribute_output_schema(...)`, and store
`last_planner_runtime` with `done_reason`. The *only* real difference is decoding: `ollama.py` gets
a third-party `ChatResponse` and reads `.message.content`, while `openai_compatible.py` normalizes
the OpenAI wire shape into `providers/ir.py` types. `providers/openrouter.py` (21 LOC) and
`providers/llamacpp.py` (7 LOC) override *nothing* — they are pure configuration over the shared
transport. **`ollama.py` is the only adapter that does not use the shared transport**, and the
stated reason (`.llm/2026-09-19_1450_backend-consolidation.md`) is that OpenRouter "still
constructs the third-party Ollama `ChatResponse` model internally as its wire-normalization
object" — i.e. the duplication exists to paper over a type mismatch.

### 5.2 Runtime-metric conversion, twice

- `providers/ollama.py:239-259` — `_ns_to_ms` + `_ollama_runtime_metrics`
- `providers/openai_compatible.py` — `_runtime_metrics`

Both produce the same six keys (`prompt_eval_count`, `prompt_eval_duration_ms`, `eval_count`,
`eval_duration_ms`, `load_duration_ms`), one from nanoseconds one from seconds.

### 5.3 Dead imports in `providers/ollama.py`

`ollama.py:12-16` imports `_PLANNER_HISTORY_BOUNDARY`, `_PLANNER_INSTRUCTIONS`,
`_semantic_planner_examples`, `planner_system_prompt` — **four symbols never referenced in the
file**. `planner_chat_messages` and the constants are used; these four are residue from when the
prompt builder lived here (it was extracted in `.llm/2026-09-19_1450`).

### 5.4 Two provenance/hashing stacks

`scripts/benchmarks/semantic_planner_benchmark.py` owns `_sha256_file` (:716), `_sha256_tree`
(:727), `collect_provenance` (:743), `_git_provenance` (:808), `_ollama_provenance` (:837).
`src/home_cortex/benchmark/environment.py` owns `sha256_file` (:74), `hash_tree` (:82),
`cached_tree_hash` (:99), `git_metadata` (:148), `hardware_metadata` (:172), `ollama_metadata`
(:205), `collect_environment` (:445). Same jobs, kept in sync by hand. Same for percentiles:
`fact_benchmark._percentile` (:419) vs `benchmark/stats.percentile` (:8), and latency summaries
`semantic_planner_benchmark.summarize_latencies` (:666) vs `benchmark/stats.latency_summary` (:21).

### 5.5 Test-suite duplication (see §12 for the full treatment)

- **`MemoryDatabase` copy-pasted into 7 files**: `test_retrieval.py:32`, `test_ingestion.py:17`,
  `test_semantic_facts.py:101` (as `_MemoryDatabase`), `test_export.py:29`, `test_hosted_by.py:16`,
  `test_spatial.py:27`, `test_writing.py:23`. All open with `AsyncSurreal("mem://")` and redefine
  `connect`/`close`/`query`/`upsert` identically; they differ only in the `use()` database name.
- **Four independent fake-Ollama implementations**: `test_agent_service.py:33`
  (`FakeOllamaService`), `test_ollama.py:22` (`FakeOllamaClient`), `test_semantic_composition.py:51`
  (`Client`), `test_semantic_facts.py:123` (`_ChatOllama`).
- **`STATIC_TEST_DATA` path constant re-declared in 7 files** (`test_semantic_facts.py:38`,
  `test_retrieval.py:11`, `test_writing.py:20`, `test_spatial.py:23`, `test_hosted_by.py:13`,
  `test_export.py:19`, `test_ingestion.py:14`).
- **`AgentRequestContext(...)` constructed inline 17 times** with no shared builder.
- **Three different registry-loading idioms** for the same thing: `load_default(tmp_path)`
  (`test_collapsed_containment.py`), `load_default()` (`test_query_generalization.py:41`),
  `from_directory(SCHEMA_DIR)` (`test_entity_alias_resolution.py:40`).

### 5.6 Duplication in `home_media` and `home_gui`

- `content.py:21-30` and `scanner.py:61-74` both implement `is_relative_to(canonical_root)`; the
  scanner version deliberately omits the `..`/absolute check but still never reuses `content.py`.
- `thumbnail_root/{media_id}.jpg` is reconstructed in three places (`thumbnails.py:30`,
  `scanner.py:105`, `scanner.py:113`) with no `path_for()` helper.
- Media-ID format defined twice with nothing tying them: generator `scanner.py:25-27`
  (`sha256(f"{media_type}\0{relative_path}")[:24]`) vs. validator `app.py:22`
  (`^media_[0-9a-f]{24}$`).
- GUI: `lib/api.ts:3-22` and `lib/media.ts:38-93` are the same fetch wrapper with two incompatible
  error types (`ApiError` class vs. `Error & {status, code}`); `api.ts:78-86` re-implements inside
  `streamMessage` the error parse `request()` already does at `:13-19`; the normalization
  `error instanceof Error ? error.message : String(error)` appears 5 times.

### 5.7 Duplication in deployment config

`docker/docker-compose.yml` and `docker/docker-compose.llamacpp.yml` duplicate the `proxy`,
`home-gui`, `home-media`, and `surrealdb` service blocks verbatim. They are not `extends`-linked.
The `gpu` variants are correct — 3-line overlay files (`docker-compose.gpu.yml` adds `gpus: all`
to `ollama`; `docker-compose.gpu.llamacpp.yml` adds it to `llama-server`). **The overlays are the
pattern the base files should also follow.**

---

## 6. Dead / Obsolete Code

The audit deliberately does **not** call something dead because production Python does not import
it. Each item below was checked against `src/`, `tests/`, `scripts/`, `pyproject.toml`, `docker/`,
`docs/`, `benchmarks/`, and `.llm/`.

### 6.1 Dead — safe to delete

| Item | LOC | Evidence |
|---|---:|---|
| `scripts/maintenance/copy_tier1_bench_into_api.sh` | 59 | Superseded by the `hc-bench` harness (which records provenance and compares runs — `scripts/README.md:16-18`). Three independent breakages: (a) its default `COMPOSE_DIR` is `$ROOT/docker/cortex` (line 9) and `docker/` has no `cortex/` — the header comment even documents "from repo root or **docker/cortex**", a pre-reorg path; (b) it hard-fails without `scripts/benchmarks/tier1_latency_bench.py` (lines 13-16), so it dies with the row below; (c) it `docker compose cp`s `scripts/` + `src/home_cortex` into a container to run a probe by hand, which is exactly what `hc-bench` now does properly. Note `scripts/README.md:11` *does* still advertise "container copying" in `maintenance/` — that line needs updating in the same commit. Referenced elsewhere only by `artifacts/tier1-baseline/REPORT-*.md`, both citing the pre-reorg path `scripts/copy_tier1_bench_into_api.sh`. |
| `scripts/benchmarks/tier1_latency_bench.py` | 153 | Its whole body wraps `run_tier1_probe`, which `hc_suites.py:475,489` already calls for the registered `latency` suite (`hc_suites.py:84`). `benchmarks/HARNESS.md:124` says so explicitly. No console script in `pyproject.toml`. Only live ref is the `--help` sweep. |
| `HOME_CORTEX_DISABLE_TIER0` | 2 lines ×2 files | Set at `docker-compose.yml:83` and `docker-compose.llamacpp.yml:106`; no `Settings` field; silently swallowed by `extra="ignore"` (`config.py:35`). Repo-wide grep finds only those two lines. |
| `src/home_cortex/http/` | 0 | **Empty directory tree** (`http/routes/`), untracked, left over from the pre-`api/` layout. |
| `vision/ports.py:63,71,85` | — | `ObservationSink`, `EvidenceClipSink`, `LiveStreamDirectory` — the only three classes in all of `src/` with **zero code references**: the sole hits outside their own definitions are prose mentions in `.llm/` work-log entries. **But see §6.3 — this is deliberate, so this is a "note", not a deletion candidate.** |

### 6.2 Referenced only by the `--help` sweep — candidates, not certainties

`tests/test_engineering_entrypoints.py:21-38` parametrizes over 17 `scripts.*` modules and runs
`python -m <mod> --help` from a temp directory. For these six, that test is the **only** live
reference (otherwise `.llm/` history + a historical `artifacts/*/REPORT.md`):

| Script | LOC | Was it an experiment? |
|---|---:|---|
| `scripts/probes/layer_failure_trace.py` | 461 | `artifacts/layer-failure-trace/REPORT.md` |
| `scripts/probes/item_location_probe.py` | 142 | no report |
| `scripts/probes/kinship_context_probe.py` | 120 | `artifacts/kinship-context/REPORT.md` |
| `scripts/profiling/profile_semantic_transport.py` | 129 | `artifacts/compact-transport/REPORT.md` (the codec was **withdrawn**: `docs/design/compact-semantic-transport.md:3`) |
| `scripts/profiling/http_latency_audit.py` | 106 | no report |
| `scripts/profiling/token_component_probe.py` | 55 | no report |
| `scripts/maintenance/context_surface_audit.py` | 84 | absent from `scripts/README.md` |

**Important caveat:** these are plausible-to-delete but each one is a *reproduction* artifact for a
published report. Deleting them removes the ability to re-run the analysis. Recommend: keep the
three with reports, retire the four without — and update the `test_engineering_entrypoints.py`
parametrize list in the same commit, or the suite fails on a missing module.

### 6.3 Deliberately paused, NOT dead — do not delete

- **`vision/` (1,069 LOC)** — zero production importers, contract-only. But this is *documented
  intent*: `vision/README.md:1` ("Paused Vision backend contracts"), `src/README.md:59-65`
  ("paused pending MicroDuck hardware"), and two architecture guards enforce that it stays out of
  the chat import graph (`test_backend_architecture.py:76`, `test_vision_architecture.py:48`).
- **`spatial/localization/**`, `spatial/units.py`, `spatial/transforms.py` (≈1,290 LOC)** — importer
  is tests only. `spatial/contracts.py` **is** production (via `persistence/ingestion.py:19-24`,
  the `/admin/ingest` path). The empty package initializer is load-bearing:
  `spatial/__init__.py:1-7` and `test_backend_architecture.py:72-73` exist so chat startup does not
  load localization math.
- **`providers/llamacpp.py` (7 LOC)**, **`agents/steward/tools.py` (8 LOC)** — live via
  `providers/base.py:77-83` and `agents/registry.py:60` (dynamic import) respectively. Grep-based
  dead-code tooling will misclassify both.
- **`scripts/probes/ollama_warm_load_probe.py` (528)** and
  **`ollama_prefix_reuse_probe.py` (171)** — Ollama-specific response fields (`load_duration`).
  Still meaningful **because the llama.cpp migration was rejected**
  (`artifacts/llamacpp-migration/REPORT.md`: "Decision: do not promote";
  `.llm/2026-09-24_2228_llamacpp-migration-eval.md:35-39`). Don't delete on the assumption the
  migration happened.
- **`scripts/probes/two_stage_semantic_planner.py` (77)** — named "legacy", but imported by
  `mutation_routing_probe.py:15` and `unified_planner_experiment.py:35`; it *is* the control arm.

### 6.4 Stale by version, not by topic

`artifacts/ollama-warm-load/REPORT.md` analyzes **Ollama 0.32.13**; `docker-compose.yml:2` deploys
**0.34.4**, and `artifacts/llamacpp-migration/REPORT.md` confirms 0.34.4 is current. The 0.32.x
`load_duration` conclusion should be read as version-scoped. `REPORT-qwen35-4b.md` is the 0.32.15
redo, also behind.

### 6.5 Not dead, but should not be in a generated-output tree

`artifacts/analysis/build_analysis.py` — **35 KB of source inside `artifacts/`**, referenced by
nothing in `scripts/`, `tests/`, or `docs/`. Also tracked: 5 `candidate-*.tar.gz` archives and two
**completely empty directories** (`artifacts/context-surface-cleanup/`,
`artifacts/vision-architecture-review/`).

---

## 7. Package / Structure Findings

### 7.1 Root namespace — already clean, keep it that way

`src/home_cortex/` contains exactly `__init__.py` (4 LOC) and `config.py` (176 LOC). The root
namespace test (`test_backend_architecture.py:86`) asserts `root_modules == {"__init__.py",
"config.py"}`. This is intentional brittleness and it is cheap. **No change recommended.**

### 7.2 `api/__init__.py` is a 42-LOC facade with mostly-unused exports

It re-exports 16 names (`api/__init__.py:24-41`). Of those, **4** are actually imported from
`home_cortex.api` anywhere in the repository (AST-resolved, ignoring same-name collisions):

| Name | Consumers |
|---|---|
| `app` | `scripts/profiling/http_latency_audit.py:19`, `tests/test_open_webui_conversations.py:7`, `tests/test_semantic_composition.py:472`, `tests/test_api.py:15` |
| `VIRTUAL_MODEL` | `tests/test_open_webui_conversations.py:7`, `tests/test_semantic_composition.py:472`, `tests/test_api.py:15` |
| `ConversationStore` | `tests/test_semantic_composition.py:512`, `tests/test_api.py:15` |
| `_stream_chat_completion` | `tests/test_api.py:15` (a private-underscore name in `__all__`) |
| `APIError`, `ChatCompletionRequest`, `ChatMessage`, `ConversationCreateRequest`, `ConversationMessageRequest`, `DEFAULT_AGENT_ID`, `ExportRequest`, `MODEL_CREATED`, `REQUEST_ID_HEADER`, `SessionRequest`, `create_app`, `lifespan` | **never imported from `home_cortex.api`** — consumers reach past it into `..api.errors`, `..api.schemas`, `..api.sse` directly |

This is **not** the forbidden re-export facade (`AGENTS.md:55` targets the *cross-module* facade
that was removed), because `api:` is a real deployment entry point (`src/README.md:88-90`). But 12
of the 16 exports are dead surface, and the four that are used all come from four test/script
files. `test_backend_architecture.py:140` also caps this file at `len(api_lines) < 100` — an
arbitrary threshold unrelated to correctness.

### 7.3 `tests/` is entirely flat

56 `.py` files in one directory — no `unit/`, `integration/`, `architecture/`, or `semantic/`
subdirectories, and no pytest markers. Meanwhile `src/home_media/tests/` **does** nest with a
conftest. **Two opposite conventions in one repository.** Full treatment in §12.

### 7.4 `scripts/` organization is good; the tail is not

`benchmarks/ probes/ profiling/ maintenance/` with a 115-LOC `scripts/README.md` that explains each
grouping, and `scripts/__init__.py:1-11` holds only `PROJECT_ROOT` with no re-exports. The
structure is right. The problem is entirely in the 8 files listed in §6.1/§6.2.

### 7.5 `benchmarks/` mixes two kinds of file

The corpus YAMLs (`semantic_planner_eval.yaml`, `mutation_routing.yaml`, …) and the infrastructure
docs (`HARNESS.md`, `README.md`, `matrix.example.yaml`) sit beside `composition/` (a 4-subdirectory
corpus with its own README, coverage matrix, annotation guide, and fingerprints). That is fine.
The gap: `benchmarks/tier1_latency.yaml` is **referenced by name** in
`scripts/benchmarks/tier1_latency_bench.py` but **does not exist** in the tree.

### 7.6 `src/README.md` is the architecture authority and is accurate

It documents the package map, the ownership rules, the HTTP surface, and the explicit non-goals,
and its claims match the source. It should be the anchor for the target tree in §18. Two small
drifts: `src/home_gui/HOME-CORTEX.md:6-8` never mentions `/media-api` (which the GUI requires), and
`src/home_media/README.md:17-20` documents `HOME_MEDIA_FFPROBE_PATH`/`HOME_MEDIA_FFMPEG_PATH` that
`docker-compose.yml:39-44` never sets.

---

## 8. Provider / Runtime Findings

### 8.1 `providers/ollama.py` carries Ollama configuration in a generic-looking form

```python
# providers/ollama.py:27-31
OLLAMA_KEEP_ALIVE = "24h"
OLLAMA_NUM_CTX = 16384
# Retain the planner names used by benchmark fingerprinting and probe scripts.
PLANNER_KEEP_ALIVE = OLLAMA_KEEP_ALIVE
PLANNER_NUM_CTX = OLLAMA_NUM_CTX
```

The four planner methods then use `PLANNER_KEEP_ALIVE`/`PLANNER_NUM_CTX` while the three chat
methods use the `OLLAMA_*` names — two aliases for the same values, kept for
`scripts/` fingerprinting. The comment at `:256` in `probe` scripts and
`tests/test_ollama.py:224` (`characters <= (OLLAMA_NUM_CTX - PLANNER_NUM_PREDICT) * 3`) both depend
on these names. **Renaming them is a cross-tree change; the duplication of names is itself the
symptom.**

### 8.2 `runtime/model_loop.py` is documented and worded as if only Ollama exists

- Docstring line 1: *"Bounded **Ollama** tool loop"*.
- `"Ollama emitted tool calls after final-answer content"` (error message)
- `"Ollama requested {n} tools in one step"` (error message)

The module imports `providers.base.ModelProvider` and works with any adapter. The wording predates
OpenRouter and llama.cpp. Cosmetic, but it is the kind of thing that makes a reader think the
abstraction is fake.

### 8.3 `run()` and `stream()` are parallel implementations of one loop

`runtime/model_loop.py` (648 LOC) implements the bounded loop twice — once collecting, once
streaming. Both enforce `MAX_AGENT_STEPS=4`, `MAX_TOOL_CALLS_PER_STEP=4`, `MAX_TOOL_RECORDS=25`,
`MAX_TOOL_RESULT_BYTES=16384`, `TOOL_EXECUTION_TIMEOUT_SECONDS=5.0`, and both must emit identical
tool-call semantics. **This is the one large-module split I would genuinely consider**, and only
after §8.1/§8.2 are settled, because the two implementations share the step/tool budget state.

### 8.4 `runtime/agent.py` names its constructor parameter `ollama`

`AgentService.__init__(... ollama: ModelProvider ...)` and `self._ollama` throughout a 379-LOC
module. Same cosmetic issue as §8.2.

### 8.5 The planner is selected by a tool-list membership test

```python
# runtime/agent.py
semantic_planner = (
    UnifiedSemanticPlanner(ollama, semantic_schema)
    if self._mutation_enabled
    else SemanticFactPlanner(ollama, semantic_schema)
)
```

`_mutation_enabled` derives from whether the agent's `allowed_tools` include `write_item`
(`agents/steward/tools.py`). This is *implicit* capability→planner routing. It works and is not
user-visible, but `src/README.md` describes the agents as differing by "prompt, identity, settings,
and tool allowlist" — the planner choice is a fifth, undocumented axis. **Recommend documenting it
in `src/README.md`, not changing it.** Changing it risks the multi-intent safety gate.

### 8.6 llama.cpp is in the tree but not promoted — correctly

`providers/llamacpp.py` (7 LOC) is opt-in via `LLM_PROVIDER=llamacpp`, in
`docker-compose.llamacpp.yml` (+ `.gpu.llamacpp.yml`), and tested
(`tests/test_llamacpp.py`, hermetic — `httpx.MockTransport`, no live server).
`artifacts/llamacpp-migration/REPORT.md`: candidate plan score 95/119 vs. baseline 86/119, but
**multi-intent 3/7 vs. 4/7 — a regression**, plus 4 partial plans vs. 3; comparison exit code 3;
verdict "do not promote." Default stays Ollama. **The tree reflects this correctly.** No action.

---

## 9. Semantic / Planner Findings

*This is the correctness-critical area. Per the task constraints, findings here are conservative
and the default recommendation for every one of them is "document, don't touch."*

### 9.1 `unified_planner.py` rebuilds and patches `planner.py`'s prompt by string surgery

```python
# semantic/unified_planner.py:372-373
        raise RuntimeError("semantic planner opening changed")
    built[0]["content"] = built[0]["content"].replace(...)
# semantic/unified_planner.py:456-457
    reminder_index = 1 + len(prompts._semantic_planner_examples())
    built[reminder_index]["content"] += (...)
```

`unified_chat_messages()` calls `prompts.planner_chat_messages(...)`, then:
1. string-replaces the content of message `[0]`, guarded by `raise RuntimeError(...)` if the
   opening changed;
2. edits message `[reminder_index]` where the index is computed as *one plus the current example
   count*.

It also imports five private symbols from the sibling module (`:21-27`): `_identity_person_mismatch`,
`_object_location_mismatch`, `_planner_clock`, `_planner_runtime_fields`, `planner_input_summary`.

**Why this matters:** the two planners must produce byte-compatible prefixes for Ollama's prompt
cache. The `RuntimeError` guard is a *good* defensive choice — it fails loudly. But positional
arithmetic over a builder owned by another module means adding a planner example silently moves
`reminder_index`. The correct fix is to make `planner.py` accept the opening and reminder
positions as parameters, or to expose a single `build_planner_messages(..., variant=...)`. **This
is a refactor of prompt construction, so it must be gated on the real-model benchmark** — the
planner prompt fingerprint (`benchmark/environment.py:106,129`) changes if a byte moves.

### 9.2 Utterance-regex guards in `planner.py:203-216`

```python
_SECOND_PERSON_IDENTITY = re.compile(r"(?is)^(?:who are you\b|what(?:'s| is) your (?:name|role)\b)|^(?:你|您)\s*(?:是\s*(?:谁|哪)|的名字|叫什么)")
_FIRST_PERSON_IDENTITY  = re.compile(r"(?is)^(?:who am i\b|what(?:'s| is) my name\b)|^我\s*(?:是\s*(?:谁|哪)|的名字|叫什么|的身份)")
_NAMED_OBJECT_LOCATION  = re.compile(r"(?is)^(?:where(?:'s| is) (?:the )?(?!my\b).+|(?!(?:我家|家里|咱家|我这个家)\s*).{1,24}在哪里)")
```

consumed by `_identity_person_mismatch` (`:347`) and `_object_location_mismatch` (`:326`), which
re-read the raw utterance **after** planning to decide whether to retry.

This is the clearest conflict with the stated invariant (`src/README.md:52-54`: *"The semantic
layers must not inspect utterance text after planning to repair a request"*; `AGENTS.md`: *"Do not
add question-specific routing, implicit semantic repair..."*).

**The honest read:** this is a **narrow, two-pattern** guard set, not a question catalog. And
`planner.py:243 _invalid_plan_retry_hint` is the *IR-based* mechanism, explicitly documented as
"never matches on utterance text" — so the module already contains the intended design and uses the
regex path only where the IR cannot distinguish. It is an exception that was taken deliberately.

**Recommendation: document it as a known, bounded exception in `src/README.md` and `AGENTS.md`,
with a comment at `planner.py:203` explaining why the IR path is insufficient here.** Do **not**
remove it in a cleanup pass — removal changes planner retry behavior and therefore real-model
accuracy, which this task must not touch.

### 9.3 `_failure_answer` synthesizes a request to produce a failure message

`semantic/facts.py` builds `SemanticFactRequest(operation="resolve_reference",
subject=SemanticReference(kind="self", entity_type="person"))` purely to render an error answer.
Functionally fine; it means a hard-coded request shape exists solely for messaging. Low priority.

### 9.4 `semantic/ir.py` imports the mutation IR

`semantic/ir.py` imports `NamedWriteRequest, NamedCreateItem, NamedMoveItem, NamedDeleteItem,
NamedUpdateAttributes` from `..mutation.ir`, and `SemanticPlan` carries both `request` and
`mutation` and `multi_intent`. This is the mechanism behind `.llm/2026-09-19_1450`'s note that
OpenRouter "still constructs the third-party Ollama `ChatResponse` model internally." It is a real
design decision (one plan type carries both branches) and **not** a duplication. Leave it.

### 9.5 Large inline example tuples in `unified_planner.py`

`_MULTI_INTENT_EXAMPLES`, `_MUTATION_MODE_EXAMPLES`, `_UNIFIED_READ_REPAIR_EXAMPLES` are
Python literals inside a 642-LOC module. This is the same shape as `_semantic_planner_examples()`
in `semantic/prompt.py`, which is `lru_cache`d and lives behind a function. **Recommendation:
move these to `semantic/prompt.py` beside their siblings, returning them from functions.**
Prompt byte content is unchanged by this move, so it is benchmark-neutral — but verify with
`tests/test_ollama.py:224` and `test_planner_prompt_audit.py:95` (the 36,000-byte ratchet).

---

## 10. API Findings

### 10.1 `api/routes/chat.py` mixes six concerns in one handler

`chat_completions` (`:80-172`) performs: agent resolution by display name, message
normalization/filtering, greeting resolution, a standalone-greeting short-circuit (`:125`),
`AnswerExecution` construction, and streaming-vs-collected response selection. The greeting policy
in particular is:
- `is_standalone_greeting` (`:174`) matching a hardcoded set `{"hello","hi","hey","你好","您好","嗨"}`
- `continued_greeting` (`:181`) with an inline bilingual string `"您好。有什么需要我处理的吗？"` /
  `"Hello. How may I help?"`, selected at `:129`

**Bilingual answer text belongs in the renderer, not in a route module.** `conversation/greetings.py`
(259 LOC) already owns greeting policy; these two helpers are policy that leaked into transport.

### 10.2 Two middlewares, one inline

`api/app.py:98-131` defines `request_observability` as an inline closure, then
`add_middleware(RequestTraceMiddleware, ...)` (`:133`). Both log request completion. Consolidating
the inline one into `common/tracing.py` (which already owns `RequestTraceMiddleware`) is a clean
win with no behavior change.

### 10.3 `api/providers.py:list_bare_models` builds HTTP discovery inline

60 LOC that constructs `httpx.AsyncClient(timeout=3.0)`, hits Ollama's `/api/tags`, sanitizes, and
branches per provider — inside a route-support module. It is the **only** place in `api/` that
speaks an HTTP endpoint other than its own. It belongs beside the adapter it is discovering
(`providers/ollama.py`), which would also remove the raw `httpx` dependency from `api/`.

### 10.4 Route module sizes are fine

`conversations.py` 252, `dependencies.py` 251, `chat.py` 182, `app.py` 156, `sse.py` 149,
`errors.py` 127, `sessions.py` 87, `system.py` 85, `execution.py` 70, `schemas.py` 65,
`providers.py` 60, `models.py` 43. **No module in `api/` needs splitting.** The structure
(`.llm/2026-09-19_1450`) is sound.

### 10.5 `AnswerExecution` is a genuine win — note it, don't change it

`.llm/2026-09-19_1450` records that it created "a single consumption point for streamed and
collected HTTP answers." `api/execution.py` (70 LOC) with `api/routes/chat.py:161,172` confirms it.
This is exactly the right shape for the streaming/collected duality and contrasts with §8.3, where
the same duality is *not* unified.

---

## 11. Configuration Findings

### 11.1 One provably dead variable

`HOME_CORTEX_DISABLE_TIER0` — see §6.1. Delete both lines.

### 11.2 No committed env template

There is no `.env.example`. `Readme.md` and `src/README.md:186-194` both document a five-key
minimum, but the full surface is ~24 keys across `config.py` and two compose files, and
`.env` is gitignored (`.gitignore:3`). A new operator has to read `config.py` to find
`CALENDAR_BINDINGS`, `RETRIEVAL_LIMIT`, `LOCAL_LLM_BASE_URL`, or the four `OPENROUTER_*` knobs.
**Recommendation: add a committed `.env.example` generated from `Settings` field names.** Do not
build a config framework — the `Settings` class is already correct and well-validated
(`config.py:118-160` enforces per-provider requirements).

### 11.3 `extra="ignore"` hides typos

`config.py:35`. This is why §11.1 was invisible. It is the right default for forward-compat with
compose env blocks that carry unrelated variables — but combined with no `.env.example`, a
misspelled variable fails silently. **Recommendation: keep `extra="ignore"`, add a test that every
key set in the compose files exists on `Settings`.** That catches §11.1 and every future instance
of it without a framework.

### 11.4 Compose duplication and drift

Both base compose files carry the same `proxy`/`home-gui`/`home-media`/`surrealdb` blocks.
`docker-compose.llamacpp.yml` is 124 lines vs. 100 — the delta is the `llama-server` service plus
the `LLM_PROVIDER`/`LOCAL_LLM_*` variables. The `gpu` overlays already demonstrate the right
pattern (3-line files). Rebase `docker-compose.llamacpp.yml` as an *overlay* on
`docker-compose.yml` (add `llama-server`, override `LLM_PROVIDER`, swap `depends_on`) and the two
files stop drifting. **Note `docker-compose.gpu.llamacpp.yml` currently only adds `gpus: all` to
`llama-server`, which presumes the base llamacpp file already defines it** — so the overlay chain
already exists in intent.

### 11.5 Auth is a no-op when unconfigured — an existing, tracked risk

`docker-compose.yml:84-85` defaults `CORTEX_API_KEY` and `CORTEX_IDENTITY_MAP` to empty; the proxy
publishes port 80 (`docker-compose.yml:14`). With both empty, `authenticate_request` has nothing to
enforce. `.llm/2026-09-24_2131_repository-review.md` already recorded this, along with:
- request headers `X-OpenWebUI-User-Id`/`-Email` preferred over the signed GUI cookie while
  `nginx.conf` does not overwrite them (identity spoofing);
- `/admin/ingest` and `/admin/export` accepting a GUI cookie rather than the bearer key, with
  export accepting any absolute path.

**This is not a cleanup item and this audit did not re-verify it beyond confirming the compose
defaults are still empty.** It is listed here because Section 15's ranked table must not bury it.
It should be handled as its own change with its own review.

---

## 12. Test / Benchmark Findings

### 12.1 No `conftest.py` — the root cause of most test duplication

`tests/conftest.py` does not exist. Neither does `tests/__init__.py`. There are **6 fixtures for
18,286 LOC** (`api_client`, `isolated_registry`, `household`, `dispatcher`, `service`, `context`).
Reuse is achieved by importing test modules, which drags module-level side effects:
`test_semantic_composition.py:521` (`from test_api import api_client`, mid-file),
`test_calendar.py:21-28` (reaches into private `_agent`, `_chat_response`, `_tool_call`),
`test_open_webui_conversations.py:11`, `test_entity_alias_resolution.py:17`, and five files
importing `household` from `test_semantic_contract.py:42`.

**Consequence:** `test_semantic_contract.py` (844 LOC) is a hidden dependency for 5 files — editing
its module scope breaks collection elsewhere. And `test_api.py:810 _protect_api()` /
`:824 _seed_people()` mutate global `app.state`, so importers inherit that.

**Recommendation: add `tests/conftest.py` holding `MemoryDatabase`, the `STATIC_TEST_DATA` path,
an `AgentRequestContext` builder, and the fake-provider base classes.** This is the highest-value
single change in the test tree, and it is purely additive — no test logic moves.

### 12.2 No structure and no markers

56 files, flat. `grep` for `pytest.mark` in `tests/` finds only `asyncio` and `parametrize` — zero
custom markers.
`pyproject.toml` `[tool.pytest.ini_options]` has only `pythonpath` and `testpaths`. So the
17-subprocess `test_engineering_entrypoints.py` and the 4-hash-seed `test_semantic_transport.py`
cannot be excluded from a fast loop.

**Recommendation: add markers (`architecture`, `semantic`, `benchmark`, `slow`) and a
`--strict-markers` config; do not move files.** Moving 56 files to `unit/`+`integration/` is a large
diff that changes nothing about what runs. Markers get the same benefit for a fraction of the risk.
If subdirectories are wanted later, do it per-cluster (start with `tests/semantic/`).

### 12.3 Tests that pin one-off experiment numbers

| Test | What it freezes | Breaks on |
|---|---|---|
| `test_prefix_reuse_probe.py:25` (`identical_leading_messages > 40`) | current prompt/example count | **removing** a planner example |
| `test_ollama.py:250` (`<= (NUM_CTX - NUM_PREDICT) * 3`) | a chars-per-token multiplier from one 8192-window incident | example-set growth |
| `test_planner_prompt_audit.py:105` (`<= 36000` bytes) | deliberate ratchet ("offline creep alarm") | any prompt growth |
| `test_planner_prompt_audit.py:114-117` | `len(cases) == 12` + **positional** `cases[8]` ("Complete father-in-law expansion"), `cases[-1]` | appending or reordering a YAML case |
| `test_unified_planner_experiment.py:245` | `prompt_eval_count == 11269` | any prompt change |
| `test_unified_planner_experiment.py:249,279-280` | `{"regressions": 1, "improvements": 0}` fed to the shadow summarizer | any change to the summarizer |
| `test_semantic_planner_benchmark.py:60,72` | two utterance sets asserted via `issubset` against the corpus | rewording corpus YAML |
| `test_fact_benchmark.py:39-63` | 7 `QUESTIONS` with **exact-string dispatch** | corpus drift → silently exercises the fallback branch instead of failing |
| `test_composition_eval.py:55,66-72` | `SCORING_REVISION` pinned; `42`/`36`/`8`/`8`/`78` | corpus resize |

`test_fact_benchmark.py` is the worst of these: because the fake planner dispatches on
`if text == "家里有几个人"` with an `else` fallback, adding or rewording a `QUESTIONS` entry makes
the test pass while testing nothing.

**Recommendation: replace hardcoded literals with reads from the corpus** (the file already imports
the corpus loader). Keep the byte-budget ratchets — they are intentional — but make their failure
message say "prompt grew; re-baseline deliberately."

### 12.4 Three architecture tests are brittle beyond their intent

- `test_backend_architecture.py:125` — forbids the *attribute name* `_dispatch` anywhere in
  `runtime/agent.py` via AST. A legitimate internal helper named `_dispatch` fails it.
- `test_backend_architecture.py:140` — `len(api_lines) < 100`, an arbitrary line count.
- `test_media_boundaries.py:27-41` — asserts literal Svelte fragments including
  `"returned the web app instead of JSON"` and `"15_000"`. **This test is what currently prevents
  de-duplicating the GUI's two fetch wrappers** (§5.6) — hoisting a timeout into a named constant
  breaks a "boundary" test while the boundary is intact.
- `test_deployment_network.py:15` — asserts `services["home-media"]["volumes"] == [...]` with three
  hardcoded host paths; adding a legitimate volume breaks it.

The *intent* of each is correct (no `_dispatch` coupling, thin API initializer, media isolation,
only-nginx-is-LAN-facing). The *mechanism* is a text snapshot. Recommend softening each to the
property it means (e.g. assert media and cortex packages share no import, rather than that a
specific string appears in `media.ts`).

### 12.5 Benchmark corpora have no single source of truth

11+ independent case definitions:

| Source | Size | Consumer |
|---|---|---|
| `benchmarks/semantic_planner_eval.yaml` | 119 utterances | `load_semantic_eval_cases`, `load_probe_dataset` |
| `semantic_planner_{heldout,synthetic,age_filters,date_intervals}.yaml` | 12/12/24/16 | `load_probe_dataset(path)` |
| `semantic_planner_bilingual.yaml` | 47 | `load_bilingual_dataset()` |
| `planner_prompt_compression.yaml` | 12 | `planner_prompt_experiment.fixed_cases()` |
| `mutation_routing.yaml` | 9 read / 14 write / … | `unified_planner_experiment.py:218` |
| `composition/{development,frozen}/*.yaml` | 42+8 / 36+8 | `composition_eval` |
| `fact_benchmark.py:35-82` + `:84-95` | **33 `QUESTIONS` + 5 `SPEAKER_CASES` as Python literals** | `FactSuite` |

Known overlaps: `mutation_routing.yaml`'s 9 read utterances share **6** with
`semantic_planner_eval.yaml` (acknowledged in `hc_suites.py:40-43`'s `_READ_POLICY` prose);
`planner_prompt_compression.yaml` shares 2 more. `SCORING_REVISION` is hardcoded in 3 places,
`FROZEN_EVAL_TIME` in 4+, the default eval path in 3.

**`benchmarks/composition/frozen/MANIFEST.yaml` is the sharpest instance:** it re-lists all 36
frozen cases + 8 sequences, is **absent from the `files` hash map in `fingerprints.json`**, and the
only test touching it (`test_composition_eval.py:408,413`) asserts non-emptiness, never agreement.

**Recommendation: (a) move `fact_benchmark.QUESTIONS`/`SPEAKER_CASES` into a YAML for consistency;
(b) add `MANIFEST.yaml` to the fingerprint map and assert it matches the YAML it duplicates.**
Do **not** attempt to unify all corpora — they test genuinely different things and the eval/probe
split is deliberate.

### 12.6 `src/home_media/tests/` is orphaned from the root run

Root `pyproject.toml` sets `testpaths = ["tests"]` and `pythonpath = ["src", "."]`;
`src/home_media/pyproject.toml` sets `pythonpath = ["src"]`, `testpaths = ["tests"]`. Since
`src/home_media/src` is not on the root path, the root suite **cannot** import `home_media` — so the
5 media test files (325 LOC — 4 test modules + a 78-LOC `conftest.py`) run only via the manual
`uv run --project src/home_media --extra dev
pytest` in `home_media/README.md:44-50`. **There is no CI in the repository** (no `.github/`), so
nothing runs them automatically. That is a real coverage gap for a package that ships in production.

### 12.7 `src/home_cortex/benchmark/` has zero production importers — confirmed

A repo-wide grep for `home_cortex.benchmark` outside the package returns **no matches**. The only
occurrence of "benchmark" in production code is a comment (`providers/ollama.py:29`). This is
correct by design (CLI-only harness) and should be **stated** in `src/README.md`, which currently
does not list `benchmark/` in its package map despite it being 2,502 LOC / 14 modules.

---

## 13. Frontend Findings

`src/home_gui` — Svelte 5 + Vite + TS, 1,554 LOC under `src/`. **No vision/spatial references.**
`dist/` and `node_modules/` are untracked (`git ls-files | grep -E "(dist|node_modules)/"` →
empty). `serve.json` is genuinely used by `Dockerfile:12,14`, not vestigial.

### 13.1 Duplicated fetch/error handling (the main refactor candidate)

`lib/api.ts:3-22` (`request<T>()`) and `lib/media.ts:38-93` (`mediaRequest()`), plus
`api.ts:78-86` re-implementing the same error parse inside `streamMessage`. Two **incompatible**
error types: `ApiError` class (`types.ts:39-47`, consumed at `App.svelte:59,106,108`) vs. an ad-hoc
`Error & {status?, code?}` (`media.ts:32-36`, consumed via casts at `Media.svelte:39,59`).
`credentials: 'include'` appears three times.

**Recommendation: unify on one `ApiError` and one wrapper.** Note this requires editing
`tests/test_media_boundaries.py:38-40`, which currently pins the string `"15_000"` — do both in one
commit.

### 13.2 `App.svelte` (275 LOC) owns six concerns

Auth/session (`:53-64,99-114,232-238`), routing (`:33,46-51,84-88,92`), conversation CRUD
(`:66-185`), SSE consumption with rAF batching (`:196-230`), i18n (`:21,35`), model selection
(`:178-185`). `Media.svelte` (212 LOC) mixes data/pagination, month grouping, lightbox, and
keyboard handling. Children (`Chat`, `Composer`, `Login`, `Message`, `Sidebar`) are correctly dumb
props-only components.

**Recommendation: extract the conversation-lifecycle block from `App.svelte` into a
`lib/conversations.ts` store.** That is the one split with a clear seam. Do not restructure the
component tree — it is already the right shape.

### 13.3 `app.css` is one 438-LOC unscoped global sheet

No component has a `<style>` block. Concrete smells: the card recipe is restated 3×
(`:252-259`, `:222-233`, `:372-381`); `.error` is self-overridden (`:264-265`); `.icon-btn` in the
shared rule at `:107` is dead (no component uses it); `.hidden` (`:267`) exists solely for
`Sidebar.svelte:42`. **Recommendation: three CSS custom properties for the card recipe, delete the
dead selectors.** No framework, no rebuild.

### 13.4 i18n stops at the media page

`lib/i18n.ts:3-48` is a typed `en`/`zh` table with `t()`. Used by `Chat`, `Composer`, `Login`,
`Message`, `Sidebar`, `App` — but **`Media.svelte` is entirely hardcoded English**
(`:116-121,125-134,139,144,148,182,192-193,210`, plus `'Unknown date'` and locale-less
`Intl.DateTimeFormat(undefined, …)` at `:89-91`). Also hardcoded: `Sidebar.svelte:39-40`,
`App.svelte:221`. **Recommendation: add the ~12 missing keys.** Small, contained, and it is the
GUI's stated purpose to be bilingual.

### 13.5 `HOME-CORTEX.md` is stale

`src/home_gui/HOME-CORTEX.md:6-8` lists `/session`, `/conversations`, `/v1` and omits `/media-api`,
which `lib/media.ts:47` requires and `nginx.conf:73-89` auth-gates. **Update the doc.**

---

## 14. home-media Findings

`src/home_media` — 1,227 LOC (902 package + 325 tests). **Independence holds in both directions:**
`grep -rn home_media src/home_cortex` → none; media's `uv.lock` has `grep -c 'home-cortex'` = 0.
Root `pyproject.toml` never installs or collects it. **This boundary is intact and must stay so.**

### 14.1 `contracts.py`, `content.py`, `config.py` are load-bearing, not pass-throughs

- `contracts.py` — the FastAPI `response_model`s (`app.py:73,95`) and the scanner's declared return
  type (`scanner.py:12`). Its cost: the same 16-field row shape is stated **four** times
  (SQLite DDL `index.py:14-32`, `upsert` kwargs `:62-73` + tuple `:101-118`, the Pydantic model
  `contracts.py:14-31`, and `_public_item` remapping `app.py:159-179`) with no single source.
- `content.py` (32 LOC) — encodes the documented path invariant in one place, two consumers
  (`app.py:122`, `thumbnails.py:41`), directly unit-tested.
- `config.py` (47 LOC) — validated frozen dataclass, 4 consumers.

**Nothing here is over-abstraction.** The only single-expression abstractions in the package are
`_index`/`_scanner`/`_thumbnails` at `app.py:138-147`.

### 14.2 Real duplication (small but concrete)

Path containment in `content.py:21-30` **and** `scanner.py:61-74` (two implementations, two failure
modes: raise vs. counter). Thumbnail cache path in 3 places. Media-ID format defined twice with
nothing tying generator to validator. **Recommendation: add `ThumbnailService.path_for()` and have
`scanner.py:105,113` use it; have `scanner.py` reuse `content.py`'s containment helper for its
symlink check.** ~15 LOC, no behavior change.

### 14.3 No CI runs the media suite

See §12.6. Given `home_media` is a deployed service with byte-range, traversal-guard, and
symlink-escape tests, this should be closed by adding `src/home_media` to a test job.

### 14.4 One-way build context leak (inert)

Root `Dockerfile:9` does `COPY src ./src`, and the cortex build context is `..`
(`docker-compose.yml:62`) — so `src/home_media/**` source ships inside the cortex-api image. It is
**not installed** (`pip install .` honors the packages list). Harmless, but adding
`src/home_media` to `.dockerignore` is free.

---

## 15. Ranked Cleanup Opportunities

Priority key — **P0**: correctness/security or actively misleading; **P1**: high-value structural
simplification with contained risk; **P2**: mechanical hygiene; **P3**: nice-to-have.

| Priority | Area | Problem | Evidence | Proposed Action | Risk | Expected Benefit |
|---|---|---|---|---|---|---|
| **P0** | `api/` auth | Auth is a no-op when `CORTEX_API_KEY`/`CORTEX_IDENTITY_MAP` are empty (compose defaults both empty, port 80 published); spoofable identity headers; `/admin/export` takes any absolute path | `docker-compose.yml:14,84-85`; `.llm/2026-09-24_2131_repository-review.md` | Separate change, separately reviewed. Fail closed when unconfigured, or make the insecure mode explicit | **High** — changes access for every client | Closes a known LAN-exposed hole |
| **P0** | `providers/` | "Provider-neutral" contract embeds three planner methods; both adapters import `semantic.prompt` + `mutation.ir` | `providers/base.py:28-67`; `ollama.py:8-16`; `openai_compatible.py` | Move `plan_*` off the Protocol into a `semantic/planner.py`-side planner that owns prompt+schema and calls `provider.chat_structured(...)` | **High** — touches planner I/O; needs real-model gate | Removes the inversion; unlocks P1 below |
| **P1** | `providers/` | Entire planner method surface duplicated; only decoding differs | `ollama.py` vs `openai_compatible.py` — 8 methods each | After P0: make `OllamaService` a thin wrapper over one shared structured-call path | Medium | ~150 LOC dedup; one place to change planner call shape |
| **P1** | `tests/` | No `conftest.py`; 6 fixtures for 18.5k LOC; `MemoryDatabase` in 7 files; 4 fake-Ollamas; cross-test imports | §12.1 | Add `tests/conftest.py` with the shared fakes and path constants | **Low** — additive | Removes the largest duplication; stops `test_semantic_contract.py` being a collection dependency |
| **P1** | `tests/` | No markers; 17-subprocess test runs beside 9-LOC test | `pyproject.toml [tool.pytest.ini_options]`; §12.2 | Add `architecture`/`semantic`/`benchmark`/`slow` markers + `--strict-markers` | **Low** | "Run the fast tests" becomes expressible |
| **P1** | `scripts/` | Superseded CLI + shell script with a nonexistent default path | `tier1_latency_bench.py` (153); `copy_tier1_bench_into_api.sh` (59) | Delete both; `hc_suites.py:468-489` already covers the former | Low | 212 LOC deleted |
| **P1** | `semantic/` | `unified_planner` patches `planner`'s prompt by index arithmetic + 5 private imports | `unified_planner.py:21-27,372,456` | Give `planner.py` an explicit message-builder parameter for the opening/reminder slots | **Medium-High** — prompt bytes must not move | Removes positional coupling; makes prefix compatibility structural |
| **P2** | `providers/` | 4 unused imports | `ollama.py:12-16` | Delete the 4 names | **None** | Trivially correct |
| **P2** | `providers/` | Two runtime-metric converters | `ollama.py:239-259` vs `openai_compatible._runtime_metrics` | One helper in `providers/ir.py` | Low | ~15 LOC |
| **P2** | `docker/` | Two dead env lines; two compose files duplicating 4 service blocks | `docker-compose.yml:83`, `.llamacpp.yml:106`; §11.4 | Delete the var; rebase llamacpp as an overlay on the base (the `gpu` files already are) | Low | Stops compose drift |
| **P2** | config | No committed env template | §11.2 | Add `.env.example` + a test that every compose env key exists on `Settings` | Low | Catches §11.1-class bugs forever |
| **P2** | `api/` | Bilingual greeting strings + greeting policy in a route module | `api/routes/chat.py:125,174-182` | Move to `conversation/greetings.py` | Low-Medium | Transport stops owning answer text |
| **P2** | `api/` | Inline middleware duplicates `RequestTraceMiddleware` | `api/app.py:98-131` | Fold into `common/tracing.py` | Low | One logging path |
| **P2** | `api/` | Ollama `/api/tags` discovery inside `api/` | `api/providers.py:list_bare_models` | Move beside `providers/ollama.py` | Low | Removes a raw `httpx` use from `api/` |
| **P2** | `benchmarks/` | `MANIFEST.yaml` duplicates frozen cases, is unhashed, untested | `fingerprints.json` `files` map; `test_composition_eval.py:408` | Add to the fingerprint map + assert agreement | Low | Removes a silent-drift path |
| **P2** | `benchmarks/` | Corpus as a Python literal with exact-string dispatch tests | `fact_benchmark.py:35-95`; `test_fact_benchmark.py:39-63` | Move to YAML; make tests read the corpus | Low | Tests stop passing vacuously |
| **P2** | `tests/` | Brittle text-snapshot assertions | §12.4 | Soften to the property each means | Low | Unblocks §13.1 |
| **P2** | `home_media` | Containment + cache-path duplication; ID format defined twice | `content.py:21-30`/`scanner.py:61-74`; `thumbnails.py:30`/`scanner.py:105,113`; `scanner.py:25-27`/`app.py:22` | Add `path_for()`; share the containment helper | Low | ~15 LOC; one owner per rule |
| **P3** | `home_gui` | Two fetch wrappers, two error types | `lib/api.ts:3-22` / `lib/media.ts:38-93` | Unify on `ApiError` | Low-Med | ~40 LOC |
| **P3** | `home_gui` | `Media.svelte` (212) + `App.svelte` (275) mixed concerns | `App.svelte:53-238` | Extract conversation lifecycle to `lib/conversations.ts` | Low-Med | Testable seam |
| **P3** | `home_gui` | Dead CSS selectors, triplicated card recipe, 12 missing i18n keys | `app.css:107,264-265,267`; `Media.svelte:116-210` | Delete dead selectors; add keys | **None** | Consistency |
| **P3** | `artifacts/` | Orphan script + 5 tracked archives + 2 empty dirs | `artifacts/analysis/build_analysis.py`; §6.5 | Move the script to `scripts/`; delete empty dirs | None | Generated tree holds only generated output |
| **P3** | docs | `benchmark/` (2,502 LOC) absent from `src/README.md` package map; GUI doc omits `/media-api` | `src/README.md` package map; `HOME-CORTEX.md:6-8` | Update both | None | Docs match the tree |
| **P3** | `spatial`/`vision` | Planner choice documented only as a tool-list side effect | `runtime/agent.py`; §8.5 | Document the axis in `src/README.md` | None | Removes a hidden coupling from the docs' blind spot |
| **P3** | `runtime/` | Ollama wording in a provider-agnostic module | `model_loop.py` docstring + 2 error strings | Rename | None | Cosmetic honesty |
| **P3** | `runtime/` | `run()` and `stream()` are parallel loop implementations | `model_loop.py` (648 LOC) | Consider unifying after P0/P1 | Med-High | Only if P0/P1 settle the seam |

---

## 16. Safe Quick Wins

Ordered by (value ÷ risk). All are additions or deletions with no semantic effect and no
prompt-byte movement.

1. **Delete `providers/ollama.py:12-16`'s four unused imports.** Zero risk; a linter would do it.
2. **Delete the 7 candidate scripts** where no report exists (§6.2) **and** update
   `test_engineering_entrypoints.py:21-38` in the same commit — otherwise the suite fails on a
   missing module.
3. **Delete `scripts/benchmarks/tier1_latency_bench.py`** and
   **`scripts/maintenance/copy_tier1_bench_into_api.sh`** (212 LOC).
4. **Delete the `HOME_CORTEX_DISABLE_TIER0` lines** from both compose files.
5. **Delete the empty `src/home_cortex/http/` directory tree.**
6. **Add `tests/conftest.py`** with `MemoryDatabase`, `STATIC_TEST_DATA`, an
   `AgentRequestContext` builder, and the fake-provider bases; repoint the 7 duplicating files.
   Purely additive.
7. **Add pytest markers + `--strict-markers`.** No file moves.
8. **Add `.env.example`** generated from `Settings` field names.
9. **Add `src/home_media` to the root `.dockerignore`.**
10. **Add the 12 missing `Media.svelte` i18n keys** and delete the dead CSS selectors.
11. **Update `src/home_gui/HOME-CORTEX.md`** to mention `/media-api`.
12. **Add `benchmark/` to the `src/README.md` package map**, and note it is CLI-only.

Items 1–5 are ~250 LOC of pure deletion; 6–7 are large dedup wins; 8–12 are documentation and
consistency.

---

## 17. High-Risk Areas — do not casually refactor

Each of these has a written invariant behind it. Read the cited `.llm/` entry and the guard test
before touching it.

| Area | Why it is dangerous | Guard / evidence |
|---|---|---|
| **The planner prompt, in any way** | Ollama's prompt cache depends on byte-stable prefixes; `num_ctx` truncation already caused a production incident | `tests/test_ollama.py:224,360`; `test_planner_prompt_audit.py:95` (36k ratchet); `providers/ollama.py:19-26` comment |
| **`semantic/planner.py`'s regex retry guards** | Removing them changes planner retry behavior → changes real-model accuracy. This task must not | `planner.py:203-216,326,347`; `src/README.md:52-54` |
| **`facts/engine.py` (828 LOC)** | The deterministic executor; every split risks the "model never computes" invariant | `AGENTS.md` invariants; `test_semantic_architecture.py:34` |
| **`facts/resolver.py` + containment/kinship semantics** | Speaker-relative and containment bugs live here | `AGENTS.md` "Fast ownership guide" names the exact tests |
| **`semantic/schema.py` ↔ `schemas/*.yaml`** | `property_source ∈ {entity, relationship}` must never be silently flipped | `AGENTS.md`; `test_planner_prompt_audit.py:58` |
| **`mutation/writing.py` transactions** | Fixed SurrealQL, no caller-supplied queries — that is the security property | `mutation/writing.py:697`; `AGENTS.md` |
| **Planner selection via `write_item` membership** | Changing it can flip an agent into the unified planner and re-open the multi-intent safety gate | `runtime/agent.py`; `artifacts/llamacpp-migration/REPORT.md` |
| **`api/dependencies.py` auth** | Identity spoofing is a live, tracked concern | `.llm/2026-09-24_2131_repository-review.md` |
| **`benchmarks/composition/fingerprints.json`** | Approval artifact; regenerating it is a reviewed act, not a build step | `.llm/2026-09-24_2136_composition-approval-fingerprint.md` |
| **`spatial/__init__.py` emptiness** | Load-bearing for chat startup cost; the only SCC break | `spatial/__init__.py:1-7`; `test_backend_architecture.py:72` |
| **`providers/llamacpp.py` / `docker-compose.llamacpp.yml`** | The migration was **rejected** on a multi-intent regression; do not "finish" it | `artifacts/llamacpp-migration/REPORT.md`; `.llm/2026-09-24_2228_llamacpp-migration-eval.md:35-39` |
| **Deleting `scripts/probes/*ollama*`** | Ollama remains production; these are not obsolete | Same as above |

---

## 18. Proposed Target Tree

**Design constraint honored: every box below is reachable incrementally from the current tree.**
Labels: `[MOVE]` path change only · `[MERGE]` collapse duplicates · `[SPLIT]` divide one module ·
`[DELETE]` remove · `[ADD]` new file · `[KEEP]` unchanged.

```
home-cortex/
├── src/
│   ├── home_cortex/                       [KEEP] package layout — it is already correct
│   │   ├── api/
│   │   │   ├── __init__.py                [KEEP] trim to app/create_app/lifespan/VIRTUAL_MODEL/ConversationStore
│   │   │   ├── app.py                     [MERGE] fold request_observability into common/tracing.py
│   │   │   ├── providers.py               [MOVE]→providers/ollama.py (model discovery)
│   │   │   └── routes/chat.py             [SPLIT] greeting policy → conversation/greetings.py
│   │   ├── benchmark/                     [KEEP] CLI-only harness (document it in src/README.md)
│   │   │   ├── environment.py             [KEEP] provenance (or MERGE with the script-side copies)
│   │   │   └── plugins.py                 [KEEP] scripts import stays, documented as the exception
│   │   ├── capabilities/{calendar}.py     [SPLIT] (optional) split OAuth/token/normalization/authorization
│   │   ├── providers/
│   │   │   ├── base.py                    [SPLIT] drop plan_* from the Protocol
│   │   │   ├── ollama.py                  [MERGE] onto the shared structured-call path
│   │   │   ├── openai_compatible.py       [KEEP] the one transport
│   │   │   ├── openrouter.py              [KEEP] 21 LOC
│   │   │   ├── llamacpp.py                [KEEP] 7 LOC, live via the factory
│   │   │   └── ir.py                      [ADD]  metrics helper next to the existing types
│   │   ├── runtime/model_loop.py          [SPLIT] (defer) unify run()/stream()
│   │   ├── semantic/
│   │   │   ├── planner.py                 [ADD]  explicit message-builder params; [ADD] comment on the regex guards
│   │   │   ├── unified_planner.py         [MOVE] inline example tuples → prompt.py
│   │   │   └── prompt.py                  [KEEP]  the single prompt owner
│   │   ├── spatial/                       [KEEP]  contracts production; localization tests-only (intentional)
│   │   ├── vision/                        [KEEP]  paused contracts (intentional)
│   │   └── http/                          [DELETE] empty leftover tree
│   ├── home_gui/
│   │   └── src/
│   │       ├── lib/api.ts                 [MERGE] one wrapper + one ApiError (absorbs media.ts's)
│   │       ├── lib/media.ts               [MERGE] keep types + isMediaPage guard; drop the duplicate wrapper
│   │       ├── lib/conversations.ts       [ADD]   lifecycle extracted from App.svelte
│   │       ├── components/Media.svelte    [ADD]   i18n keys
│   │       └── app.css                    [KEEP]  delete dead selectors; token for the card recipe
│   └── home_media/                        [KEEP]  independent; add path_for() + share containment
├── tests/
│   ├── conftest.py                        [ADD]   MemoryDatabase, STATIC_TEST_DATA, context builder, fakes
│   ├── markers                              [ADD]   architecture/semantic/benchmark/slow + --strict-markers
│   └── *.py                               [KEEP]  no file moves; soften the text-snapshot assertions
├── scripts/
│   ├── maintenance/copy_tier1_bench_into_api.sh    [DELETE]
│   ├── benchmarks/tier1_latency_bench.py           [DELETE] superseded by the latency suite
│   ├── probes/{item_location,kinship_context}.py   [DELETE] only the --help sweep references them
│   ├── profiling/{http_latency_audit,token_component_probe,
│   │              profile_semantic_transport}.py   [DELETE] same
│   ├── maintenance/context_surface_audit.py        [DELETE] same
│   ├── probes/layer_failure_trace.py               [KEEP]   reproduces a published report
│   └── README.md                                   [KEEP]   update the deletion list
├── benchmarks/
│   ├── composition/frozen/MANIFEST.yaml            [KEEP] + [ADD] to fingerprints.json files map
│   ├── fact_questions.yaml                         [ADD]  move fact_benchmark's 33+5 literals here
│   └── tier1_latency.yaml                          [ADD or DELETE] add it, or drop the reference
├── artifacts/
│   ├── analysis/build_analysis.py                  [MOVE]→scripts/maintenance/
│   ├── context-surface-cleanup/  vision-architecture-review/   [DELETE] empty
│   └── *.tar.gz                                    [KEEP] verify each is still needed
├── docker/
│   ├── docker-compose.yml                          [KEEP] base
│   ├── docker-compose.llamacpp.yml                 [SPLIT] rebase as an overlay of the base
│   └── docker-compose{,.llamacpp}.gpu.yml          [KEEP] already correct overlays
├── .env.example                            [ADD]
└── src/README.md                           [KEEP] + note benchmark/, the planner-selection axis, the regex exception
```

---

## 19. Suggested Migration Sequence

Five stages. **Stages 1–2 are safe and independent. Stage 3 is mechanical. Stage 4 needs a
real-model gate. Stage 5 is optional.** No stage requires a big-bang cutover.

### Stage 1 — Hygiene (no behavior change, no prompt change)

Deletions and additions only: quick wins 1–5, 8–12 from Section 16, plus the `.env.example`, the
markers, and the `tests/conftest.py` extraction (win 6). **Gate: `python -m pytest -q` → 983 pass
(or 983 minus the deleted `--help` entries).** No benchmark needed: nothing here can change model
behavior — verify that claim by confirming no file under `semantic/`, `providers/`, `facts/`, or
`mutation/` is touched except the four unused imports.

### Stage 2 — Test and config consolidation

Marker adoption across the existing files; `MemoryDatabase`/fake-provider dedup completed;
`benchmarks/composition/frozen/MANIFEST.yaml` added to `fingerprints.json` and its agreement
asserted; `fact_benchmark.QUESTIONS` moved to YAML with the tests reading the corpus; the brittle
text-snapshot assertions softened. **Gate: 983 pass, and `test_fact_benchmark` must fail if a
corpus entry is removed** (that is the point of the change — prove it by temporarily removing one).

### Stage 3 — Engineering-tree reduction

Delete the six `--help`-only scripts and the two clearly-superseded ones; move
`artifacts/analysis/build_analysis.py` into `scripts/maintenance/`; delete the empty artifact
directories; update `scripts/README.md` and `test_engineering_entrypoints.py`. **Gate: 983 pass.**

### Stage 4 — Provider seam (requires the real-model benchmark)

The only stage that can change model behavior, so it must run on the production GPU host from an
isolated fingerprinted package (`scripts/README.md`; `benchmarks/HARNESS.md`).

1. **Step 4a** — delete the four unused imports; unify the runtime-metric helper; rename the Ollama
   wording in `runtime/` and `model_loop.py`. *No benchmark needed — no prompt byte moves; verify
   with `test_ollama.py:360` (prefix stability).*
2. **Step 4b** — give `semantic/planner.py` explicit opening/reminder parameters and drop
   `unified_planner.py`'s positional arithmetic and string replacement. **Gate: the shared prefix
   must be byte-identical.** Verify offline with
   `tests/test_ollama.py:360` (`test_static_prefix_survives_clock_history_and_identity_notes`), and
   confirm the `planner_prompt_audit` byte budget is unchanged. Then run `hc-bench --regression` to
   confirm plan/multi-intent scores are unchanged.
3. **Step 4c** — move `plan_*` off the `ModelProvider` Protocol into a planner-side object that
   owns the prompt and schema and calls a single `chat_structured(...)`. **Gate: `hc-bench
   --regression` matches the recorded baseline.** This is the largest semantic-adjacent change in
   the plan and should be its own reviewed cycle.
4. **Step 4d** — collapse `ollama.py`'s planner methods onto the shared path. **Gate: same.**
5. **Step 4e** — *only if 4c settled the seam*, unify `model_loop.run()`/`stream()`. **Gate: same,
   plus the tool-budget tests.**

If Step 4c cannot show parity, **stop there and document why.** The duplication is a cost, not a
bug; the planner is the correctness core.

### Stage 5 — UX and documentation debt (optional, independent)

GUI fetch/error unification (edit `test_media_boundaries.py:38-40` in the same commit);
`lib/conversations.ts` extraction; `Media.svelte` i18n; dead CSS; the two README updates; the
`benchmarks/tier1_latency.yaml` decision.

---

## 20. Estimated Simplification

**Read this table carefully: deleted, de-duplicated, and moved LOC are three different things and
are reported separately.** Moved code is *not* a reduction in code.

### 20a. Deletion (real reduction)

| Item | LOC |
|---|---:|
| `scripts/maintenance/copy_tier1_bench_into_api.sh` | 59 |
| `scripts/benchmarks/tier1_latency_bench.py` | 153 |
| `scripts/probes/{item_location_probe,kinship_context_probe}.py` | 262 |
| `scripts/profiling/{http_latency_audit,token_component_probe,profile_semantic_transport}.py` | 290 |
| `scripts/maintenance/context_surface_audit.py` | 84 |
| `providers/ollama.py` unused imports | 5 |
| `docker-compose{,.llamacpp}.yml` dead env var | 2 |
| `api/__init__.py` unused exports (12 of 16) | ~20 |
| `app.css` dead selectors + self-override | ~8 |
| `artifacts/` empty dirs + orphans | ~0 |
| **Subtotal (Stage 1 + 3)** | **≈ 880** |
| Stage 4b–4d: resolver/method collapse in `providers/` | ~150–250 |
| **Total realistic deletion** | **≈ 1,030–1,130** |
| **Optimistic** (Stage 4e plus `api/providers.py` relocation plus greeting helpers) | **≈ 1,300–1,900** |

### 20b. De-duplication (LOC stays; copies drop)

| Item | Copies today | After |
|---|---:|---:|
| `MemoryDatabase` in tests | 7 | 1 |
| Fake-Ollama implementations | 4 | 1–2 |
| `STATIC_TEST_DATA` constant | 7 | 1 |
| `AgentRequestContext` inline builds | 17 | ~4 (named builders) |
| Provider planner method surface | 2 | 1 |
| Runtime-metric converters | 2 | 1 |
| Provenance/hashing stacks | 2 | 1–2 |
| Percentile/latency summarizers | 2 | 1 |
| GUI fetch wrappers | 2 (+1 inner) | 1 |
| `error instanceof Error ? …` normalization | 5 | 1 |
| `home_media` containment check | 2 | 1 |
| `thumbnail_root/{id}.jpg` | 3 | 1 |
| Compose service blocks | 2 × 4 blocks | 1 × 4 |
| **Estimated duplicated LOC removed from the codebase** | | **≈ 700–900** (each copy removed, but one instance remains — so this is *copy* reduction, not net deletion) |

### 20c. Moves (zero reduction — listed for honesty)

`semantic/unified_planner.py` example tuples → `semantic/prompt.py` (~120 LOC); greeting policy
`api/routes/chat.py` → `conversation/greetings.py` (~15); model discovery `api/providers.py` →
`providers/ollama.py` (60); `artifacts/analysis/build_analysis.py` → `scripts/maintenance/` (~900);
inline middleware → `common/tracing.py` (~35). **Total ≈ 1,130 LOC moved, 0 LOC saved.**

### 20d. Additions (honest accounting)

`tests/conftest.py` (~120 LOC — replaces ~300 LOC of duplication); `.env.example` (~25); test
markers (~10); `path_for()`/containment helper in `home_media` (~15); `lib/conversations.ts`
(~60); `benchmarks/fact_questions.yaml` (~120, replacing an equivalent Python literal).
**≈ 350 LOC added.**

### 20e. Net

| Measure | Estimate |
|---|---:|
| Deleted | ≈ 1,030–1,130 |
| Deduplicated (copies removed, one instance kept) | ≈ 700–900 |
| Added | ≈ 350 |
| Moved (no reduction) | ≈ 1,130 |
| **Net source LOC reduction (deleted + dedup − added)** | **≈ 1,380–1,680** |
| **As a share of human-authored Python (21,626 LOC in `src/home_cortex`)** | **≈ 6–8%** |

**Do not read 6–8% as disappointing.** The earlier reorganization already captured the large wins —
43 root modules → 1, facades → 0, 19,023 raw LOC at the end of
`.llm/2026-09-19_2210_functional-package-reorganization.md`. What remains is not bulk; it is
locality: the planner seam, the test fixture layer, and the engineering tail. Those are worth more
than their line count.

**What this audit deliberately did not recommend, and why:**

- **Splitting `facts/engine.py` (828), `semantic/schema.py` (914), or `facts/operators.py` (638)** —
  they are cohesive; the invariant being protected (deterministic execution, explicit property
  ownership) is harder to see across more files.
- **Deleting `vision/` or `spatial/localization/`** — both are documented, guarded, intentional
  pauses. Their LOC is not debt.
- **Adding a config framework** — `config.py`'s `Settings` is already correct and validated; the
  gap is a template and a test, not an abstraction.
- **Reorganizing `tests/` into subdirectories** — markers deliver the benefit at a fraction of the
  risk.
- **Finishing the llama.cpp migration** — it was rejected on a multi-intent regression.
- **Consolidating `semantic/`, `facts/`, and `mutation/`** — they have genuinely different
  semantics. Per the task constraints, no consolidation is proposed merely to reduce LOC.

---

## Appendix — Verification Recommendations (Phase 25)

For each stage, the check that actually proves the change was safe:

| Stage | Verification |
|---|---|
| 1 | `python -m pytest -q` → 983 pass. `git diff --stat` touches no file under `semantic/`, `facts/`, `mutation/` (except the 4 unused imports in `providers/ollama.py`). |
| 2 | 983 pass. Additionally: temporarily delete one entry from the moved `fact_questions.yaml` and confirm `test_fact_benchmark` **fails** — if it passes, the exact-string dispatch problem is not fixed. |
| 3 | 983 pass after updating `test_engineering_entrypoints.py:21-38`. Confirm `hc-bench list` still lists all 8 suites (`hc_suites.py:67`). |
| 4a | 983 pass. `test_ollama.py:360` (static-prefix stability) passes unchanged. |
| 4b | 983 pass, and the planner prompt is **byte-identical**: compare `environment.semantic_prompt_fingerprint()` (`benchmark/environment.py:129`) before and after, and confirm `test_planner_prompt_audit.py:18` (`test_components_partition_actual_message_content`) still passes. |
| 4c/4d/4e | On the production GPU host, from an isolated fingerprinted package: `hc-bench --regression`. Compare plan, multi-intent, and partial counts against the recorded Ollama baseline in `artifacts/llamacpp-migration/REPORT.md` (plan 86/119, multi-intent 4/7, partial 3). **Any multi-intent regression stops the stage.** Do not run this locally and do not report a number obtained here. |
| 5 | 983 pass. For the GUI change, `tests/test_media_boundaries.py` must be updated in the same commit and still assert the boundary (media and cortex packages share no import) rather than a string. |

Two standing rules for whoever executes this:

1. **Every stage ends with a `.llm/` work log** per `AGENTS.md` and
   `.agents/skills/persistent-llm-work-log/SKILL.md`.
2. **Do not mix a Stage 4 step into a Stage 1–3 commit.** The whole value of the staging is that
   Stages 1–3 are provably behavior-free; that proof is destroyed by bundling a prompt change.
