# Work ticket — Repository cleanup after source review

**Prepared:** 2026-09-26 19:00 PDT  
**Status:** In progress; local implementation on `47f7d35`  
**Source:** [repository-cleanup-audit.md](repository-cleanup-audit.md), checked against `47f7d35`  
**Goal:** Reduce duplicated ownership and stale tooling while preserving the semantic interpreter, deterministic executor, benchmark contracts, and deployed behavior.

## Progress (2026-09-26)

- S0 local code and request tests: API key required; signed GUI session identity wins over client headers; admin routes require bearer; export is restricted to `CORTEX_EXPORT_ROOT`. The user chose the shared-key trust model, so a holder of that key remains authorized to supply any mapped user identity. Production rollout and smoke checks remain open.
- C1: dead Tier-0 Compose values and one truly unused Ollama import removed; `.env.example`, config checks, and boundary documentation added. Public API exports have been inspected but left intact pending compatibility review.
- C2: fact corpus moved to YAML with unchanged hashes; frozen manifest added to the approved fingerprint map and compared with source cases; `make test` runs both packages; initial pytest markers added. Shared fixture extraction and brittle assertion cleanup remain open.
- C3: four no-report scripts retired after reference inventory; standalone Tier-1 runner, its copy helper, and report-backed scripts retained pending compatibility review. Compose deduplication remains open.
- C4 and C5 remain open. No production model benchmark or deployment has run for this cleanup.
- Current local gate: `make test` passes **991 root tests** and **16 home-media tests**. Both Compose variants render with a configured key and refuse an empty key. Production still runs `e689819` from `/home/jkuang/Workspace/home-cortex` with its key in `docker/.env`.

## Verified starting point

- Current deterministic baseline: **985 root tests** (`.venv/bin/python -m pytest -q`) and **16 home-media tests** (from `src/home_media`, `.venv/bin/python -m pytest -q`). The audit's 983 tests were measured at `e689819`; do not use 983 as an acceptance count for this tree.
- [`providers/base.py`](../src/home_cortex/providers/base.py) puts three `plan_*` methods and `last_planner_runtime` on `ModelProvider`. Both live transport implementations import planner and mutation concerns. The adapters have different response types and transport options, so a shared path needs explicit response normalization.
- [`semantic/unified_planner.py`](../src/home_cortex/semantic/unified_planner.py) replaces text in the fact planner's first message and computes a reminder position from the number of examples. These operations couple prompt layout to another module's internals.
- [`scripts/benchmarks/hc_suites.py`](../scripts/benchmarks/hc_suites.py) calls `run_tier1_probe` from `semantic_planner_benchmark.py`; it does **not** import `tier1_latency_bench.py`. The standalone Tier-1 CLI still emits its own provenance, planner contract, JSONL progress, and report shape. Retire it only after comparing those outputs with the supported `hc-bench latency` command and updating the documented workflow.
- The audit disagrees with itself about script retirement: its detailed inventory retains three report-backed tools, while later sections say to delete six or seven. Treat only the four no-report candidates as initial retirement candidates; check references and reproducibility before deleting any script.
- `hc-bench --regression` is not a command in [`benchmark/cli.py`](../src/home_cortex/benchmark/cli.py). Use `hc-bench run ...` followed by `hc-bench compare BASE CANDIDATE` on the production GPU host from isolated, fingerprinted packages.
- The API auth concerns have source support in [`api/dependencies.py`](../src/home_cortex/api/dependencies.py), [`common/identity.py`](../src/home_cortex/common/identity.py), the admin routes, and Compose defaults. The reported exploit paths have **not** been reproduced in this review. Treat security work as a separate change with its own tests and rollout review.

## Scope and order

Implement as separate reviewable changes. Each change gets a dated `.llm/` log and the tests that cover its boundary. Do not combine provider or prompt edits with housekeeping, because their real-model evidence differs.

### S0 — Security hardening ticket (separate from cleanup; highest priority)

**Inspect:** [`api/dependencies.py`](../src/home_cortex/api/dependencies.py), [`common/identity.py`](../src/home_cortex/common/identity.py), [`api/routes/system.py`](../src/home_cortex/api/routes/system.py), [`docker/docker-compose.yml`](../docker/docker-compose.yml), and [the earlier security review](2026-09-24_2131_repository-review.md).

1. Reproduce the unconfigured-auth, identity-header precedence, and admin-export path behavior with request-level tests. Record which ingress routes can reach each endpoint in the actual Compose topology.
2. Define and implement an explicit production auth policy: a missing shared key must fail closed; a signed GUI session must keep its issued identity despite later headers; bearer clients holding the shared key may supply mapped identities; admin operations require the bearer credential directly; export destinations need an approved root or equivalent safe destination policy.
3. Specify credential provisioning, OpenWebUI/GUI compatibility, a rollback procedure, and production smoke checks before deployment. Keep this security rollout independent of cleanup and llama.cpp migration.

**Acceptance:** Tests demonstrate denial for unauthenticated requests, GUI-session header spoofing, cookie-only admin access, and out-of-policy export paths; legitimate authenticated clients still work; deployed configuration is explicit and verified. The shared bearer key is a household-wide authority and does not prevent a key holder from choosing a mapped person.

### C1 — Mechanical hygiene and documentation

1. Remove the truly unused `_PLANNER_HISTORY_BOUNDARY` import in [`providers/ollama.py`](../src/home_cortex/providers/ollama.py) and the unused `HOME_CORTEX_DISABLE_TIER0` values in both Compose files. Three other locally unused imports (`_PLANNER_INSTRUCTIONS`, `_semantic_planner_examples`, `planner_system_prompt`) are compatibility exports used by tests and engineering scripts; migrate those callers to `semantic.prompt` before removing the imports. Add a sanitized `.env.example` and a config check that distinguishes service wiring variables from `Settings` fields; do not put secrets or live household values in the template.
2. Review the 16 names exported by [`api/__init__.py`](../src/home_cortex/api/__init__.py) against production, tests, and documented import use before changing public exports. `_stream_chat_completion` is used by `tests/test_api.py`.
3. Document the existing bounded retry guards in [`semantic/planner.py`](../src/home_cortex/semantic/planner.py) as a conscious exception to the utterance-text invariant. Changing or removing the guards is a behavior task with a real-model gate, outside this mechanical slice.
4. Document `benchmark/` as CLI-only in `src/README.md`. Its dynamic import of `scripts.benchmarks.hc_suites` is an intentional benchmark-entry exception to the rule that runtime modules do not import scripts. Update the GUI API document to include `/media-api`; document the planner-selection boundary.
5. Inventory proposed artifacts and file moves before editing: keep report-backed tools and required archives; move a report builder only if its import paths and report reproduction are preserved. Empty directories have no tracked deletion value by themselves.

**Acceptance:** Root and home-media suites pass; Compose config renders for the supported variants; documentation and example variables match current configuration. No prompt messages or corpus bytes change in this slice.

### C2 — Test fixtures and benchmark integrity

1. Extract repeated `MemoryDatabase`, fake-provider helpers, test data paths, and request-context builders into `tests/conftest.py` or a small test-support module. Remove cross-test imports without creating global mutable fixtures. Add documented `architecture`, `semantic`, `benchmark`, and `slow` markers only where useful; preserve a full-suite gate. Replace fragile source-string and line-count tests with assertions of the actual import, routing, and media boundaries.
2. Include the separate `src/home_media/tests` suite in the repository's repeatable verification command or CI job. Root pytest does not collect it. Preserve separate package dependencies and the no-cross-import boundary.
3. Add `benchmarks/composition/frozen/MANIFEST.yaml` to the approved fingerprint process and assert that its case IDs and sequence IDs agree with the frozen YAML inputs. Review the resulting fingerprint change explicitly; do not silently bless modified frozen data.
4. Move the 33 fact questions and five speaker cases in [`fact_benchmark.py`](../scripts/benchmarks/fact_benchmark.py) to a fixed benchmark input path. Preserve order, wording, and scoring. Replace exact-string fake dispatch that falls through silently with exhaustive corpus-aware assertions; demonstrate that deleting or changing a covered entry makes the test fail. Record before/after corpus hashes and any benchmark provenance change.

**Acceptance:** Both suites pass; the new repeatable check runs both; frozen-manifest drift and fact-corpus drift fail targeted tests; suite membership and scoring are unchanged.

### C3 — Engineering tool retirement and deployment-file deduplication

1. Inventory every script against imports, shell calls, documentation, report reproduction instructions, and `tests/test_engineering_entrypoints.py`. Evaluate the four no-report candidates (`item_location_probe.py`, `http_latency_audit.py`, `token_component_probe.py`, `context_surface_audit.py`) for removal. Retain `layer_failure_trace.py`, `kinship_context_probe.py`, and `profile_semantic_transport.py` while published reports depend on them, unless a separately reviewed archive plan preserves reproduction.
2. Compare standalone `tier1_latency_bench.py` output and options against `hc-bench run --suite latency`; migrate any needed provenance or progress contract before removing it and `copy_tier1_bench_into_api.sh`. Update `benchmarks/HARNESS.md`, `scripts/README.md`, and entrypoint tests in the same change. The shell helper's `docker/cortex` default is stale, but documentation still advertises container copying.
3. If the llama.cpp Compose file becomes an overlay, verify resolved services, volumes, env values, networking, and published ports for both Ollama and llama.cpp variants. Keep GPU overlays working. A Compose rewrite is a deployment change and needs a production-like configuration comparison.

**Acceptance:** No retained report loses its reproduction command; `hc-bench list` still registers its expected suites; root suite passes; old and new Compose renders are compared before deployment. Report actual file deletions and moved lines separately, without treating moves as savings.

### C4 — Planner and provider seam (separate semantic review)

1. First isolate transport-neutral runtime-metric normalization and remove Ollama-specific wording from generic code, with adapter tests for Ollama, OpenRouter, and llama.cpp response shapes.
2. Replace unified-planner string replacement and example-count indexing with an explicit shared message builder. Capture representative fact and unified messages before the edit and assert **byte-for-byte** equality afterward, including static prefix, example order, reminder, clock, history, and identity notes. Preserve schema and decoding behavior.
3. Move prompt construction, output schema, validation, and the three planner entry points out of `ModelProvider`. Expose a transport-level structured call and normalize results and timing in one defined place. Preserve existing provider-specific transport parameters, errors, streaming behavior, and the architecture import guard. Do not widen semantic repair or merge entity and relationship property handling.
4. Only after parity is established, consider collapsing duplicated `runtime/model_loop.py` run/stream control flow. Treat this as a separate change with tool-budget and stream-cancellation tests.

**Acceptance:** Root suite and targeted adapter/planner tests pass; captured prompt messages are byte-identical; the production GPU host records a same-model, same-data, same-runtime baseline and candidate from isolated packages with model, prompt, corpus, schema, package, and runtime fingerprints. Run the affected `hc-bench` suites and compare run IDs; inspect plan, mixed-intent, partial-plan, mutation, and latency outcomes separately. Any new safety regression blocks promotion. The previous Ollama result (4/7 mixed intent with three partial plans) already fails the absolute safety gate and cannot be called a passing baseline. A prompt-byte change requires its own behavioral evaluation and approval, even if deterministic tests pass.

### C5 — Optional package-local cleanup

- GUI: share the API error envelope and common request handling while preserving media timeout, credentials, and content-type behavior; extract conversation lifecycle only if tests can cover the new seam; complete media-page translation keys and remove proven dead CSS. Run the GUI build and relevant boundary tests.
- Home-media: give path containment, media IDs, and thumbnail cache paths one owner each. Preserve traversal and symlink protections and run the separate media suite.
- API: move greeting text/policy, provider discovery, or trace middleware only when the resulting owner is clearer and response behavior is covered. Keep these moves out of C4's prompt comparison.

## Closure criteria

- Each completed slice has a focused diff, the relevant deterministic and package-local tests, updated docs, and a dated `.llm/` work log. No accuracy claim comes from the local fake-provider suite.
- Any production model claim uses an isolated package and recorded fingerprints on `home-cortex-0`, with `hc-bench run ...` and `hc-bench compare ...` or the documented suite-specific harness. Keep security rollout, provider refactor, and llama.cpp promotion as distinct deployment decisions.
- The audit's LOC estimates are planning estimates, not acceptance targets. Verify actual deletion, deduplication, additions, and moves from the final diffs. Preserve the existing semantic layer boundaries and compositional behavior.
