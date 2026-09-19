Date: 2026-09-19 14:50 PDT
Type: refactor
Status: completed

## Objective

Slim and clarify the active backend after the client-runtime extraction: split the
HTTP monolith, unify transport execution, establish a provider contract, create a
public mutation boundary, isolate optional spatial/maintenance imports, and remove
obsolete documentation and production dependencies without semantic regression.

## Context

The ticket-phase baseline was 18,902 raw Python lines in `src/home_cortex` (16,957
nonblank/non-comment lines), a 1,383-line `api.py`, 949 lines across the two provider
adapters, 922 non-Python package lines, and an 836-line `src/README.md`. The earlier
Vision/client extraction had already removed the device runtime and recorded the
missing composition approval artifact as a pre-existing test failure.

## Findings

- HTTP transport, lifecycle wiring, auth, serialization, and route families were
  separable without changing endpoint paths.
- Streamed and collected conversation responses repeated source consumption and
  assistant persistence; both can consume one answer execution object.
- Planner prompt construction was shared semantic behavior embedded in `ollama.py`;
  OpenRouter imported the Ollama adapter to reuse it.
- `AgentService` called private `ModelLoop._dispatch` for planned writes.
- `writing.py` imported maintenance-heavy `ingestion.py` for one record-ID helper.
- Spatial anchor/observation/localization code is pure validation/math, not a live
  device loop, but package re-exports loaded it during unrelated imports.
- The fact planner and unified planner remain distinct active strategies. No
  semantic module had a proven authoritative replacement suitable for deletion.

## Decisions

- Keep `home_cortex.api:app` as a small compatibility surface and put active HTTP
  ownership under `home_cortex.http`.
- Preserve spatial contracts and deterministic solvers, but use an empty package
  initializer and keep live robotics runtime in `home_cortex_client`.
- Keep the semantic executor, schema, resolver, renderer, and both planner strategies
  intact. Move only provider-neutral prompt construction.
- Preserve non-stream persistence failure behavior while suppressing post-response
  persistence errors for an already-started stream.
- Do not fabricate or silently restore the absent composition approval artifact;
  leave that for the separate evaluation-correction cycle.

## Changes

- Replaced the HTTP monolith with app assembly, dependencies, errors, schemas, SSE,
  provider cache, execution, and five route-family modules. `api.py` is 42 lines;
  app assembly is 153 lines.
- Added `AnswerExecution`, used by conversation and OpenAI-compatible streamed and
  collected responses, with one assistant persistence point.
- Added `ModelProvider`, `ModelMessage`, and `ModelResponse` protocols plus the
  deployment factory. Agent/planner callers no longer import concrete adapters.
- Extracted the 269-line semantic prompt builder. Ollama fell from 505 to 225 lines;
  OpenRouter no longer imports the Ollama adapter.
- Added `MutationService`; planned writes no longer use `ModelLoop._dispatch`.
  Extracted model-facing tool schemas to `tool_catalog.py`; `tools.py` fell from
  563 to 342 lines.
- Moved `implicit_edge_component` to `record_ids.py`, so chat startup no longer
  imports ingestion/export. Startup also excludes Vision and optional spatial
  localization/observation modules.
- Consolidated four Vision documents into one 110-line boundary note and reduced
  `src/README.md` from 836 to 215 lines.
- Moved pytest, pytest-asyncio, pytest-cov, and httpx2 out of production dependencies
  and refreshed `uv.lock` offline.
- Added architecture and answer-execution regression tests. Removed unused app-state
  service aliases and migrated engineering prompt experiments to the new owner.

## Validation

- `git diff --check`: passed.
- Provider/API/mutation focused runs: 179 passed, then 155 passed after final
  extraction; prompt-owner focused run: 48 passed.
- Mandatory combined regression run (providers, mutations, conversations, household
  semantics, ingestion, spatial): 435 passed.
- Final full deterministic suite: 943 passed, 1 failed. The sole failure is the
  pre-existing `test_fingerprints_match_current_tree`, which raises
  `FileNotFoundError` for absent
  `benchmarks/composition/codex-approval.md`.
- No real-model semantic benchmark was run locally; repository policy requires the
  production GPU host and recorded fingerprints.
- Final package size: 19,119 raw Python lines and 17,080 nonblank/non-comment lines.
  This ticket phase added 217 raw Python lines for explicit boundaries while
  removing 1,320 documentation lines. Relative to the pre-extraction architecture
  baseline of 20,523 Python lines, the backend is down 1,404 lines (6.8%).

## Remaining Issues

- Repair the composition fingerprint artifact in the planned evaluation-correction
  cycle, then rerun the full suite.
- The 15-25% Python reduction target was not reached. Further reduction would
  require deleting retained Vision contracts or restructuring validated semantic
  code; neither had sufficient evidence for safe removal in this cycle.
- `ModelLoop.run` retains a metadata-producing non-stream path for `/v1/chat` while
  stream-first HTTP endpoints use the unified answer stream. Replacing it requires
  an event/result protocol that preserves steps, stop reason, and tool-call counts.
- OpenRouter still constructs the third-party Ollama `ChatResponse` model internally
  as its wire-normalization object. Application callers depend only on the neutral
  structural response protocol, but a fully owned canonical response model remains
  possible future cleanup.

## Recommended Next Step

Perform the separate evaluation-correction cycle for the missing composition
approval/fingerprint artifact. After a green baseline, profile whether an
event-bearing `AgentExecution` can replace `ModelLoop.run` without changing public
metadata or stream cancellation semantics.
