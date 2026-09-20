Date: 2026-09-19 22:10 PDT
Type: refactor
Status: completed

## Objective

Reorganize the flat `home_cortex` backend into functional packages, remove
low-value re-export surfaces, make dependency direction explicit, and preserve
the validated semantic, mutation, HTTP, spatial, provider, and Vision behavior.

## Context

The starting backend had 43 production modules at the package root, 19,119 raw
Python lines, and 17,080 nonblank/non-comment Python lines. HTTP implementation
lived under `home_cortex.http` behind a root `api.py` public shim. The previous
cleanup had already recorded one unrelated deterministic-suite failure caused by
the absent `benchmarks/composition/codex-approval.md` fingerprint input.

## Findings

- Semantic interpretation, fact execution, mutation, persistence, provider,
  conversation, application-runtime, capability, and shared primitives each had
  distinct ownership but were flattened into the root namespace.
- `capabilities.dispatcher` imported catalog values only to re-export them.
- `vision.__init__` was a 105-line compatibility facade; concrete Vision modules
  already had direct callers. `vision.ports` remains justified because it owns
  transport-neutral ingestion and idempotency policy.
- The static import graph contained one five-module spatial cycle:
  `contracts`, `transforms`, `units`, `localization.pose`, and
  `localization.anchors`. Higher-level contract validation owned constants and the
  shared error that its lower-level dependencies needed.
- Of the final files over 500 lines, `capabilities/calendar.py` is the one clear
  mixed-responsibility follow-up: provider-neutral authorization/service policy
  and the Google adapter/HTTP transport share one module.

## Decisions

- Update repository imports atomically instead of retaining dozens of obsolete
  root compatibility modules.
- Preserve `home_cortex.api:app` as the deployment/public HTTP surface by making
  `api` the implementation package itself. Keep the intentional named-agent
  package surface as well; other package initializers remain minimal.
- Keep planner semantics, ontology, resolution, fact execution, rendering, and
  mutation behavior unchanged. `semantic/prompt.py` remains semantic rather than
  provider-owned.
- Extract only leaf spatial constants and `SpatialContractError` to
  `spatial/primitives.py`; this breaks the cycle without changing the public
  `spatial.contracts` imports used by callers.

## Changes

- Created functional packages: `api`, `capabilities`, `common`, `conversation`,
  `facts`, `mutation`, `persistence`, `providers`, `runtime`, and `semantic`.
- Moved HTTP implementation from `http/` to `api/`; moved localization-specific
  support to `spatial/localization/`. Root production modules fell from 43 to one
  (`config.py`).
- Removed the obsolete `api.py` and all other old root module paths after updating
  source, tests, and engineering scripts. Removed catalog re-exports from the tool
  dispatcher and reduced `vision.__init__` to a one-line ownership note.
- Added `spatial/primitives.py`, eliminated the only import SCC, and added
  architecture tests that require an acyclic production import graph and the lean
  root namespace.
- Updated `src/README.md`, `AGENTS.md`, and the benchmark path note with the active
  package map and a "Where does new code go?" guide.
- Final backend size is 19,023 raw Python lines and 16,992 nonblank/non-comment
  lines, down 96 and 88 respectively (about 0.5%). The scoped active documentation
  set grew from 555 to 598 lines to record the new ownership map.

## Validation

- `python -m compileall -q src/home_cortex scripts tests`: passed via `.venv`.
- Architecture/semantic/Vision guards: 25 passed before the final graph guards;
  final backend architecture guard: 10 passed.
- Spatial contract, unit, transform, ingestion, and localization focused run:
  112 passed.
- Mandatory chat, conversation, streaming, agent/bare-model, semantics, bilingual
  rendering, unified planner, mutation, provider, spatial, ingestion, and Vision
  run: 446 passed.
- Static production import audit: zero cycles and zero suspicious dependency
  direction edges.
- Final full deterministic suite: 945 passed, 1 failed. The sole failure remained
  the pre-existing missing
  `benchmarks/composition/codex-approval.md` file.
- `git diff --check`: passed. Direct imports of all major package owners passed.
- No real-model benchmark was run; no semantic or production-accuracy claim was
  made.

## Remaining Issues

- Restore or deliberately regenerate the separately governed composition approval
  artifact before expecting a fully green suite.
- Consider a later, behavior-preserving split of `capabilities/calendar.py` into a
  provider-neutral service and a Google adapter. The other >500-line modules have
  a coherent single responsibility despite their size.
- Old root import paths were removed, not shimmed. Only the required deployment
  surface `home_cortex.api:app` remains compatible.

## Recommended Next Step

Complete the evaluation-artifact correction, then split the calendar adapter only
if its provider-neutral contract can be preserved with focused authorization and
transport tests.
