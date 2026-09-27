Date: 2026-09-26 19:23 PDT
Type: coding
Status: partial

## Objective

Implement the verified repository-cleanup ticket, starting with security, mechanical hygiene, and benchmark integrity.

## Context

Started from `47f7d35`. The ticket and its review log were already untracked from the previous work cycle. The user chose to retain the household shared-key model; bearer clients remain trusted to select mapped identities.

## Findings

- Three locally unused imports in `providers/ollama.py` are compatibility exports used by tests and engineering scripts; only `_PLANNER_HISTORY_BOUNDARY` was safely removable without a caller migration.
- Production runs `e689819` with the API key configured in `docker/.env`, not a repository-root `.env`. The running services are cortex-api, home-gui, home-media, ollama, proxy, and surrealdb. No production files or containers were changed.
- The standalone Tier-1 runner and copy helper have historical report and reproduction references. The four retired no-report scripts had no active source callers beyond the entrypoint help sweep; historical `.llm` notes remain in git.
- The fact corpus contained 33 questions and five speaker cases. Their existing benchmark hashes remained unchanged after the move to YAML.

## Decisions

- Keep the shared bearer key as household-wide authority. A signed GUI session always uses its issued identity; admin operations require a direct bearer credential. Require the key at Compose interpolation and API startup.
- Limit export to `CORTEX_EXPORT_ROOT`, fixed to the mounted `/app/export` in both Compose stacks. Resolve the target path before checking containment.
- Retain the Tier-1 runner, report-backed scripts, public API exports, and llama.cpp candidate configuration until their compatibility or model gates are met.
- Keep provider/prompt refactoring and production deployment out of this local cleanup pass.

## Changes

- Added security request tests and fail-closed auth/startup behavior; bound GUI identity to the session; restricted admin credential and export paths. Updated docs and Compose configuration.
- Removed the dead Tier-0 Compose variable and one unused provider import. Added a sanitized `.env.example` and configuration checks.
- Moved fact cases to `benchmarks/fact_questions.yaml`; tightened the fake planner; added corpus hash checks. Added the frozen composition manifest to the fingerprint map and checked its contents against the source cases and sequences. Frozen case YAML and household inputs did not change.
- Added `make test` for root and home-media suites, documented focused pytest markers, and retired four no-report engineering scripts with their help-sweep entries.
- Corrected the harness documentation to distinguish the shared Tier-1 probe from its standalone CLI; removed an unreachable no-key branch from session reading.
- Updated the live cleanup ticket with implemented and remaining work.

## Validation

- `make test`: 991 root tests and 16 home-media tests passed. Home-media reported two dependency deprecation warnings.
- The `not slow` selection collects 978 tests and deselects 13 subprocess help cases.
- Both Compose variants rendered with a test key; the base variant rejected an empty key as intended.
- `hc-bench list` still registered all eight suites; the retained engineering help tests passed.
- A temporary removal of one fact question made the approved-corpus test fail; the original YAML bytes were restored afterward. Before and after: question hash `e0d1101c0a13dd477ef23d0dc95842b13b794ef0448cfa05a7d9038d818345ee`, speaker-case hash `6fd19da5ab57b708db20f79d211699ef752d17b95182fd17bbf6e32af6712bb2`.
- `git diff --check` passed. No live model run or production deployment was performed.

## Remaining Issues

The ticket remains open: production security rollout and smoke checks; shared test fixtures and brittle assertions; Tier-1 compatibility assessment and Compose deduplication; prompt/provider seam with production GPU parity gate; optional GUI/media cleanup. The shared-key model still lets any bearer-key holder select any mapped household identity, as chosen by the user.

## Recommended Next Step

Review and deploy the security slice from an isolated production package using `docker/.env`, with health, authenticated chat, GUI session identity, admin denial, and export-root smoke checks and a rollback image. Then continue C2/C3 in separate review cycles. Capture a same-runtime fingerprinted baseline before C4.
