Date: 2026-09-24 21:31 PDT
Type: review
Status: review-only

## Objective

Review the current repository and recommend next steps, including deployment and model evaluation.

## Context

Reviewed active Python API, identity handling, Docker proxy, benchmark gates, saved analysis, composition fingerprint, and frontend/media validation. The working tree already had user edits to benchmark analysis and Docker Compose plus untracked artifacts and logs; those were left intact.

## Findings

- A GUI cookie authenticates requests, but `mapped_person_id` passes both the signed session identity and client request headers to `resolve_user_entity_id`, which prefers `X-OpenWebUI-User-Id`/Email. Nginx proxies those headers without a trusted overwrite. A valid GUI session can therefore select another mapped identity via a request header. The shared bearer key also lets its holder request a session for any mapped identity.
- `/admin/ingest` and `/admin/export` use general `authenticate_request`, which accepts a GUI cookie. Export accepts any absolute destination writable by the API process. When no identity map and no API key are configured, authentication returns without a check; Compose defaults both to empty and exposes port 80. The settings validator does require a key when an identity map exists.
- `hc-bench run` absolute gates cover preview-as-commit and partial multi-intent plans. Rejection correctness is gated by `hc-bench compare`; a standalone candidate run with a rejection regression can exit successfully.
- Saved matched-cohort results favor `qwen3.5:9b` as current default (108/119 planner, 8/8 rejection, 1.462 s p50). `qwen3.8:27b` leads planner (111/119) but has 4/8 rejection and 14.757 s p50. These are single matched runs on Ollama 0.34.2 with no GPU/offload telemetry; Compose now pins 0.34.4 in an existing user edit.
- Composition fingerprint test references deleted `benchmarks/composition/codex-approval.md`. The historical blob in commit `adee6e6` exactly matches the recorded SHA-256.

## Decisions

Review only; no code or benchmark changes. Prioritize identity/admin authorization, then restore a green deterministic gate, then rerun controlled model comparisons.

## Changes

Added this work log only. No source, deployment, or benchmark files were changed.

## Validation

- `.venv/bin/python -m pytest -q`: 977 passed, 1 failed (missing composition approval file).
- `src/home_media/.venv/bin/python -m pytest -q`: 16 passed.
- `npm run build` in `src/home_gui`: passed.
- `src/home_gui/node_modules/.bin/tsc --noEmit`: passed.
- No live LLM benchmark or deployment test was run.

## Remaining Issues

Identity and admin authorization are not separated by role or credential. Composition fingerprint is stale. Model comparison lacks repeat runs, per-case diagnostics in this checkout, and resource telemetry.

## Recommended Next Step

Bind person identity to a trusted credential/session and add cross-user/admin denial tests. Restore the historical composition approval file or deliberately update the fingerprint if it is obsolete. On the GPU host, compare repeated 9B and 27B full runs on the same runtime/hardware, inspect the 27B rejection misses, and record effective model options plus GPU/offload telemetry before changing the default.
