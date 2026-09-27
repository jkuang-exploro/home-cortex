Date: 2026-09-26 19:00 PDT
Type: review
Status: completed

## Objective

Inspect the repository-cleanup audit against the current tree and turn its actionable findings into a work ticket.

## Context

The audit was measured at `e689819` with 983 root tests. This review used `47f7d35`, after the llama.cpp parity follow-up. The audit remained unchanged.

## Findings

- The provider protocol contains planning methods, and the unified planner edits fact-planner messages by string replacement and example-count indexing.
- The audit gives conflicting script-retirement counts. Three candidates reproduce reports; the standalone Tier-1 runner has output and provenance behavior that needs comparison before removal.
- `hc-bench --regression` is absent from the current CLI; `run` and `compare` are supported.
- The root suite now has 985 passing tests; home-media has 16 tests in a separate project. The first media test invocation used the media interpreter from the repository root and collected the wrong test directory; rerunning from `src/home_media` passed.
- Auth configuration, identity-header precedence, and admin routes support a separate security investigation. Exploit behavior was not reproduced in this review.

## Decisions

- Keep security, hygiene, benchmark integrity, provider/prompt changes, and optional package cleanup as distinct implementation slices.
- Treat LOC totals and historical test counts as estimates and historical baselines, respectively.
- Require byte-for-byte prompt comparison and a fingerprinted production GPU benchmark for semantic changes.

## Changes

- Added [repository-cleanup-work-ticket.md](repository-cleanup-work-ticket.md) with scope, order, acceptance criteria, and corrected verification commands.
- No product source, audit, deployment configuration, or benchmark data changed.

## Validation

- `.venv/bin/python -m pytest -q`: 985 passed in 19.23s.
- From `src/home_media`, `.venv/bin/python -m pytest -q`: 16 passed, two dependency deprecation warnings.
- Checked current source, script references, benchmark CLI, composition fingerprints, and repository status.

## Remaining Issues

The work ticket has not been implemented. Security behaviors require request-level reproduction; script removal and provider refactoring require the listed compatibility gates.

## Recommended Next Step

Open the security hardening slice and then implement C1 and C2 as independent changes. Schedule C4 only with a reproducible real-model comparison baseline.
