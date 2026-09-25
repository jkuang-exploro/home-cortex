Date: 2026-09-24 21:36 PDT
Type: debugging
Status: completed

## Objective

Determine the purpose of the missing `benchmarks/composition/codex-approval.md` and remove its obsolete test dependency if nonessential.

## Context

The approval file was a pre-generation review of the composition dataset. It was removed in commit `8340189`; its approved counts and locked decisions remain in `coverage-matrix.md`. The active fingerprint helper still tried to hash the deleted file, so the backend suite failed before comparing the other fingerprints.

## Findings

The approval prose is not read by the composition loader or scorer. The fingerprint test is useful because it checks current dataset counts and hashes. Once the missing-file dependency was removed, the recorded README, coverage matrix, ontology, and test-file hashes also needed to reflect the current tree.

## Decisions

Keep the fingerprint test and remove only its dependency on the deleted approval document. Preserve the approved dataset rules in the coverage matrix.

## Changes

- Removed the deleted file from `composition_fingerprint_payload()` and the composition README.
- Removed stale links to the deleted file from the coverage matrix.
- Regenerated `benchmarks/composition/fingerprints.json` from the current tree. No benchmark cases or household fixtures changed.

## Validation

- `.venv/bin/python -m pytest -q`: 978 passed.
- `git diff --check`: passed.
- No real LLM benchmark was run.

## Remaining Issues

None identified for this request.

## Recommended Next Step

Continue using the retained fingerprint test to detect future composition dataset or contract drift.
