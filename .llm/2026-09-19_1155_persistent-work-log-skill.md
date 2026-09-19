Date: 2026-09-19 11:55 PDT
Type: refactor
Status: completed

## Objective

Make `.llm/` the required persistent engineering memory for meaningful LLM-assisted
work, implement the workflow as a repository skill, and consolidate existing
LLM-oriented reports without moving benchmark evidence or product documentation.

## Context

The repository had an empty `.llm/` convention until the backend architecture
breakdown was added. Engineering-memory reports were spread across `artifacts/`,
`docs/`, and `benchmarks/`, while `artifacts/` also contained measured benchmark
summaries that should remain with their evidence.

## Findings

Nine historical documents were clearly engineering-session memory: architecture
and context refactors, utility consolidation, a Vision review, a planner authority
investigation, a semantic next-work handoff, a composition review, and a
query-generalization handoff/implementation pair. Other artifact reports directly
explain benchmark evidence and remain in `artifacts/`.

## Decisions

Use a repository-scoped skill at
`.agents/skills/persistent-llm-work-log/SKILL.md`, with `AGENTS.md` enforcing it as
part of completion. Keep measured output and benchmark reports in `artifacts/`,
active architecture in `src/` or `docs/`, and concise work-cycle memory in `.llm/`.
Consolidate the two query-generalization documents because they came from the same
commit and work cycle.

## Changes

- Added the persistent work-log skill and linked it from `AGENTS.md`.
- Normalized the backend architecture breakdown to the dated work-log format.
- Moved and normalized eight historical engineering-memory documents.
- Consolidated the query-generalization handoff and implementation note into one
  dated entry.
- Updated repository links and directory descriptions to their new locations.
- Removed duplicate originals from `artifacts/`, `docs/`, and `benchmarks/`.

## Validation

- Skill validator: `Skill is valid!`
- Every `.llm/*.md` entry was checked for all required summary sections.
- Search found no references to the retired document paths.
- `git diff --check` passed.
- No runtime tests were run because runtime code was unchanged.

## Remaining Issues

Historical logs retain their detailed original report bodies after the concise
summary. They can be shortened later if context size becomes a demonstrated problem;
Git history preserves the originals.

## Recommended Next Step

On the next meaningful work cycle, inspect relevant recent `.llm` entries first and
write or update the dated work log after final validation.
