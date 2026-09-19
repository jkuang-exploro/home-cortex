---
name: persistent-llm-work-log
description: Record concise repository engineering memory in .llm after meaningful coding, debugging, refactoring, investigation, benchmarking, architecture, or review work.
---

# Persistent LLM work log

Use `.llm/` as the repository's shared engineering memory. A meaningful work cycle
is incomplete until its log is written or updated.

## Start a work cycle

1. Inspect the newest `.llm/*.md` entries relevant to the task.
2. Treat them as historical notes, then verify material assumptions against the
   current repository.
3. Continue an existing file only for the same task in the same working session.
   Never overwrite or repurpose an unrelated historical entry.

## Finish a work cycle

Create `.llm/YYYY-MM-DD_HHMM_<short-topic>.md` using local time. Keep it concise,
fact-based, and useful to an engineer who did not see the conversation.

Use this structure:

```markdown
Date: YYYY-MM-DD HH:MM TZ
Type: coding | review | debugging | investigation | benchmark | architecture | refactor
Status: completed | partial | blocked | review-only

## Objective

## Context

## Findings

## Decisions

## Changes

## Validation

## Remaining Issues

## Recommended Next Step
```

Record only work actually performed and validation actually run. If no code changed,
write `No repository changes were made.` under Changes. If nothing remains, write
`None identified.` under Remaining Issues.

Do not include transcripts, hidden reasoning, credentials, secrets, tokens, private
environment values, or large diffs. Keep benchmark output in `artifacts/`; link its
small summary or report from the work log. Keep product architecture in `src/` or
`docs/`; record only the decision and result in `.llm/`.

Before finishing, confirm the log describes the final working tree rather than an
intermediate attempt.
