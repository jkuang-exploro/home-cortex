Date: 2026-09-24 08:09 PDT
Type: investigation
Status: completed

## Objective

Identify useful follow-up local models after the Huihui Qwen3.5 9B abliterated benchmark regression.

## Context

The retained full-suite exports show base `qwen3.5:9b` at 108/119 planner and 7/7 multi-intent, versus `huihui_ai/qwen3.5-abliterated:9b` at 87/119 and 4/7. The benchmark report records matched prompt/corpus/config fingerprints but not final Ollama renderer or effective model-package options.

## Findings

- `lukey03/qwen3.5-9b-abliterated` is a separately produced Qwen3.5 9B abliteration with a published Q4_K_M Ollama tag around 5.6 GB. Its published package has its own system prompt and parameters, so an untouched tag does not isolate weight effects.
- `huihui_ai/qwen3.5-abliterated:27b` and official `qwen3.5:27b` each list approximately 17 GB Q4 weights. These would require substantially more memory than the historical 8 GiB GPU for a clean latency comparison.
- Another null-space 9B GGUF publisher labels its current release low quality, so it is a lower-priority candidate.
- `src/home_cortex/providers/ollama.py` uses `think=False`, temperature 0, a JSON schema, and 16,384 context for planner calls. The existing 27B benchmark is tagged `qwen3.8:27b`, not official `qwen3.5:27b`.

## Decisions

Recommend a short controlled evaluation of the independent 9B abliteration first, then a matched official/base versus abliterated 27B pair only on a host with sufficient memory. Preserve mutation rejection and partial multi-intent plans as gates.

## Changes

No source, benchmark, or analysis artifacts were changed for this recommendation.

## Validation

Checked existing export/report metrics, provider request settings, benchmark harness guidance, and publisher model pages. No benchmark or tests were run.

## Remaining Issues

No Home Cortex accuracy data exists for the recommended new candidates. Model package templates, system prompts, and effective options need to be recorded in any comparison.

## Recommended Next Step

On the frozen GPU-host package, inspect candidate/base Modelfiles and run the same targeted planner, bilingual, rejection, and multi-intent cases before committing to another full suite.
