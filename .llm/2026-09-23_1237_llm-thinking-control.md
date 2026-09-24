Date: 2026-09-23 12:37 PDT
Type: investigation
Status: completed

## Objective

Locate where Home Cortex controls LLM thinking effort.

## Context

The active `artifacts/qwen3.5-4b.yaml` is a generated benchmark summary. Recent benchmark and Ollama endpoint notes were reviewed against current provider code.

## Findings

`src/home_cortex/providers/ollama.py` passes `think=False` on ordinary chat, tool, mutation, semantic fact, and unified planner calls. `src/home_cortex/providers/openrouter.py` builds request payloads without an explicit reasoning effort field. `PLANNER_NUM_PREDICT = 384` in `semantic/prompt.py` limits generated output tokens; it does not select thinking effort. Provider choice and model name come from `config.py` through `providers/base.py`.

## Decisions

Thinking controls belong in provider request construction, with configuration in `config.py` if the setting should be adjustable at deployment time.

## Changes

No source changes were made.

## Validation

Inspected current provider, prompt, and factory source. No tests were run because no behavior changed.

## Remaining Issues

Supported thinking levels depend on the deployed model and provider API and were not verified in this investigation.

## Recommended Next Step

If adjustable effort is desired, add a validated setting and pass it through the selected provider adapter; test the generated request payload and run a real-model benchmark on the GPU host.
