Date: 2026-09-21 23:55 PDT
Type: debugging
Status: completed

## Objective

Fix the mutation component crash in `hc-bench --suite full`.

## Context

Run `20260922-064513-eb26` completed planner, fact, latency, and bilingual, then mutation raised `NameError: name 'suite' is not defined` while attaching the progress callback.

## Findings

`_experiment_group` referenced `suite.name`, but `suite` exists only in `_run_unified`. Mutation therefore produced no rows and no safety gates. The other component scores in that run are unaffected.

## Decisions

Pass the suite name into `_experiment_group`. Progress lines for that component use `mutation` or `unified-planner`.

## Changes

`scripts/benchmarks/hc_suites.py`. A deterministic test calls the group helper with a fake runner and checks the progress line.

## Validation

`python -m pytest -q tests/test_benchmark_framework.py::test_mutation_progress_callback_is_bound` passed. The corrected file was copied to `home-cortex-0`. No model run.

## Remaining Issues

Run `20260922-064513-eb26` has no mutation metrics.

## Recommended Next Step

`hc-bench run --suite mutation --model qwen3.5:9b` on the host. The rest of `20260922-064513-eb26` does not need to be repeated for planner, fact, latency, or bilingual.
