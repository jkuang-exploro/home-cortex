Date: 2026-09-21 23:45 PDT
Type: coding
Status: completed

## Objective

Show that an `hc-bench` run is moving while model calls are in flight.

## Context

`full` previously printed the Ollama address and then nothing until the final report. A multi-minute suite looked stalled.

## Decisions

Progress lines go to stderr and `benchmarks/results/<run-id>/progress.log`, flushed on every line. Each case prints `...` before the model call and `pass` or `fail` with latency when it returns. The final report on stdout is unchanged aside from the progress-log path. Scoring is unchanged.

## Changes

`src/home_cortex/benchmark/progress.py` and the run recorder. Optional progress callbacks on the planner, fact, latency, unified, and bilingual runners. Adapters forward them.

## Validation

`python -m pytest -q tests/test_benchmark_framework.py tests/test_fact_benchmark.py tests/test_unified_planner_experiment.py tests/test_semantic_planner_benchmark.py` — 49 passed. No model run.

## Remaining Issues

None identified.

## Recommended Next Step

Re-run the suite on `home-cortex-0` and follow stderr or `progress.log`.
