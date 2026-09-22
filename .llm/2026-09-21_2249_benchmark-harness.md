Date: 2026-09-21 22:49 PDT
Type: coding
Status: completed

## Objective

Add one recorded, comparable command for Home Cortex model runs without a second benchmark corpus.

## Context

Existing scorers already live in `scripts/benchmarks/` and `scripts/probes/`. Runtime modules must not import `scripts`. Real-model suites belong on `home-cortex-0`. This session ran on `Jians-MacBook-Air`.

## Findings

Fact, planner, latency, unified-planner, and bilingual runners already score the datasets under `benchmarks/`. Composition eval, conversation tests, and the older mutation-routing probe are correctness gates or duplicates, not model suites. The planner diagnostics dropped Ollama `done_reason`, so context-length stops were not visible to a failure taxonomy.

## Decisions

`hc-bench` orchestrates. The old runners remain the scoring authority. Suite adapters live in `scripts/benchmarks/hc_suites.py` so `src/home_cortex` still has no static `scripts` import. The CLI loads them through the `home_cortex.benchmark_suites` entry point or, in an uninstalled checkout, by module name. `standard` is planner plus mutation intents. `full` adds fact, latency, and bilingual. The paired read regression stays in `unified-planner` because it repeats the planner corpus. Semantic suites score once. Warmup and repetitions apply to `latency` only. Safety gates fail the process when preview-as-commit, partial multi-intent plans, or a drop in preview, commit, rejection, or multi-intent correctness is recorded. No new semantic cases.

## Changes

- `src/home_cortex/benchmark/` records runs, compares them, and exposes `python -m home_cortex.benchmark`.
- `scripts/benchmarks/hc_bench.py` and the `hc-bench` entry point register the adapters.
- Fact, unified, and bilingual runners accept an optional model, schema directory, or smoke limit. Default CLI behavior is unchanged.
- Planner diagnostics keep Ollama `done_reason` when the runtime reports it.
- `benchmarks/HARNESS.md` documents the commands. Results go to `benchmarks/results/<run-id>/` and are gitignored.

## Validation

`.venv/bin/python -m pytest -q` — 971 passed, 1 failed. The failure is `tests/test_composition_eval.py::test_fingerprints_match_current_tree`: `benchmarks/composition/codex-approval.md` is missing. That file was already absent and was not part of this change. `hc-bench list`, `hc-bench run --help`, and `hc-bench compare --help` succeed in the project virtualenv. No real-model run was started.

## Remaining Issues

Repeatability on `home-cortex-0` was not measured. SSH to that host was denied (`publickey,password`). The composition fingerprint test still expects `codex-approval.md`.

## Recommended Next Step

On `home-cortex-0`, run `hc-bench run --suite standard --model qwen3.5:9b` twice and compare the two run ids before treating a small latency delta as a model difference.
