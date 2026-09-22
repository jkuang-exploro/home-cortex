Date: 2026-09-21 23:31 PDT
Type: debugging
Status: completed

## Objective

Stop the bilingual component of `hc-bench --suite full` from aborting on gold-plan expansion.

## Context

Run `20260922-061658-b67d` on `home-cortex-0` finished planner, mutation, fact, and latency, then the bilingual probe raised `ValueError: unknown reference concept`.

## Findings

`benchmarks/fixtures/semantic-contract` had only person and address. Location, room, and contents concepts are omitted unless item and space exist, but the bilingual gold uses those concepts. `canonical()` raised while expanding the expected plan and discarded the component. The other suite numbers in that run are real model results. Planner answer correctness is `n/a` because `semantic_planner_eval.yaml` has plan gold only; the probe split carries answer gold (17/20 on the latency sample).

## Decisions

Add invented item, space, `located_in`, and `hosted_by` records to the semantic-contract fixture so those concepts expand. If a future gold plan still cannot expand, record that case and continue instead of aborting the probe. The synthetic-fixture prompt byte alarm moved from 32,000 to 36,000 because an item entity also pulls in declared item attributes.

## Changes

Fixture JSON under `benchmarks/fixtures/semantic-contract/`. `scripts/probes/bilingual_planner_probe.py` catches an unexpandable expected plan. `hc_suites.py` classifies that as a harness failure. `tests/test_planner_prompt_audit.py` bound updated.

## Validation

Local expansion of every bilingual expected plan succeeds. `python -m pytest -q tests/test_planner_prompt_audit.py tests/test_benchmark_framework.py tests/test_tools.py tests/test_semantic_planner_benchmark.py tests/test_semantic_planner_bilingual.py tests/test_unified_planner_experiment.py` — the prompt-budget test failed at 34,059 bytes before the bound change; after it, the budget test and the benchmark tests passed (33 in the last run). On the host, `hc-bench run --suite bilingual --model qwen3.5:9b --label bilingual-after-fixture` completed: run `20260922-062934-4201`, plan match 56/59, parity 10/10, no harness failure. The host tree is dirty from the copied files.

## Remaining Issues

Three bilingual semantic mismatches remain. They were not investigated. `full` was not re-run after the fixture change.

## Recommended Next Step

Re-run `hc-bench run --suite full` only if a single record that includes bilingual is required. Otherwise compare later candidates with `20260922-061658-b67d` for planner, mutation, fact, and latency, and with `20260922-062934-4201` for bilingual.
