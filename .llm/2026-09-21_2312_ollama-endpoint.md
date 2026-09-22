Date: 2026-09-21 23:12 PDT
Type: debugging
Status: completed

## Objective

Stop `hc-bench` from scoring a run when it cannot reach Ollama on the GPU host.

## Context

`hc-bench run --suite full --model qwen3.5:9b` on `home-cortex-0` recorded plan correctness 0/119 and fact completion 38/38. Every failure was `ConnectionError`. The harness defaulted to `http://127.0.0.1:11434`. Compose does not publish Ollama, and the hostname `ollama` does not resolve on the host. The container `cortex-ollama-1` answers at its bridge address.

## Findings

The fact runner turns a connection error into `semantic_plan_unsupported` and keeps going, so the adapter reported those rows as completed. The planner runner records the same error as a missed plan. Either way the printed ratios were not model scores.

## Decisions

The default endpoint remains the project `OLLAMA_URL` (`http://ollama:11434`). If that is not reachable and the user did not pass `--ollama-url`, the harness tries localhost and then a local Docker container whose name contains `ollama`. An explicit URL is never replaced. If nothing answers, the command exits 1 and writes no score. A component whose scored cases are all Ollama connection failures is a harness failure, not a zero accuracy.

## Changes

`src/home_cortex/benchmark/environment.py`, `runner.py`, `cli.py`, `types.py`, `present.py`, and `scripts/benchmarks/hc_suites.py`. `benchmarks/HARNESS.md` describes the address rule.

## Validation

`python -m pytest -q tests/test_benchmark_framework.py` — 23 passed. On `jkuang@home-cortex-0`, after copying the fix, `hc-bench run --suite planner --model qwen3.5:9b --limit 1 --label connect-smoke` used `http://172.21.0.4:11434` (container `cortex-ollama-1`). Ollama 0.32.15, model digest `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`, quantization `Q4_K_M`. Plan correctness 1/1. Run `20260922-061054-be28`. That case cap is not a suite score. The earlier run `20260922-060318-d936` is not a model result.

## Remaining Issues

The host checkout is dirty because the fix was copied onto commit `38148ca`. Container bridge addresses can change after `docker compose up`; each run discovers the address again.

## Recommended Next Step

Re-run `hc-bench run --suite full --model qwen3.5:9b` on the host. Do not compare it with `20260922-060318-d936`.
