# Tier-1 baseline (scored probe + full suite)

Date: 2026-09-06  
Host git commit at collection: `69776d06ca8b63f7fa582c705e71ab33bf449c64` (dirty: benchmark tooling changes)  
Container provenance `git_commit`: unavailable (ran from `/tmp`, no `.git`)  
Model: `qwen3:8b` digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`  
Ollama: `0.32.13`  
Hardware: Linux aarch64 (linuxkit). `ollama ps` during earlier work: **100% CPU**.  
Backend: JSON graph. Frozen time: `2026-09-03T12:00:00-07:00`. Tier-0 disabled.  
Settings: think=false, temperature=0, num_predict=384, keep_alive=24h.

JSON artifacts contain household records. Use them locally; do not treat stdout logs as the source of truth.

## Commands

Full 119-case suite:

```bash
PYTHONPATH=src python -m home_cortex.semantic_planner_benchmark \
  --data-dir data --schema-dir schemas/edge \
  --eval benchmarks/semantic_planner_eval.yaml \
  --ollama-url http://127.0.0.1:11434 --model qwen3:8b \
  --output artifacts/tier1-baseline/suite.json
```

20-question probe, 1 warm-up + 5 measured passes:

```bash
PYTHONPATH=src python scripts/tier1_latency_bench.py \
  --data-dir data --schema-dir schemas/edge \
  --eval benchmarks/semantic_planner_eval.yaml \
  --ollama-url http://127.0.0.1:11434 --model qwen3:8b \
  --warmup 1 --repeat 5 \
  --output artifacts/tier1-baseline/probe.json
```

This collection used the Cortex API container + `http://ollama:11434` because host `:11434` has no models.

## 20-question probe (first measured pass)

| Metric | Numerator / denominator |
|---|---|
| Plan accuracy | **15 / 20** (rate 0.75). Unscored **0 / 20**. |
| Answer correctness | **15 / 20** (rate 0.75). Unscored **0 / 20**. |
| Executor `found` | **16 / 20** |
| `not_run` | **2 / 20** (both marriage questions, validation failed) |
| `ambiguous` | **1 / 20** (`son_birth_date`) |
| `computation_impossible` | **1 / 20** (`pairwise_older_wife`) |

Across all measured samples: **75 / 100** plan, **75 / 100** answer. Warm-up: 1 `first_request` (not verified-cold). Latency n=100, P50 **7547 ms**, P95 **10192 ms**.

### Named failures (first measured pass)

| Case | Stage | Actual vs expected |
|---|---|---|
| `marriage_start` | planner_validation `UNKNOWN_PROPERTY` | `select(self→spouse, start_date)` with default `property_source=entity`. Expected `property_source=relationship`. Executor not run. |
| `marriage_duration_wife` | planner_validation `UNKNOWN_PROPERTY` | `duration(..., start_date, mode=days)` also `property_source=entity`. Expected `relationship`. Executor not run. |
| `pairwise_older_wife` | plan_mismatch | Emitted household `argmax(birth_date)` plus both `adult` and `minor` filters. Expected `argmin(self, other=self→spouse[female], birth_date)`. Executor `computation_impossible`. |
| `wife_father_given_name` | **pass** | `person:zhigang_ba`. |
| `son_birthday_countdown` | plan_mismatch | `date_difference` on son's `birth_date` → value **3595**. Expected `annual_occurrence` mode=days → **57**. Executor `found` (wrong operation, right child). |
| `named_dylan_birthday_countdown` | **pass** | `德伦` → `person:dylan_kuang`. |
| `named_dylan_identity` | **pass** | `匡德伦` → `person:dylan_kuang`. |
| `son_birth_date` | plan_mismatch | `self→child` without `gender=male` → `ambiguous` Dylan+Evelyn. Expected filtered son `select birth_date`. |

## 119-case suite (single pass)

Plan accuracy **88 / 119** (0.7395). Answer correctness **unscored 119 / 119** (suite groups have no result expectations). Executor: found 94, not_run 11, ambiguous 10, relationship_not_found 2, computation_impossible 2. P50 **5913 ms**, P95 **9192 ms**. Tier-0 parity 6/6.

Weakest capabilities: temporal_operation 3/15, multi_hop_kinship 3/7, relationship_property_lookup 4/8.

## Uncertain expectations (documented in YAML `notes`)

- `given_name` → deployed `first_name` (`Evelyn`, `Zhigang`).
- Duration 4505 days and countdown 57 days depend on frozen `2026-09-03`.
- Pairwise older = `argmin(birth_date)` on self vs wife.
