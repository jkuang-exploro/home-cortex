# Tier-1 production baseline: qwen3.5:9b GPU

Date: 2026-09-06  
Owner: Grok  
This report is **only** the 9B GPU production baseline plus scoring/alias corrections. Do not mix with the 8B CPU artifacts (`REPORT.md`, `suite.json`, `probe.json`).

JSON artifacts contain household records. Keep them on authorized machines; this handoff uses diagnostic summaries only.

## Provenance (Work 1 collection)

| Item | Value |
|---|---|
| Production host | `home-cortex-0` (`192.168.68.59`) |
| Host git | `780213290a22838a4712847a7e46229f19a2afac` (clean) |
| Container git | unavailable (cwd `/tmp`, no `.git`) |
| Imported package | `/tmp/tier1-bench/pkg/home_cortex/__init__.py` |
| Copied-package SHA256 | `6c37f94c26a3d92b1edffe542715554665c83b16f647e0f6b103e09c26f95444` (64 files) |
| Eval YAML SHA256 | `9b8b14a68f771e0cfd1e81211ca3d3b54868c1e884961383dafb8331fd100e01` |
| Schema tree SHA256 | `9d93fd024b005ec231a29db53a213046a2e57f954789516441bb57f72c49188f` |
| `/app/data` tree SHA256 | `5d065b8e4c99544412e068a5a8f7d619a92e8e8c94d308c45e7fa7bfd035142a` (10 files) |
| Image `/app/src` SHA256 | `d6213bf0cab98324aee7d3dd2cb9be0c663d53141c0d604ce32f0e724375d681` |
| Model | `qwen3.5:9b` digest `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` |
| Ollama | `0.32.13` |
| GPU | **100% GPU** (`ollama ps` `size == size_vram == 5490081790`, Q4_K_M) |
| Platform | Linux 6.8.0-138-generic x86_64 (container: linuxkit/glibc 2.41) |
| Settings | think=false, temperature=0, num_predict=384, keep_alive=24h, Tier-0 **disabled** |
| Backend | JSON graph over `/app/data` |
| Frozen time | `2026-09-03T12:00:00-07:00` |
| Speaker / household | `person:jian_kuang` / `address:fort_cerritos` |

The running application image was not deployed. Benchmarking used the isolated `/tmp/tier1-bench` package.

## Commands (Work 1, pre-scoring / pre-alias)

Package already present from the 9B probe. Suite used that same copy (no recopy, no scoring changes):

```bash
docker exec -e PYTHONPATH=/tmp/tier1-bench/pkg -w /tmp cortex-cortex-api-1 \
  python -m home_cortex.semantic_planner_benchmark \
    --ollama-url http://ollama:11434 \
    --model qwen3.5:9b \
    --data-dir /app/data \
    --schema-dir /tmp/tier1-bench/schemas/edge \
    --eval /tmp/tier1-bench/semantic_planner_eval.yaml \
    --output /tmp/tier1-artifacts/suite-qwen35-9b.json
```

20-question probe (already collected; first request is `first_request`, not verified-cold):

```bash
docker exec -e PYTHONPATH=/tmp/tier1-bench/pkg -w /tmp cortex-cortex-api-1 \
  python /tmp/tier1-bench/run_probe.py \
    --ollama-url http://ollama:11434 \
    --model qwen3.5:9b \
    --data-dir /app/data \
    --schema-dir /tmp/tier1-bench/schemas/edge \
    --eval /tmp/tier1-bench/semantic_planner_eval.yaml \
    --warmup 1 --repeat 5 \
    --output /tmp/tier1-artifacts/probe.json
```

## 119-case suite (original `/app/data`, original scorer)

Artifact: `artifacts/tier1-baseline/suite-qwen35-9b.json`

| Metric | Result |
|---|---|
| Plan accuracy | **92 / 119** (0.7731). Unscored **0 / 119**. |
| Answer correctness | **unscored 119 / 119**. Do not infer from `found`. |
| Executor | found 93, not_run 16, ambiguous 8, entity_not_found 2 |
| Planner latency | n=**119**, P50 **1888.503 ms**, P95 **2128.107 ms** |
| Tier-0 parity | 6/6 compared, equivalent |

### Plan accuracy by capability

| Capability | Correct / total | Rate |
|---|---|---|
| speaker_relative_reference | 9/9 | 1.0000 |
| entity_reference | 7/7 | 1.0000 |
| relationship_traversal | 16/17 | 0.9412 |
| filtering | 12/13 | 0.9231 |
| aggregation | 21/23 | 0.9130 |
| property_selection | 18/20 | 0.9000 |
| multi_hop_kinship | 3/7 | 0.4286 |
| temporal_operation | 6/15 | 0.4000 |
| relationship_property_lookup | 0/8 | 0.0000 |

Failure-reason counts: `UNKNOWN_PROPERTY` 15, `FILTER_MISMATCH` 8, `OPERATION_MISMATCH` 3, `MALFORMED_OUTPUT` 1.

### Plan failures (actual vs expected)

Planner / validation issues (unchanged planner behavior):

- **Marriage start (8) and duration (7):** expected `property_source=relationship` on `start_date`. Actual `property_source=entity` → `UNKNOWN_PROPERTY`, executor `not_run`.
- **`adult_count::咱家满十八岁的有几人`:** `MALFORMED_OUTPUT`, executor `not_run`.
- **`male_children::请找出我的男性子女`:** gender filter placed on the child path instead of request-level `filters` → `FILTER_MISMATCH` (executor still `found`).
- **Son birth date / countdown (4):** `self→child` without `gender=male` → `FILTER_MISMATCH`, executor `ambiguous`.
- **Father-in-law / wife-father (4):** parent path missing `gender=male` (and wife-father given-name used `resolve_reference`) → `FILTER_MISMATCH` / `OPERATION_MISMATCH`, executor `ambiguous`.
- **Pairwise older (2):** expected `argmin(self, other=spouse[female], birth_date)`. Actual `date_difference` or `select` on wife birth date → `OPERATION_MISMATCH`.

Execution-only (plan matched):

- **`德伦再过多久过生日` / `德伦再过多久过生日？`:** plan `named_entity=德伦` + `annual_occurrence` matched. Executor **`entity_not_found`**. Full name `匡德伦是谁` matched and **found**.

## 20-question probe (original `/app/data`, original scorer)

Artifact: `artifacts/tier1-baseline/probe-qwen35-9b.json`  
Preserve original scores. First request is not verified-cold.

| Metric | First measured pass | All measured samples |
|---|---|---|
| Plan accuracy | **14 / 20** (0.70) | **70 / 100** |
| Answer correctness | **12 / 20** (0.60) | **60 / 100** |
| Latency | n=100, P50 **1933 ms**, P95 **2142 ms** | same |

Executor (first pass): found 13, not_run 3, ambiguous 3, entity_not_found 1.

Original plan-correct / answer-wrong:

1. `daughter_given_name_jian` — identity/display-name plan accepted; stored names `["Evelyn Kuang","匡悠然"]` for `person:evelyn_kuang`; expected_value was exactly `"Evelyn"`; `failure_stage=null`.
2. `named_dylan_birthday_countdown` — plan matched; `entity_not_found`.

Other first-pass failures (planner): marriage start/duration `UNKNOWN_PROPERTY`; pairwise `INVALID_PLAN`; 爱人的爸爸 / 妻子的父亲 missing male filter (`ambiguous`); son countdown missing male filter (`ambiguous`).

## Work 2 — daughter-name scoring (evaluation correction)

Intended meaning of `我的女儿叫什么`: the daughter's name for speaker Jian. Accepted plans already included `daughter_given_name` and `daughter_identity`. The result contract is now the same: `person:evelyn_kuang` plus an appropriate stored name.

Scoring revision: `2026-09-06.1-daughter-names-answer-mismatch`.

- `expected_value: Evelyn` still accepts given-name/first-name.
- `expected_names: [Evelyn, Evelyn Kuang, 匡悠然]` accepts identity/display-name output whose names are a subset of that stored set.
- Wrong person, unrelated names, missing values, and non-`found` statuses still fail.
- `failure_stage=answer_mismatch` when a scored answer is incorrect and no earlier stage applies.

Rescored original structured results (not a model improvement):

| | Original | Rescored |
|---|---|---|
| First-pass answer | 12/20 | **13/20** |
| All-measured answer | 60/100 | **65/100** |
| Plan | 14/20 and 70/100 | unchanged |
| Daughter `failure_stage` | `null` | `null` (now correct) |

Artifact: `artifacts/tier1-baseline/probe-qwen35-9b-rescored.json`

## Work 3 — 德伦 resolution

**Root cause: production JSON data, not planner/code/ingestion path.**

`record_aliases()` already indexes `name`, `aliases`, and `first_name`. Production `/app/data/nodes/person.json` (and the host copy) stores dylan as `name: ["Dylan Kuang","匡德伦"]` with **no `aliases`**. Local workspace data already has `aliases: ["Dylan","德伦"]`.

Deterministic reproduction (no LLM):

- Production-shaped record: `匡德伦` → `person:dylan_kuang`; `德伦` → `entity_not_found`.
- With stored aliases: both resolve to `person:dylan_kuang`; `annual_occurrence` value **57**.
- Unknown aliases stay unresolved; duplicate aliases stay ambiguous; speaker/household appellation scoping unchanged.

Smallest correction: store `aliases: ["Dylan", "德伦"]` on `person:dylan_kuang` using the existing aliases field. No household-specific Python rule, inferred kinship, or fuzzy match.

**Live data was not changed.** `/app/data` dylan `aliases` is still `null`. Required live change if later approved: add that `aliases` list on the dylan person record (JSON and any ingested SurrealDB copy). `大宝` appellation is not required for `德伦`.

Staged overlay only: `artifacts/tier1-baseline/staged-data-dylan-aliases/overlay.json`, applied under `/tmp/tier1-bench/data` for post-change benches.

## Post-change production validation

Isolated package recopy + staged `/tmp/tier1-bench/data` (dylan `aliases: ["Dylan","德伦"]`). Live `/app/data` and host `data/nodes/person.json` still have dylan `aliases: null`.

Copied-package SHA256 after scoring/alias tooling: `7b90ab9326bb313eb54a415267f6a9867d64dc829d771e279f63326250fa35e1`. GPU still `100% GPU`. Scoring revision `2026-09-06.1-daughter-names-answer-mismatch`.

```bash
docker exec -e PYTHONPATH=/tmp/tier1-bench/pkg \
  -e HOST_GIT_COMMIT=780213290a22838a4712847a7e46229f19a2afac \
  -e HOST_GIT_DIRTY=false \
  -e COPIED_PACKAGE_SHA256=7b90ab9326bb313eb54a415267f6a9867d64dc829d771e279f63326250fa35e1 \
  -w /tmp cortex-cortex-api-1 \
  python /tmp/tier1-bench/run_probe.py \
    --ollama-url http://ollama:11434 --model qwen3.5:9b \
    --data-dir /tmp/tier1-bench/data \
    --schema-dir /tmp/tier1-bench/schemas/edge \
    --eval /tmp/tier1-bench/semantic_planner_eval.yaml \
    --warmup 1 --repeat 5 \
    --output /tmp/tier1-artifacts/probe-qwen35-9b-after.json

docker exec -e PYTHONPATH=/tmp/tier1-bench/pkg \
  -e HOST_GIT_COMMIT=780213290a22838a4712847a7e46229f19a2afac \
  -e HOST_GIT_DIRTY=false \
  -e COPIED_PACKAGE_SHA256=7b90ab9326bb313eb54a415267f6a9867d64dc829d771e279f63326250fa35e1 \
  -w /tmp cortex-cortex-api-1 \
  python -m home_cortex.semantic_planner_benchmark \
    --ollama-url http://ollama:11434 --model qwen3.5:9b \
    --data-dir /tmp/tier1-bench/data \
    --schema-dir /tmp/tier1-bench/schemas/edge \
    --eval /tmp/tier1-bench/semantic_planner_eval.yaml \
    --output /tmp/tier1-artifacts/suite-qwen35-9b-after.json
```

Artifacts: `probe-qwen35-9b-after.json`, `suite-qwen35-9b-after.json`.

### Probe before / after

| | Original 9B (`/app/data`, old scorer) | Rescored original (eval correction) | After (staged aliases + new scorer) |
|---|---|---|---|
| Plan first pass | 14/20 | 14/20 | **15/20** |
| Plan all measured | 70/100 | 70/100 | **75/100** |
| Answer first pass | 12/20 | 13/20 | **15/20** |
| Answer all measured | 60/100 | 65/100 | **75/100** |
| P50 / P95 ms | 1933 / 2142 | same run | 1922 / 2085 |
| `entity_not_found` | 1 | 1 | **0** |

Split of the answer change 12 → 15 on the first measured pass:

| Case | Kind | What changed |
|---|---|---|
| `daughter_given_name_jian` | **evaluation correction** | Same identity/display-name result `["Evelyn Kuang","匡悠然"]` for `person:evelyn_kuang`. Now scored true. Plan already matched. |
| `named_dylan_birthday_countdown` | **alias/data fix** | Plan already matched. Executor `entity_not_found` → `found` value **57**. |
| `father_in_law_lover_dad` | **planner variance, not this ticket** | Original missing `gender=male` (`ambiguous`). After run emitted the male filter and found `person:zhigang_ba`. Prompts/ontology/validation were not changed. |

Remaining after-probe planner failures (unchanged class): marriage start/duration `UNKNOWN_PROPERTY`; pairwise `INVALID_PLAN`; `wife_father_given_name` and `son_birthday_countdown` missing male filters (`ambiguous`).

### Suite before / after

| | Original 9B `/app/data` | After staged aliases |
|---|---|---|
| Plan | **92 / 119** (0.7731) | **95 / 119** (0.7983) |
| Answer | unscored 119/119 | unscored 119/119 |
| Executor | found 93, not_run 16, ambiguous 8, entity_not_found **2** | found 96, not_run 18, ambiguous 5, entity_not_found **0** |
| Latency n/P50/P95 | 119 / 1888.503 / 2128.107 | 119 / 1899.569 / 2114.480 |

`德伦` countdown (both punctuated and unpunctuated): plan still matched; executor `entity_not_found` → `found`. That is the alias fix.

Suite plan 92→95 is **not** the alias fix (those two 德伦 plans already matched). Extra matches were kinship/property-selection variance (`FILTER_MISMATCH` 8→5; multi_hop 3/7→5/7; property_selection 18/20→19/20). Marriage `UNKNOWN_PROPERTY` 15/15 unchanged.

Do not treat 92→95 or 14→15 plan as a model improvement from this ticket.

## Local tests

`.venv/bin/python -m pytest -q` → **332 passed** (was 318 before this ticket’s tests).

## Changed files

- `src/home_cortex/semantic_planner_benchmark.py` — names scoring, `answer_mismatch`, rescoring, provenance fingerprints. No planner prompt/ontology/normalization/validation changes.
- `benchmarks/semantic_planner_eval.yaml` — daughter `expected_names` + contract note.
- `scripts/tier1_latency_bench.py` — pass data/schema dirs into provenance.
- `scripts/copy_tier1_bench_into_api.sh` — replace copied trees; strip macOS AppleDouble files.
- `tests/test_semantic_planner_benchmark.py` — accepted/rejected daughter results; answer-mismatch.
- `tests/test_entity_alias_resolution.py` — full name vs alias, unknown, duplicate, scoping.
- `tests/test_semantic_facts.py` — engine-level 德伦 with/without stored alias.
- `artifacts/tier1-baseline/staged-data-dylan-aliases/overlay.json`

## Unresolved decisions

1. Apply dylan `aliases: ["Dylan","德伦"]` to live `/app/data` and SurrealDB? Not done here.
2. Planner remaining failures (marriage relationship property, male filters on son/father-in-law, pairwise older) are Astra’s planner work; this ticket did not change them.
3. `given_name` still maps to deployed `first_name`; documented, not renamed.

## Astra handoff

Work 1 baseline is complete and reproducible from `/tmp/tier1-bench` + `/app/data` + `qwen3.5:9b`. Original suite plan **92/119**; answers unscored. Original probe plan **70/100**, answer **60/100**, P50 **1933 ms**, P95 **2142 ms**, 100% GPU.

Scoring now agrees with accepted identity/display-name plans. Rescore of the saved probe is an evaluation correction: answer **65/100**. After staged aliases, `德伦` executes to 57; probe answer **75/100** on the after artifact mixes that data fix with one planner-variance kinship pass. Remaining planner failures are marriage `property_source=entity`, pairwise older, and missing male filters on son/wife-father.

Do not apply the dylan alias to live `/app/data` from this ticket. Do not treat suite 92→95 or probe plan 14→15 as a planner improvement from this work.
