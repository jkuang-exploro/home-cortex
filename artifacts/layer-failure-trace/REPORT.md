# Layer traces for reported failures — 2026-09-08

Investigation only. Prompts, ontology, executor, and scoring were not changed.

## Production snapshot

Confirmed on `jkuang@home-cortex-0` before the run. Private conversations and production graph were **not** read.

| Item | Value |
|---|---|
| Host | `home-cortex-0` |
| Model | `qwen3.5:9b` digest `6488c96fa5fa…`, 100% GPU, context 8192, 5.6 GB VRAM |
| Ollama | 0.32.15 |
| Cortex `OLLAMA_MODEL` | `qwen3.5:9b` |
| Deployed `ollama.py` | `bc004e6f6b158d10ace48ba62fd3a33ff8f4f0ab975cea1d99b00fd77650d679` (wife-countdown example present) |
| Host git | `9c8135e` |
| Isolated package | `/tmp/hc-layer-trace`, tree `17e7445bbd39945f…` |
| Probe | `scripts/layer_failure_trace.py` `4995eba0ed8d3ebc…` |
| Ontology | `2aea40ef0569a0ba…` |
| Data | synthetic `benchmarks/fixtures/semantic-contract`, speaker `person:a`, home `address:fictional` |
| Clock | `2026-09-03T12:00:00-07:00` |
| Inference | think=false, temperature/seed 0, num_ctx=8192, num_predict=384, keep_alive=24h |
| Prompt SHA-256 | `6d7ff35a9dd14bde9a7921e5cd2364875a5bf60ba13fe5a5afb012ec571b7e13` |
| Repeats | 3 standalone + 3 multi-turn per case; 1 discarded warmup |
| Runner restarts | 0 |

Scoring: exact expanded request vs expected IR. `found` and a matching number are **not** semantic correctness.

## Attributions

| Failure | Standalone | Multi-turn | Layer | Evidence |
|---|---|---|---|---|
| `家里都有谁` | 3/3 correct | n/a | none | `select current_household→member` |
| `我家里都有谁` | **0/3** | n/a | **interpretation** then **validation** | Raw IR is `select self→member`. `member` is an address-scoped concept; plan is `INVALID_PLAN`. Render: unsupported query. Not a list of people. |
| `家里有几个男的` / `有几个男的` / `家里有几个女的` | **0/3** each | n/a | **interpretation** | Raw filters are `adult` **and** `minor`, **no gender**. Count is 0. Render “零位成年人”. Found is not correctness. |
| After `家里有几个成年人` → `家里有几个男的` | first 3/3 | **0/3** | **interpretation** (prior-turn leak) | Second turn keeps `adult` (duplicated), still no gender. Render “五位成年人”. Same count 5 as adult-count is an accidental number, different people than male-count gold. |
| `什么时候过生日` (self / wife / daughter) standalone | **0/3** | self 0/3; wife/daughter **3/3** in the 8-turn sequence | **interpretation** | Standalone compiles `select(birth_date)` instead of `annual_occurrence`. Right person, wrong operation. Later turns in the reported sequence do countdown correctly. |
| `我岳父是谁` after `我父亲是谁` | standalone **3/3** | **0/3** | **interpretation** (prior-turn leak) | Raw path `father` then `spouse` (speaker’s father’s spouse), not `father_in_law` (`spouse` then `father`). Executor `relationship_not_found` on that wrong path. Isolated `岳父` then `老婆过生日` **without** the father turn is 3/3. |
| Reported `配偶的女儿` for wife birthday | standalone: wife person correct, operation wrong | 8-turn and 岳父→老婆: **3/3** wife countdown | **not reproduced** on this 9B isolated run | No `spouse+daughter` IR in 9 traces of `我老婆什么时候过生日`. Cause of the original live answer remains **unknown** (model/session/data may differ; no production log). |
| Gender conditions omitted in text | n/a | n/a | **rendering** (gold path) | Expected male-count plan executes to 5, gold render is “家里目前有五个人” — no 男. `_count_noun` only special-cases `adult`/`minor`, not gender. Not observed on actual gender-count IR because interpretation never produced gender filters. |
| `我父亲是谁` on this fixture | IR 3/3 `father` | IR 3/3 | not the live bug | Synthetic graph has no `parent_of` into `person:a`, so executor `relationship_not_found` matches gold. Renderer says 父母 not 父亲. Production “匡洪杰” was not re-checked (private graph not read). |

## Notes

- Executor followed whatever IR it was given. No case had a correct plan and a wrong graph result.
- Renderer of a **wrong** plan (e.g. “零位成年人”) is not a rendering-layer cause.
- `我家里` vs `家里` is a real interpreter split: leading 我 moves the subject from `current_household` to `self` and attaches `member`, which the schema rejects.
- Gender-count **does** incorrectly introduce `adult` (and `minor`). After an adult-count turn it **only** leaks `adult`.
- The 8-turn live sequence’s wife-as-daughter error did not appear here. The failure that **did** appear in that sequence is 岳父 after 父亲.

## Reproduction

```sh
# isolated tree already used: /tmp/hc-layer-trace on home-cortex-0
docker exec -e PYTHONPATH=/tmp/hc-layer-trace/src -e PYTHONHASHSEED=0 \
  -w /tmp/hc-layer-trace cortex-cortex-api-1 \
  python /tmp/hc-layer-trace/scripts/layer_failure_trace.py run \
    --ollama-url http://ollama:11434 --model qwen3.5:9b \
    --output /tmp/hc-layer-trace/probe-results.json

PYTHONPATH=src python scripts/layer_failure_trace.py summarize \
  --results /tmp/hc-layer-trace/probe-results.json \
  --output artifacts/layer-failure-trace/probe-summary.json
```

Requires resident `qwen3.5:9b` at 8192. Full per-call JSON stays in `/tmp/hc-layer-trace/probe-results.json`.
