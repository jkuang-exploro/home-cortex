# Warm-load / latency redo — `qwen3.5:4b` — 2026-09-08

## Verdict

On the current production host the warm-request **0.7 s `load_duration` is gone**.
Resident `qwen3.5:4b` reports **~0.5–1.8 ms** load on chat/generate. Isolated
Cortex matches direct Ollama.

That drop is **not** because 4B is smaller. The host also moved from Ollama
**0.32.13 → 0.32.15**, which caches resolved model metadata and removes the
duplicate `GetModel()` on the inference path (PR #17752). Uncached `/api/show`
is still **~375 ms**, so GGUF metadata parse is still expensive when the cache
is bypassed.

Planner-matched wall time is **586 ms** vs **1,476 ms** on the earlier 9B /
0.32.13 run. Almost all of the 0.7 s load stage disappeared; generation of 38
tokens is also faster (**452 vs 678 ms**).

The remaining Cortex fix is unchanged: send `num_ctx=8192` on ordinary chat,
not only the planner. 4B still restarted llama-server between 4096 and 8192
this morning.

## Environment

| Item | This run | Prior 9B run |
|---|---|---|
| Host | `jkuang@192.168.68.59` | same |
| GPU | RTX 2060 SUPER, 4066 MiB used / 8192 | 6518 MiB used |
| Ollama | **0.32.15** | 0.32.13 |
| Model | `qwen3.5:4b` digest `2a654d98e6fb…`, 4.7B Q4_K_M | `qwen3.5:9b` `6488c96fa5fa…` |
| Resident | PID 968, port 36499, `-c 8192 -np 1 -b 1024`, 3.3 GB VRAM, keep-alive 24h | `-c 8192 -np 1 -b 512`, 5.6 GB |
| Cortex `OLLAMA_MODEL` | `qwen3.5:4b` | `qwen3.5:9b` |
| Isolated package | `/tmp/hc-warm-load-4b` in `cortex-cortex-api-1` | `/tmp/hc-warm-load` |
| Probe SHA-256 | `000f9562f52497baf3452f91b474aa99ab73d632b2e1ae178ea47f99b68b451a` | previous script |
| Tree SHA-256 | `3be0580b79ca91ed0a0ffc5471ef9b2fe9b223e580d8916f107a0612a59c2e2c` | — |
| Git | `bd161e0dc5c9e80235e7f715af06c6db3c0ec6e2` | `baeaf013…` |
| Run | 2026-09-08T09:31:01Z–09:31:30Z, 28.2 s | ~113 s |

Constraints: no unload, no mismatched `num_ctx`, no settings change. All
inference used `num_ctx=8192` and `keep_alive=24h`. Zero `starting llama-server`
events during the probe. Digest and context unchanged after.

Two variables changed versus the 9B report (model **and** Ollama version).
Do not treat the load collapse as a 4B-only effect. 4B was not re-run on
0.32.13 (that image is gone); 9B was not re-run on 0.32.15 (would unload 4B).

## Warm measurements

One warmup discarded. Sequential requests from the API container. Times in ms.

| Experiment | n | Request bytes | `load_duration` mean / p50 / p95 | Wall mean | Prefill mean | Generation mean |
|---|---:|---:|---|---:|---:|---:|
| Cached `/api/show` | 15 | 22 | n/a | 1.4 | — | — |
| Uncached `/api/show` | 15 | 78 | n/a | **375.3 / 371.1 / 398.0** | — | — |
| Tiny generate `num_predict=1` | 15 | 180 | **0.5 / 0.5 / 0.6** | 62.9 | 60.6 | 0 |
| Tiny chat `num_predict=1` | 15 | 210 | **0.5 / 0.6 / 0.7** | 60.8 | 58.7 | 0 |
| Tiny chat + format | 15 | 6,769 | 0.7 / 0.7 / 0.8 | 87.7 | 81.7 | 0 |
| Large chat, no format, `num_predict=1` | 15 | 25,236 | 1.0 / 1.0 / 1.3 | 104.4 | 60.0 | 0 |
| Large chat + format, `num_predict=1` | 15 | 31,795 | 1.2 / 1.1 / 1.8 | 114.3 | 65.6 | 0 |
| Planner-matched `num_predict=384` | 8 | 31,797 | **1.6 / 1.6 / 1.8** | **586.3** | 60.7 | **451.5** (38 tok) |
| Tiny chat, new TCP | 15 | 210 | 0.6 / 0.5 / 1.0 | 61.8 | 59.0 | 0 |
| Isolated Cortex planner | 8 | same body | **1.5 / 1.7 / 1.7** | **587.1** | 60.7 | 451.5 |

Planner payload: 52 messages, 25,086 message bytes, 6,549 format bytes, 4,984
input tokens, SHA-256 `8bc00c542a24dbe1dc5d678a5fd6a5db831798a0024d84a3a36971bfea5fae91`.
Utterance `Who am I?` on the synthetic fixture (`entity_not_found`; load probe,
not accuracy). Prompt is slightly larger than the 9B run (48 messages / 4,617
tokens) because the isolated package is current source.

llama-server `/health` on 36499, 10 samples: **1.74–1.80 ms**. Warm load
(~1.6 ms) is now the same order as that ping plus JSON bind.

Planner residual (wall − load − prefill − generation) remains **~70–74 ms**
(template/tokenize after `scheduleRunner`). That residual did not shrink with
the metadata cache.

## Comparison with 9B / 0.32.13

Same host, same 8192 keep-alive, same isolated Cortex path. Not a single-variable
A/B.

| Stage (planner-matched means) | 9B / 0.32.13 | 4B / 0.32.15 | Comment |
|---|---:|---:|---|
| `load_duration` | 658 ms | **1.6 ms** | 0.32.15 metadata cache (PR #17752). Uncached show still ~375–405 ms on both |
| Prefill (cached prefix) | 70 ms | 61 ms | Prefix cache; not a model-size claim |
| Generation, 38 tokens | 678 ms (17.8 ms/tok) | **452 ms (11.9 ms/tok)** | Smaller model; demonstrated |
| Cortex vs direct load | 675 vs 658 | 1.5 vs 1.6 | Cortex still adds nothing material |
| Planner wall | 1,476 ms | **586 ms** | Load gone + faster decode |

Ollama 0.32.15 release notes: “Caches resolved model metadata between requests,
cutting time-to-first-token by roughly half.” PR #17752: previously every
chat/generate re-read GGUF metadata (~300 ms) multiple times; the cache plus
removing duplicate `GetModel()` from scheduling is the mechanism that matches
this host.

## 4096 vs 8192 still happens on 4B

Ordinary Cortex chat still omits `num_ctx`. Production logs after the model
swap:

```text
2026-09-08T09:07:05Z  starting llama-server ... qwen3.5:4b ... -c 4096 -b 512
2026-09-08T09:08:00Z  starting llama-server ... qwen3.5:4b ... -c 8192 -b 1024
```

Same `needsReload` thrash as 9B. 4B starts faster (~2.8 s) than 9B (~2.94 s)
but it is still a full runner restart, not the warm 1 ms path.

## What needs to be fixed

1. **Still required in Cortex:** pass `num_ctx=8192` (and `keep_alive=24h` if a
   keep-alive is sent) on `chat`, `chat_with_tools`, and `stream_chat_with_tools`.
   Hand to Codex. Rollback: omit `num_ctx` again. Do not send a different
   context than the resident 8192 runner.
2. **Warm 0.7 s load:** already gone on this deployment via Ollama 0.32.15.
   No Cortex change. Do not attribute that saving to the 4B swap alone.
3. **Remaining planner wall (~586 ms):** generation (~452 ms) plus ~61 ms
   cached prefill plus ~73 ms post-schedule residual. Further decode gains
   would be model/runtime, not Python.

## Reproduction

```sh
docker exec -e PYTHONPATH=/tmp/hc-warm-load-4b/src -e PYTHONHASHSEED=0 \
  cortex-cortex-api-1 \
  python /tmp/hc-warm-load-4b/scripts/ollama_warm_load_probe.py \
    --ollama-url http://ollama:11434 \
    --isolated-root /tmp/hc-warm-load-4b \
    --model qwen3.5:4b --num-ctx 8192 \
    --repeat 15 --warmup 1 --planner-repeat 8 \
    --utterance "Who am I?" \
    --output /tmp/hc-warm-load-4b/probe-results.json
```

The script refuses to start unless `qwen3.5:4b` is resident at 8192.

Distributions: `artifacts/ollama-warm-load/probe-summary-qwen35-4b.json`.
Prior 9B analysis remains in `REPORT.md` / `probe-summary.json`.
