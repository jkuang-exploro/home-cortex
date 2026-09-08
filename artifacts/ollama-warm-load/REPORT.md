# Ollama warm-request `load_duration` — 2026-09-08

## Verdict

Ollama 0.32.13’s `load_duration` is **not weight loading**. On a warm,
resident `qwen3.5:9b` runner it is the wall interval from request start until
`scheduleRunner` returns: JSON bind, **uncached `GetModel` / capability GGUF
metadata parsing**, runner reuse checks, and a cheap llama-server `/health`
ping.

That interval is **~0.66–0.81 s per call** while VRAM, runner PID, and
`num_ctx=8192` stay fixed. GPU utilization is 0% and GPU memory is unchanged
during it. Direct Ollama and the isolated Cortex planner report the same load
stage.

**There is no demonstrated Cortex-side way to remove those ~700 ms.** The
smallest justified Cortex change is different: send `num_ctx=8192` on every
Ollama call, not only the planner, so production stops restarting
llama-server between 4096 and 8192 (~3 s real reloads). That does not reduce
the warm 0.7 s figure. Hand that implementation to Codex; do not upgrade
Ollama without authorization.

## What this version puts in `load_duration`

Ollama 0.32.13 `server/routes.go`:

```text
checkpointStart := time.Now()
// bind JSON, parse name, GetModel, capability checks
r, m, opts, err := s.scheduleRunner(...)   // GetModel again + getRunner
checkpointLoaded := time.Now()
// template / tokenize / completion happen AFTER this
res.LoadDuration = checkpointLoaded.Sub(checkpointStart)
```

`scheduleRunner` always calls `GetModel`. `GetModel` opens the GGUF and reads
`tokenizer.chat_template`. `Capabilities()` / `CheckCapabilities()` open it
again. `getRunner` reuses a resident runner unless `needsReload` is true;
that path DeepEquals runner options and `Ping()`s llama-server `/health`.

Prompt evaluation and token generation are `prompt_eval_duration` and
`eval_duration`. They are not inside `load_duration`.

API docs calling this “time spent loading the model” are misleading for a
warm runner.

## Environment

| Item | Value |
|---|---|
| Host | `jkuang@192.168.68.59` (`home-cortex-0`) |
| GPU | RTX 2060 SUPER, 8192 MiB |
| Ollama | 0.32.13 in `cortex-ollama-1` |
| Model | `qwen3.5:9b` digest `6488c96fa5fa…`, family `qwen35`, Q4_K_M, 9.7B |
| Resident runner | PID 2095, port 45603, `-c 8192 -np 1`, 100% GPU, 5.6 GB VRAM, keep-alive 24h |
| Architecture note | `qwen35` forces `num_parallel=1` |
| VRAM-tier default `num_ctx` | 4096 (total VRAM &lt; 23 GiB) |
| Probe package | isolated `/tmp/hc-warm-load` in `cortex-cortex-api-1` |
| Fixture | `benchmarks/fixtures/semantic-contract` |
| Git | `baeaf013374d4722854f2c4425cdba3b148aa324` |
| Probe script SHA-256 | `f5630382ee7b231c5f0cce80cca82cbcac88981a37c9d0b75c7612f28fdca233` |
| Isolated tree SHA-256 | `2be3682126907a7953ed0dd0576cdae0a34e08d7d46fbbea6286e08a2b8a5da8` (45 files) |
| Planner request SHA-256 | `50310fde521fbb49fd993b1e1bbee6dda7b7ef6325f36be1c67f4fc70345c1fc` |

Constraints honored: no `ollama stop`, no `keep_alive=0`, no mismatched
`num_ctx` in probes, no deployment/settings/upgrade changes. All inference
calls used `num_ctx=8192` and `keep_alive=24h`.

## Warm measurements

One warmup discarded per series. Sequential requests from the API container
to `http://ollama:11434`. No llama-server restart during the measured run
(`starting llama-server` count = 0). Resident digest and 8192 context were
unchanged. GPU memory stayed 6518 MiB. Client traffic was only Cortex-network
`172.21.0.5`.

Times are milliseconds.

| Experiment | n | Request bytes | `load_duration` mean / p50 / p95 | Wall mean | Notes |
|---|---:|---:|---|---:|---|
| Cached `/api/show` | 15 | 22 | n/a (not an inference metric) | 1.3 | Show cache hit |
| Uncached `/api/show` (`system` + `options` bypass) | 15 | 78 | n/a | **404.6 / 410.9 / 431.2** | Same GGUF metadata work as `GetModelInfo` |
| Tiny `/api/generate` `num_predict=1` | 15 | 180 | **807.8 / 813.1 / 844.2** | 886.4 | 19 prompt tokens |
| Tiny `/api/chat` `num_predict=1` | 15 | 210 | **697.0 / 704.7 / 731.2** | 775.8 | |
| Tiny chat + planner format schema | 15 | 6,769 | 671.0 / 668.9 / 710.9 | 776.0 | Format does not move load |
| Large chat, no format, `num_predict=1` | 15 | 23,340 | 690.3 / 699.3 / 730.7 | 831.0 | 4,617 prompt tokens |
| Large chat + format, `num_predict=1` | 15 | 29,899 | 697.9 / 688.6 / 735.9 | 845.1 | Matched planner bytes |
| Planner-matched chat `num_predict=384` | 8 | 29,901 | **657.5 / 661.1 / 710.3** | 1,476.0 | 38 output tokens |
| Tiny chat, new TCP each call | 15 | 210 | 692.9 / 690.8 / 721.5 | 772.0 | Connection reuse is not the 0.7 s |
| Isolated Cortex planner | 8 | same planner body | **674.5 / 674.7 / 688.8** | 1,495.5 | One model call/request |

Planner payload: 48 messages, 23,190 message bytes, 6,549 format bytes,
`think=false`, temperature/seed 0, `num_ctx=8192`, `keep_alive=24h`.
Utterance `Who am I?` against the synthetic fixture (structured status
`entity_not_found`; this is a load probe, not an accuracy claim).

llama-server `/health` on port 45603, 20 samples from inside
`cortex-ollama-1`: **1.74–2.18 ms**, body `{"status":"ok"}`. Ping is not the
700 ms.

50 ms GPU samples around one warm tiny chat (`load_duration` 717 ms, wall
1032 ms): **utilization 0%, memory 6518 MiB on every sample**. Prefill is
only ~78 ms and can sit between samples; the load interval is not GPU work.

Host `nvidia-smi dmon` over the broader window: mostly 0% SM, short 4–9%
pulses, occasional ~64% during longer generations. Not a reload signature.

## What does *not* explain the 0.7 s

Changed one variable at a time against a warm 8192 runner:

- **Prompt size:** 210 vs 29,901 bytes, load stays ~0.67–0.70 s.
- **Format schema:** on vs off, load unchanged.
- **`num_predict`:** 1 vs 384, load unchanged (generation 0 vs 678 ms).
- **Generate vs chat:** both ~0.7–0.8 s load.
- **HTTP connection reuse vs new TCP:** no load change.
- **Cortex pipeline vs direct Ollama:** load 675 vs 658 ms. Cortex is not
  adding the stage.
- **Client construction:** both paths reuse one process-local client.

The original token-latency audit’s 752.6 ms/request load on 38 planner calls
is the same stage. This run’s planner-matched mean is 658 ms; the difference
is sample and KV-cache state, not a new mechanism.

Prefill in this sequential identical-prompt series is ~70 ms for 4,617
tokens because llama-server prefix-cached almost the entire prompt. The
audit’s ~450 ms prefill was a different cache mix. Do not treat this 70 ms
as a prefill speedup claim.

## Cold vs warm (not reproduced here)

Verified cold load was **not** run (would unload the shared model).

Production logs from 07:38–07:39 UTC, same host/version/model, show what a
real runner restart looks like:

```text
starting llama-server ... -c 4096 ...
llama-server started in 2.94 seconds
GIN POST /api/chat  8.33 s

starting llama-server ... -c 8192 ...
llama-server started in 2.94 seconds
GIN POST /api/chat 13.26 s
```

That path logs `template selection` + `loaded runners` and changes the
llama-server PID/port. Warm probes did none of that. Cold/restart cost is
~3 s of process start plus inference, not the 0.7 s warm `load_duration`.

## Related production confound: 4096 vs 8192

`OllamaService.plan_semantic_fact` sends `num_ctx=8192`. Ordinary
`chat` / `chat_with_tools` / `stream_chat_with_tools` send no `num_ctx`.
Ollama then uses the VRAM-tier default **4096**. `needsReload` compares
runner options, expires the 8192 runner, and starts another llama-server.

Observed in live Cortex traffic (`172.21.0.5`) immediately before this
probe: alternating `-c 4096` and `-c 8192` restarts, 8–13 s chats, then
stable ~2 s chats once 8192 stayed loaded.

This is **not** the warm 0.7 s. It is a larger, demonstrated extra delay
when planner and ordinary chat interleave. qwen3.5 cannot run those two
contexts in parallel (`num_parallel` forced to 1).

## Remaining uncertainty

Not instrumented inside the Ollama process:

- Exact split of the ~700 ms among `GetModel` #1, `Capabilities` GGUF
  opens, `GetModel` #2, and leftover schedule work. Uncached `/api/show`
  is **405 ms** for one metadata pass; two such passes on the chat path
  account for the load interval to within ~100 ms. That is circumstantial,
  not a profiler trace.
- Whether `gguf.Open` is dominated by tokenizer-array parse vs disk. Page
  cache was warm (`/api/show` cache hits at 1 ms; uncached still 405 ms),
  so the expensive part is metadata parsing, not cold I/O.

## Demonstrated savings vs hypothetical

| Change | Status | Effect |
|---|---|---|
| Reuse the HTTP client / TCP connection | Measured | None on load (~693 vs 697 ms) |
| Drop the JSON format schema | Measured | None on load |
| Shrink the prompt | Measured | None on load |
| Isolated Cortex vs direct Ollama | Measured | None on load (~675 vs 658 ms) |
| Send `num_ctx=8192` on **all** Cortex Ollama calls | Demonstrated in production logs; **not** applied | Avoids ~3 s llama-server restarts when planner and ordinary chat alternate. Does **not** remove the warm 0.7 s |
| Cache `GetModel` / skip GGUF re-parse in Ollama | Hypothetical, needs Ollama change or upgrade | Upper bound ≈ measured warm `load_duration` (0.66–0.81 s). Not available without authorization to upgrade |

## Recommendation

**Do not change Cortex to chase the 0.7 s.** It is Ollama 0.32.13 request
setup, not a Cortex client bug.

**Smallest justified Cortex change (for Codex, not done here):** pass the
same runner options the planner already uses (`num_ctx=8192`, and
`keep_alive=24h` if a keep-alive is sent at all) on `chat`,
`chat_with_tools`, and `stream_chat_with_tools`.

- Benefit: stop 4096/8192 runner ping-pong. Production evidence is the
  07:38 UTC restart pair, ~3 s `llama-server started` plus 8–13 s HTTP.
- Risk: ordinary chat then always occupies the 8192 KV budget already used
  by the planner. On this 8 GB card that is the resident configuration.
  Do not send a *different* `num_ctx` than 8192; that would itself reload.
- Rollback: revert those three call sites to omit `num_ctx`.
- Out of scope without authorization: Ollama upgrade, `OLLAMA_*` env
  changes, unloading, or a second runner.

## Reproduction

From a checkout, copy an isolated package into the API container. Do **not**
omit `num_ctx` and do **not** send `keep_alive=0`.

```sh
# on the GPU host, isolated tree already used:
#   /tmp/hc-warm-load  inside cortex-cortex-api-1
#   PYTHONPATH=/tmp/hc-warm-load/src PYTHONHASHSEED=0

docker exec -e PYTHONPATH=/tmp/hc-warm-load/src -e PYTHONHASHSEED=0 \
  cortex-cortex-api-1 \
  python /tmp/hc-warm-load/scripts/ollama_warm_load_probe.py \
    --ollama-url http://ollama:11434 \
    --isolated-root /tmp/hc-warm-load \
    --repeat 15 --warmup 1 --planner-repeat 8 \
    --utterance "Who am I?" \
    --output /tmp/hc-warm-load/probe-results.json
```

The script refuses to start if `qwen3.5:9b` is not resident or if resident
context is not 8192. It aborts if a later snapshot shows unload or a
context change.

Safety checks used here:

```sh
docker exec cortex-ollama-1 ollama ps
docker logs --since 2026-09-08T07:58:32 --until 2026-09-08T08:00:26 cortex-ollama-1 2>&1 \
  | grep -c "starting llama-server"
```

Distributions without per-call rows: `artifacts/ollama-warm-load/probe-summary.json`.
Full per-call JSON remains in the isolated GPU directory
`/tmp/hc-warm-load/probe-results.json` (80 KiB).
