# Static-prefix reuse investigation — 2026-09-08

## Implementation and confirmed invalidations

`planner_system_prompt` now serializes capability object keys canonically with
`sort_keys=True`. Semantic array order, values, examples, instructions, clock
precision, discourse, ownership checks, and execution remain unchanged.

The original `semantic_relation_properties` map is populated by iterating a set.
Its JSON ordering differs under `PYTHONHASHSEED=0` and `1`. Because the capabilities
are cached within a schema instance, this is a **cross-process** instability,
not evidence of per-turn reshuffling in one process. Canonical serialization
also covers nested maps built from differently ordered registries. Tests confirm
identical final request hashes under both seeds and preserve array ordering.

The existing layout already places one invariant system message plus 22 example
pairs (45 messages) before the clock, user history and identity reminders. No
prompt relocation was justified. Across different conversations, over 23 KB of
serialized message text remains shared. Across clock changes, approximately
22.8 KB remains shared. These are byte/message comparisons, **not tokenizer or
KV-cache hit measurements**; Ollama's chat template is applied afterwards.

Ordinary steward chat starts with a different system prompt and tool definitions.
It shares zero complete leading messages with the planner. This does not alone
prove eviction: the first exploratory run retained fast exact-repeat planner
prefill even with a small ordinary chat interposed.

## Method

Approved isolated package: `/tmp/hc-prefix-reuse` inside `cortex-cortex-api-1`
on `jkuang@192.168.68.59`. No deployment, upgrade, model unload, runner-option
change, or private household data read. The host was **already on Ollama 0.32.15**.
The model stayed listed as qwen3.5:9b, digest `6488c96fa5fa…`, 8192 context and
5.6 GB resident VRAM. Presence/context/digest snapshots cannot prove absence of
an intervening restart; production runner logs were not available to this task.

The script captures requests from the real `OllamaService` and planner message
builder/schema, using the synthetic semantic-contract fixture. Requests go
straight to `/api/chat`; this isolates inference and excludes validation retries,
resolution/execution, WebUI, identity authorization, and ingress. Conversation
histories are synthetic interleaved user turns, not a live multi-user browser test.

Both variants use 8192 context, 24h keep-alive, thinking disabled. Planner sampling
is temperature/seed 0 and 384 output tokens. Ordinary chat retains its actual
sampling defaults and has no artificial output cap. The refined run uses the
real steward prompt, clock helper and tool definitions; it excludes private
speaker bindings. Clock-only changes alter only the timestamp's timezone suffix;
query-only changes keep the clock fixed. Neither freezes or coarsens production time.

Exploratory run: native then canonical, three measured four-request cycles per
scenario after one discarded warmup cycle, using a short generic chat prompt.
Refined run: canonical then native, two measured four-request cycles per scenario
after one warmup cycle, using the actual steward prompt/tools and additional
clock-only/query-only/varied-planner alternation controls. Do not pool these runs
as identical cohorts. The GPU is shared, not reserved exclusively by the probe.

Full numeric rows remain in the isolated directory and local `/tmp` copies.
The checked-in summaries retain distributions, request/prefix hashes, model and
source/schema/fixture fingerprints, and paired output comparison counts. Parsed
JSON equality is a regression comparison, **not a semantic accuracy score**.

## Refined GPU results

Planner means in milliseconds. Eight measured planner calls per variant in
planner-only scenarios; four per variant in alternating scenarios.

| Traffic | Native prefill | Canonical prefill | Native wall | Canonical wall |
|---|---:|---:|---:|---:|
| repeat | 70.8 | 69.5 | 817.7 | 809.6 |
| vary_clock_only | 492.6 | 482.9 | 1245.4 | 1243.1 |
| vary_query_only | 490.6 | 484.3 | 1328.1 | 1314.7 |
| vary_query_clock | 386.2 | 382.2 | 1234.2 | 1224.8 |
| two_conversations | 488.3 | 485.9 | 1439.3 | 1434.6 |
| alternating_chat | 70.6 | 71.0 | 980.5 | 979.6 |
| alternating_chat_varied | 488.7 | 486.4 | 1482.1 | 1483.6 |

**No material latency reduction was demonstrated by canonicalization.** Exact
repeats retain ~70 ms prefill. Changing just the timestamp string or just the
query raises it to ~483–493 ms, despite the large unchanged prefix. Interleaved
conversation histories behave similarly. The combined query/clock scenario's
lower mean includes an adjacent exact-repeat at the cycle boundary; it is not a
representative general-conversation improvement.

Actual steward chat/tool requests between **identical** planner prompts still
leave planner prefill near 71 ms. With **different** planner prompts in the same
alternation, prefill rises to ~486–489 ms. Alternation alone is not evidence that
all cached planner state was evicted. Wall time also includes additional overhead
not reflected in prefill; do not infer wall savings directly from cache timings.

All **48/48** paired planner outputs in the refined run and **42/42** in the
exploratory run matched as parsed JSON between native and canonical order.
All planner responses parsed as JSON objects. This is a small synthetic output
regression check, not an accuracy or production reliability certification.
The complete local deterministic suite passes: **462 tests**.

## Next step justified by these measurements

Keep canonical serialization for process-independent requests and reproducible
benchmarks. Do not attribute a warm single-process speedup to it.

Investigate the inference runtime's ability to restore a **partial shared prefix**
for this model and chat template. The measured pattern is consistent with a
partial-prefix/checkpoint limitation rather than just losing an exact cached
request. It does not prove which cache, template, checkpoint or recurrent-state
mechanism is responsible. Older upstream reports describe related Qwen3.5
checkpoint behavior, but involve different versions/models and are leads only:
[llama.cpp issue #20225](https://github.com/ggml-org/llama.cpp/issues/20225).

A useful next isolated runtime experiment would record template-token prefix
length and restored checkpoint/token counts for the clock-only and query-only
controls, then compare a runtime configuration/build that can retain a checkpoint
before the dynamic tail. It must preserve fresh clock values, complete semantic
constraints, and strict execution/ownership boundaries. Do not freeze the clock,
reuse semantic answers, append planner instructions to ordinary chat, or add an
extra warmup inference to every user request to manufacture a cache hit.

The roughly 410 ms repeat-versus-changed-prompt gap is an opportunity to test,
not savings delivered by this change. Even eliminating it would remove only
about 29% of the ~1.43 s conversation inference time measured here, not 50%.
Setup is now ~1–2 ms on the already-upgraded host, so the old 700 ms setup budget
must not be counted again.

## Reproduction

Use a separately staged source/schema/synthetic-fixture package, with the model
already resident at context 8192. The script refuses an unknown/mismatched context.

```sh
PYTHONPATH=/tmp/hc-prefix-reuse/src PYTHONHASHSEED=0 \
python /tmp/hc-prefix-reuse/scripts/ollama_prefix_reuse_probe.py \
  --root /tmp/hc-prefix-reuse \
  --output /tmp/hc-prefix-reuse/probe-real-chat.json --repeat 2 --reverse
```

Local, no-inference prefix check:

```sh
PYTHONPATH=src PYTHONHASHSEED=0 .venv/bin/python \
  scripts/ollama_prefix_reuse_probe.py --root . --dry-run --output /tmp/prefix-0.json
```

Repeat with hash seed 1. `tests/test_prefix_reuse_probe.py` verifies final request
stability across both subprocesses and distinguishes conversation prefixes from
ordinary chat. `tests/test_ollama.py` locks unchanged array semantics and the
static-before-dynamic layout.

Source changes are not deployed. Approval review denied production log inspection
because logs may contain sensitive requests; no logs were retrieved, and no
runtime-level cause or restart count is claimed. The approved isolated numerical
measurements completed successfully.
