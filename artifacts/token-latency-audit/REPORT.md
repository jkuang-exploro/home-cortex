# Token and latency audit — 2026-09-07

## Scope and baseline

The optimization is deliberately small: cache immutable few-shot example text,
returning fresh message dictionaries. Semantic IR, resolution, ontology, prompts,
model settings, factual authority, and routing are unchanged. No application
release was deployed.

Measurements use `qwen3.5:9b` on `jkuang@192.168.68.59`, RTX 2060 SUPER,
Ollama 0.32.13, 100% GPU, context 8192, `think=false`, temperature/seed 0,
384 output-token ceiling, and 24-hour keep-alive. Both packages run from isolated
`/tmp/hc-token-audit-*` directories in the API container. All graph data is the
invented `benchmarks/fixtures/semantic-contract` household.

The main probe has 35 measured requests after an excluded full warm-up pass:
33 simple/derived/reference/ambiguous/filter/date cases and a follow-up measured
with and without persistent focus. It combines the synthetic, age-filter, and
date-interval datasets, with additional identity/count/youngest cases confined to
the harness. Each dataset retains its own frozen clock. Follow-ups use the
synthetic daughter's birthday and age. One measured pass is a diagnostic sample,
not a reliability or production-accuracy certification.

**Controlled results:** both runs produce identical normalized IR, **29/35 semantic
matches and 31/35 correct structured results**, with the same failures. Prompt
fingerprint `2a2c63502d037fa13cc52f277c63c9a312f0126ff8f32ae84a9bcb2ed9c18ea7`
and input/output counts match. Semantic-pipeline p50 is **2,191.6 → 2,170.2 ms**,
p95 **3,749.6 → 3,696.8 ms**. Mean model wall time is effectively unchanged
(**2,455.1 → 2,456.2 ms/request**); percentile differences are not evidence of
an inference speedup. Existing failures are three birth-year/filter cases and
a household-count request; two further cases return the expected result with
nonmatching semantic plans. No equivalence rules were loosened.

The supplemental API probe exercises ten requests after ten warm-ups through the
actual `/v1/chat` ASGI route, authenticated identity mapping, agent coordinator,
real SurrealDB queries, and JSON response serialization. It creates and removes
its own UUID-named synthetic namespace. It excludes browser, reverse-proxy and
TCP ingress latency. p50 is **2,143.6 → 2,146.4 ms**, p95 **2,962.2 → 3,028.1 ms**.
All ten final answer hashes match, with **10 model calls and 39 DB queries** per
pass. There is no demonstrated total-API speedup.

The local deterministic replay runs 350 measured requests per condition with gold
IR and no inference: **350/350 semantic and structured-result checks pass both
before and after**. These are executor checks, not LLM accuracy. The original
pytest baseline was **450 passed**; final validation is **457 passed**.

## Largest token components

Model-native raw-prompt probes generated one discarded token per component,
outside the latency runs. Counts exclude chat framing and are approximate
attributions when compared with the full chat prompt; they are not an exact
additive token partition. The measured UTF-8 byte shares are exact shares of
message content, excluding message JSON/framing and the separate schema.
Approximate token shares divide isolated counts by the measured mean of 4,598
chat-input tokens; separate tokenization/framing leaves an unallocated remainder.

| Rank/component | Raw Qwen tokens | Approx. share of mean chat prompt | Content bytes/share | Static/dynamic; necessity and duplication |
|---|---:|---:|---:|---|
| 1. Capability/ontology payload | 1,562 | 34.0% | 7,243 / 36.5% | Static per schema instance. Full grammar is sent even for identity. Operation, reference and ownership descriptions overlap instructions and output-schema enums. Required vocabulary is query-dependent; safe scoping has not been established. |
| 2. System instructions | 1,479 | 32.2% | 6,786 / 34.2% | Static. Includes identity, composition, ownership, filtering and dates. Only a subset applies to an individual query; repeated reminders are intentional safeguards, not proven removable. |
| 3. 22 example pairs | 1,230 | 26.8% | 5,525 / 27.8% | Static. Demonstrate grammar already described in prose. Three assistant-identity examples reinforce related distinctions. Necessity of each example requires an ablation experiment; no content was removed. |
| Reminder, current query, notes | Not independently tokenized | Unallocated | 279 / 1.4% in representative first case | Dynamic clock/query and optional identity hint; identity instructions repeat main prompt and capability descriptions. |
| Household records/identity bindings | 0 | 0% | 0 | Not injected into the semantic interpreter. Named references remain literal user language. |
| Prior assistant answers | 0 | 0% | 0 | Omitted from semantic interpretation. |

The output schema is **6,549 bytes** and independently encodes to **1,564 raw
Qwen tokens**. It is a separate `format` constraint, not a message component;
do not add its isolated count to the provider's prompt count. The representative
wire messages occupy 23,096 UTF-8 bytes, including roles and JSON escaping.
Real prior discourse adds up to eight user turns, separated by a fixed boundary
message; persisted identity bindings stay out of the prompt. User-turn count is
bounded, but the length of each supplied user turn is not bounded here.

The controlled probe sends **174,726 input tokens** and generates **2,127
output tokens** in **38 calls**, unchanged after optimization. Mean call size is
**4,598 input / 56 output tokens**; maximum completion is **95 tokens**. Most
requests make one call; two validation retries and one stateless antecedent
account for the three extra calls. Output is compact structured IR, with no
measured output-volume bottleneck beyond ordinary token-generation cost.
The output cap is not itself a token saving: reducing it could truncate valid
IR. Ollama explicitly disables thinking. Separate reasoning-token usage is
unavailable in these planner responses; this is not reported as a measured zero.
OpenRouter usage, including provider-supplied reasoning counts, is captured by
the new trace but was not benchmarked against a live OpenRouter model.

## Largest latency components

Ranked by contribution per request in the controlled baseline:

| Rank/stage | Mean ms/request | Share of mean semantic-pipeline elapsed time |
|---|---:|---:|
| 1. Token generation | 1,138.7 | 46.4% |
| 2. Reported model load stage | 752.6 | 30.6% |
| 3. Prompt evaluation/prefill | 485.3 | 19.8% |
| Remaining transport/SDK/runtime wall interval | 78.5 | 3.2% |
| All Python + JSON-fixture graph processing | 1.32 | 0.054% |

The largest non-model contributors in the **real API/SurrealDB baseline** are
DB queries (**3.28 ms/request**), unattributed HTTP/auth/parsing/serialization
work (**1.47 ms**), and planner message construction (**0.389 ms**). These are
separate API-cohort measurements, not additional terms in the table above.

Ollama's `load_duration` is reported on warm, resident-model requests too. It
must not be equated with a verified weight reload. Its internal cause needs
runtime investigation. The remaining transport/SDK interval is computed as wall
time minus reported load, prefill and generation; it is not another inference
measurement. Non-streaming planner TTFT is **unavailable**, not zero and not
`load + prefill`. Streaming instrumentation distinguishes first observed token,
post-first-token wall time, and provider generation duration.

Deterministic stage timings include cached schema lookups, prompt construction,
planner/validation overhead, resolution, graph records, computation and rendering.
The real API's non-model work is **6.13 → 6.62 ms/request**; DB latency varies
**3.28 → 3.79 ms**, overwhelming the **0.389 → 0.189 ms** message-builder saving
in this sample. Baseline message normalization costs **0.006 ms**, identity
normalization **0.003 ms**, trusted context construction **0.037 ms**, and rendering
**0.015 ms**. The 1.47 ms HTTP residual combines work rather than measuring JSON
parsing separately. Validation/deserialization share the planner's approximately
**0.30 ms** exclusive interval; existing diagnostics provide a narrower validation timer.

A measured baseline marriage-date request illustrates the timeline:

```text
ASGI request starts                    0.000 ms
identity DB read                   1.200–1.806 ms
normalize messages/identity        1.843–1.862 ms
planner message construction       1.948–2.392 ms
model request                      2.397 ms
  input/output tokens              4,595 / 47
  load / prefill / generation       708.3 / 495.6 / 873.9 ms
  TTFT                             unavailable (non-streaming)
model response complete            2,151.924 ms
planner validation complete        2,152.211 ms
fact execution                     2,152.281–2,153.879 ms
  graph DB query                   1.355 ms within execution
render complete                    2,153.890 ms
ASGI response complete             2,154.554 ms
```

No filesystem or schema rebuild occurs on the warm request path. Ontology,
capability/output schemas, and HTTP model clients are already reused. Entity
loads are cached within one `_FactExecution`; alias resolution still reads and
normalizes relevant table records, and traversals/loads are sequential. The small
fixture does not establish scaling behavior on a large household graph.

## Model call graph and discarded work

```text
HTTP / identity / speaker context
  -> semantic interpreter (one call; at most one validation retry)
     -> valid household IR -> resolve -> execute -> deterministic render
     -> not a fact -> ordinary chat/tool loop (up to four model steps)

Stateless discourse:
  current-turn interpreter -> unresolved execution
  -> interpret referenced prior user turn(s), memoized within this request
  -> execute current IR again using reconstructed focus

Persistent discourse:
  current-turn interpreter -> resolve stored focus -> execute -> render
```

There is no LLM entity resolver, LLM validator, or LLM factual renderer. Ordinary
chat's interpretation call affects routing and is not dead work. Failed planner
attempts consume tokens but protect strict validation. Their generated plans are
discarded; the retry resends the full grammar/examples plus a correction note.
Stateless discourse discards the initial unresolved execution/render and performs
extra antecedent interpretation. Existing persistent conversations avoid those
antecedent model calls without changing resolution authority.

The controlled baseline stateless follow-up takes **3,588 ms / two calls**,
versus **1,764 ms / one call** with a persistent conversation; both return the
correct age. After caching these measure **3,520 / 1,806 ms**. This is a measured
benefit of an already-supported path, not a new architecture change. Both
current-turn calls consume 4,636 input / 50 output tokens; persistence avoids
the additional antecedent's **4,591 input / 46 output tokens**.

| Observed extra-call path | First call: input/output, wall ms | Second call: input/output, wall ms | Contribution |
|---|---|---|---|
| Birth-year count retry | 4,594 / 89; 3,067.8 | 4,652 / 95; 3,140.3 | First invalid plan discarded; retry still fails expected result. |
| Household count retry | 4,593 / 45; 2,091.6 | 4,651 / 45; 2,033.2 | Both invalid; final response remains unsupported. |
| Stateless follow-up | 4,636 / 50; 1,847.3 | 4,591 / 46; 1,738.5 | Prior-turn grounding makes current IR executable. |

Neither retry produces a correct result in this small sample. That is evidence
for investigating those semantic failures, not proof that removing validation
retries would preserve behavior.
Independent requests may run concurrently; turns sharing trusted conversation
state must remain serialized. Speculating about an antecedent before seeing the
current IR would reintroduce unnecessary calls. Structured callers already have
`HouseholdFactEngine.execute` / `answer_request` as a general zero-LLM path.
Sentence-based Tier-0 routing has been removed; no new semantic fast path was added.

## Implemented changes and measurement corrections

- **HIGH VALUE / LOW RISK — observability:** opt-in context-local, bounded traces
  capture every model call rather than a shared latest-call snapshot. Tokens and
  timing stay null when unknown. Numeric metadata only; no prompt, output, SQL,
  graph value or credential is logged. HTTP tracing spans streaming completion.
  Tests cover concurrent isolation, bounds, retries/errors, missing usage, stream
  closure/cancellation, OpenRouter streaming usage, and the HTTP lifetime.
- **LOW RISK / SMALL BENEFIT — immutable examples:** one process-local cache,
  invalidated by process/code replacement. No household/session cache. Exact
  message equality was verified for all 53 source-dataset cases. In ten alternating
  batches of 1,000 constructions in the same process, median construction cost
  fell **0.0919 → 0.0405 ms (56%)**. Token content is unchanged. Enabled tracing
  itself added about **0.0094 ms** to a local identity replay (1,000 samples).
- **Evaluation correction:** initial combined-dataset replay incorrectly shared
  one clock; corrected before the recorded successful model runs. Initial model
  runs then exposed process-dependent dictionary ordering from a relation set.
  They are preserved as `probe-gpu-exploratory-*-summary.json`, not used as the
  controlled improvement claim. Controlled runs set `PYTHONHASHSEED=0` in both
  processes and compare prompt fingerprints. Older diagnostic `routing_ms`,
  `llm_ms` and `request_ms` overlap; `last_planner_runtime` retains only the last
  attempt. New trace intervals and provider-call events are the accounting source.

Instrumentation is the main LOC addition, is disabled by default, and introduces
no new dependency. The cache is the only runtime optimization. No cross-request
household cache was introduced because there is no measured need or explicit
fact-revision invalidation contract. Existing conversation state remains scoped
by conversation, speaker, household and agent.

## Three further changes with highest expected return

1. **HIGH VALUE / LOW RISK:** ensure clients consistently reuse the existing
   authorized `conversation_id`. The measured two-turn example avoids an entire
   antecedent inference call. This uses current conversation semantics and fresh
   authoritative entity reloads.
2. **MEDIUM:** investigate the substantial warm-request Ollama load stage with
   runtime tracing, keeping model/context resident and checking scheduling/setup
   overhead. Existing persistent clients and keep-alive already rule out the
   simple “create a client once” fix. This has a much larger potential return than
   Python micro-optimizations, but the measured duration is an upper bound on
   possible savings, not a proven removable delay.
3. **MEDIUM:** stabilize serialized static-prefix ordering and run focused
   instruction/example ablations with matched prompt fingerprints, clocks,
   repeated model runs and held-out semantic cases. Preserve the ownership,
   predicate, projection, and discourse distinctions. The prefix dominates input
   tokens, and set-dependent ordering is already a demonstrated confound.

**ARCHITECTURAL — requires review:** query-scoped ontology selection, new routing,
resolver/IR ownership changes, and persistent household indexes without a
fact-revision invalidation contract. None was implemented. Removing examples or
scoping ontology was not silently accepted on token size alone.

## Reproduction and retained evidence

See `src/README.md` for tracing and script commands. Set `PYTHONHASHSEED=0` for
comparison processes. Source/schema/data/model fingerprints are retained in the
summary JSONs and `manifest-summary.json`. Full synthetic traces remain in the
isolated GPU directories and local `/tmp/hc-*` result files. A separate read-only
root-metadata check confirms **no temporary audit namespaces remain**. The report separates
semantic-pipeline, full-ASGI/SurrealDB, and replay measurements; no browser/proxy
latency, cold-load latency, or live private-household accuracy is claimed.

## Before/after comparison

The main cohort has 35 requests; API values come from the separate ten-case
ASGI/SurrealDB cohort. Times are milliseconds. Model variability is not
attributed to a sub-millisecond Python cache.

| Area | Before | After | Improvement | Correctness impact |
|---|---:|---:|---|---|
| Prompt tokens/call, main cohort | 4,598 | 4,598 | None; identical text | None |
| Output tokens/call, main cohort | 56.0 | 56.0 | None | None |
| LLM calls/request, main cohort | 1.086 | 1.086 | None | None |
| Planner TTFT | Unavailable | Unavailable | Non-streaming | N/A |
| Model wall time/request, main cohort | 2,455.1 | 2,456.2 | No demonstrated gain | 31/35 results both |
| Message building/request, main cohort | 0.465 | 0.222 | 0.243 ms / 52% | Identical IR |
| Non-model work/request, main cohort | 1.320 | 1.089 | 0.231 ms | None |
| Full API non-model work/request | 6.13 | 6.62 | No demonstrated gain | Identical answer hashes |
| Full API end-to-end p50 | 2,143.6 | 2,146.4 | No demonstrated gain | Identical answer hashes |
| Full API end-to-end p95 | 2,962.2 | 3,028.1 | No demonstrated gain | Identical answer hashes |

