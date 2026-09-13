# Tier-1 planner prompt compression — controlled experiment

No serving prompt reduction was accepted. Removing just two apparently redundant
example pairs saved 134 input tokens (1.63%) but introduced four held-out semantic
and answer regressions. Reduction stopped at the specified accuracy stop condition.
The 42 example pairs, capabilities, instructions, model, mutation routing, and
16,384-token context remain unchanged. The stretch size/latency goal was not met.

## Baseline and final prompt composition

Production GPU, `qwen3.5:9b`, digest `6488c96fa5fa…`, Ollama 0.32.15. These are
measured native **raw component** counts, excluding chat framing; they are not
additive chat-template accounting. Percentages use their raw-count sum.

| Component | UTF-8 bytes | Raw tokens | Share of raw counts |
|---|---:|---:|---:|
| Instructions, including semantic contract prose | 11,313 | 2,413 | 30.93% |
| Capabilities, including derived property contracts | 12,620 | 2,819 | 36.14% |
| Few-shot examples: 42 pairs / 84 messages | 10,935 | 2,505 | 32.11% |
| Clock and final reminder | 191 | 62 | 0.79% |
| User input (`家里有几个人`) | 18 | 2 | 0.03% |
| History / retry notes for this request | 0 | 0 | 0% |
| **Message content** | **35,077** | — | — |

Serialized message JSON is 41,316 bytes. The separate structured `format` schema
is 35,296 bytes; it constrains decoding and is not added to prompt-message bytes
or treated as input tokens. Identity requests also have a measured 131-byte
supplemental hint. Full chat input tokens come from actual request responses below.

## Fixed benchmark: before versus rejected candidate

Twelve specified queries, three complete measured passes each, one excluded
warm-up per run, seed 0, temperature 0, thinking disabled, 384 output tokens.
The missing-fact case is the assistant's birth date: gold execution returns
`property_unavailable`. Gold plans execute against the same frozen graph to
provide a differential answer oracle; this does not independently validate the
executor or household facts.

| Metric | Baseline | Candidate A | Change |
|---|---:|---:|---:|
| Median planner input tokens | 8,243.5 | 8,109.5 | −134 (−1.63%) |
| Median output tokens | 47 | 47 | 0 |
| Median attempts | 1 | 1 | 0 |
| Requests retried | 0/36 | 0/36 | 0 |
| Median planner latency | 1,148.020 ms | 1,459.676 ms | +27.15% |
| P95 planner latency | 1,226.203 ms | 1,552.888 ms | +26.64% |
| Median total semantic request latency | 1,148.613 ms | 1,460.265 ms | +27.13% |
| Valid structured plans | 36/36 | 36/36 | 0 |
| Correct semantic plans / answers | 30/36 (83.33%) | 30/36 (83.33%) | 0 |

The unchanged baseline already fails kitchen inventory by replacing the named
kitchen with household rooms, and the assistant birth-date request by using
`self`. Both recur on every fixed pass. They were retained, not repaired or
excluded. Context-window correctness recovery is distinct from semantic accuracy.
Per-query bytes/tokens, attempts, timing, operation, validation and result statuses
are in `fixed-cases-summary.json`; all raw attempts remain in remote JSONL files.

These timings cover planner, deterministic JSON-graph execution and rendering.
They exclude HTTP, mutation routing, conversation persistence and live DB I/O.
No end-to-end production API speedup is claimed. Runs were sequential on the
production GPU, not an exclusively reserved GPU; the initial timing difference
is not sufficient to establish a causal latency regression.

## Rejected reduction and held-out verification

Candidate A removed only original example indices 6 and 34: the earlier Chinese
adult-count example (identical plan to index 38), and the English kettle-location
example (same structure as index 28 with a different literal name). Instructions,
capabilities, decoding schema, context and output budgets were identical. No
candidate was stacked or installed into serving code.

The broader evaluation contains 171 cases: the 12 fixed requests, existing
20-case probe, 119 generalization utterances and 20 bilingual utterances.

| Held-out metric | Baseline | Candidate A |
|---|---:|---:|
| Plan correctness | 155/171 (90.64%) | 151/171 (88.30%) |
| Differential answer correctness | 151/171 (88.30%) | 147/171 (85.96%) |
| New plan/answer failures | — | 4 / 4 |
| Newly corrected cases | — | 0 |
| Retried requests | 3 | 3 |
| Median planner latency | 1,481.629 ms | 1,478.922 ms |
| P95 planner latency | 1,691.681 ms | 1,695.774 ms |

No meaningful latency gain accompanied the token saving. New failures were:
`What is my name?`, `我们成为夫妻多长时间了`, `哪个男孩是我儿子`, and
`我的男性子女何时出生`. The paired comparison checks each case, not only aggregate
accuracy. Three additional passes over those four queries gave baseline 12/12 correct and
candidate 3/12: self-identity, marriage duration and son identity failed on all
three passes. The birth-date failure did not recur in this focused confirmation,
so it is not presented as repeat-stable. Both configurations passed measured
context and normal-budget checks; this was a semantic failure, not truncation.

The candidate was rejected; no larger example or capability reduction
was attempted after this accuracy stop condition.

## Final benchmark: retained baseline remeasured

The final configuration is unchanged and was independently remeasured for three
passes (36 requests) with per-call budget enforcement.

| Metric | Before | Retained final | Change |
|---|---:|---:|---:|
| Planner input tokens | 8,243.500 | 8,243.500 | +0.000 |
| Planner output tokens | 47.000 | 47.000 | +0.000 |
| Planner attempts | 1.000 | 1.000 | +0.000 |
| Median planner latency (ms) | 1,148.020 | 1,468.317 | +320.297 |
| P95 planner latency (ms) | 1,226.203 | 1,555.829 | +329.626 |
| Median total semantic latency (ms) | 1,148.613 | 1,469.036 | +320.423 |
| Correct semantic plans / answers | 30/36 | 30/36 | 0 |
| Retries / length stops | 0 / 0 | 0 / 0 | 0 |

All final calls had native counts available and passed the 8,500-token normal
budget and 16K-minus-384 measured context check. The higher elapsed timing for
identical prompts reinforces the timing variability observed in these sequential
GPU runs; no prompt compression or latency improvement was accepted.

## Example audit and token-sink ranking

No exact duplicate utterance/answer pair or deprecated operation was identified.
Identical answer plans occur at 6/38 and 37/40. The latter demonstrates two
cross-language discourse directions; removing it is not justified by JSON
identity. Indices 28/34 differ in literal names and language. The full inventory
records exact-plan and name-independent structural matches.

| Semantic purpose | Original indices | Retained |
|---|---|---:|
| Identity and ordinary chat | 0, 1, 41 | 3 |
| Membership, predicates and gender | 2, 3, 6–9, 38 | 7 |
| Ordering, date/age filters, pairwise comparison | 4, 5, 10–12, 20, 27 | 7 |
| Specific and nested kinship | 14–17 | 4 |
| Properties and date computations | 18, 19, 21, 24–26, 35 | 7 |
| Rooms, named inventory, subspaces and location | 13, 28–34 | 8 |
| Same-entity / residence comparison | 22, 23 | 2 |
| Bilingual discourse sequences | 36, 37, 39, 40 | 4 |

A smaller stable representative set was **not established**. Similar answer
shapes do not prove demonstrations are interchangeable for this model.

Measured sink ranking is capabilities (2,819 tokens), examples (2,505), then
instructions (2,413). Examples offered the easiest isolated test but failed it.
Capabilities already derive vocabulary and contracts from canonical schema,
ontology and operator definitions. Their largest measured fields are reference
concepts (3,537 bytes), property contracts (3,124), operation requirements (1,376),
and property aliases (1,248), excluding enclosing keys/separators. These carry
substantial semantic meaning. Ownership and reference/composition prose overlap
with other contract material; no field was declared unused merely because it
looks repetitive. No second semantic authority or compact mini-language was added.

## History measurement

Eight representative short prior turns, including omission boundaries, add
**1,418 bytes / 270 native raw tokens**: 3.89% of message bytes and 3.35% of the
raw component-token sum with history. Total message content becomes 36,495 bytes.
See `history-components-summary.json` for all component counts and fingerprints.
This short-turn scenario does not justify a history-policy redesign; long turns
remain outside this measured bound. The eight-turn policy is unchanged.

## Guardrails and retained engineering changes

- Keep `OLLAMA_NUM_CTX = 16384` and the 384-token output allocation.
- Encode **8,500 native input tokens** as the measured normal-request regression
  budget for this model/catalog. It is based on the ~8.24K baseline, not the 4K
  stretch goal. `--enforce-budgets` checks every observed call for context headroom
  and length stops, and single-attempt requests for the normal budget.
- A separate 32,000-byte synthetic-fixture test catches local prompt creep without
  pretending UTF-8 bytes are model tokens. The measured fixture is 31,514 bytes.
- The existing representative eight-turn context test remains. Its comment now
  accurately describes a scenario, not a universal maximum: user-turn length and
  catalog growth are unbounded, so arbitrary worst-case fit is not established.
- Added exact component accounting, semantic-example inventory, isolated candidate
  overrides, fixed/differential benchmarks, per-call metrics, paired regression
  comparison and reproducibility instructions. Fixed the existing latency harness's
  stale system-prompt method call and the raw component probe's old 8K setting.

Architecture and production accuracy changes: **none**. Evaluation corrections:
the initial new harness needed ontology expansion of gold concept paths before IR
validation; this was corrected before any baseline model measurements. Existing
baseline failures were retained. Deterministic validation: **767 tests passed**.

## Reproduction and evidence

See `scripts/README.md` for runnable commands. The unchanged source baseline is
frozen in `candidate-e1b8720e64529cb4.tar.gz`; source revision and SHA-256 are in
`frozen-baseline-summary.json`. Evaluation packages and their file manifests are
recorded in the other `frozen-*-summary.json` files.

On `jkuang@home-cortex-0`, runs used independent `/tmp/hc-prompt-*` directories
inside `cortex-cortex-api-1`, with explicit `PYTHONPATH` selecting the frozen
package. No `/app` source or deployment configuration was changed. Graph facts
were copied once into temporary `frozen-data`, never placed in prompts or committed.
Fixed runs: `/tmp/hc-prompt-4baa972b`; held-out runs: `/tmp/hc-prompt-e7843be4`;
final instrumentation/confirmation: `/tmp/hc-prompt-727a43d1`. Adjacent JSONL files
contain per-case details. These temporary packages/data must be retained to repeat
the identical historical graph run; they are not a durable deployment backup.

All compared runs record matching graph, schema, model and evaluation fingerprints.
Graph SHA-256: `a6ab9fa06584cc0a369d4e7501252c4f2402f36ec700cc68e4126d1e360150ef`.
The final harness also records the frozen manifest and fully expanded case-set
fingerprints. Summary artifacts accompany this report; no model results were
inferred from deterministic tests or token estimates.

The measured next optimization target is the derived capability view, particularly
contract/ownership and repeated prose. It requires its own isolated semantic
experiment; this run supplies no evidence that removing those fields is safe.
