# Unified planning with single-intent semantics

## Decision

**ACCEPTED AND INTEGRATED.** Mutation-enabled agents now use one structured
semantic interpreter call whose mutually exclusive branches are `fact`,
`mutation`, `conversation`, and `multi_intent`. The historical mutation-first
pre-classifier is no longer on the production request path. A multi-intent turn
carries no fact request or mutation, executes nothing, and returns a localized
request for one instruction at a time.

The serving `/app` deployment and household database were not changed. Model
evaluation used an isolated package and frozen JSON graph under `/tmp` in
`cortex-cortex-api-1` on `jkuang@192.168.68.59`.

## Runtime contract

`SemanticPlan` remains the common envelope and now carries `multi_intent`:

```text
fact:         requires_fact=true,  request=<fact>, mutation=null,    multi_intent=false
mutation:     requires_fact=false, request=null,   mutation=<write>, multi_intent=false
conversation: requires_fact=false, request=null,   mutation=null,    multi_intent=false
multi-intent: requires_fact=false, request=null,   mutation=null,    multi_intent=true
```

The Pydantic model and structured output schema reject overlapping branches.
Mutation modes are required in model output. Explicit previews must emit
`mode=preview`; the existing named create, location update, attribute update,
and delete union remains unchanged. Fact requests still pass ontology expansion,
entity-ID rejection, semantic-schema validation, identity/location checks,
deterministic resolution/execution, and deterministic rendering.

`AgentService` selects `UnifiedSemanticPlanner` when `write_item` is enabled and
keeps `SemanticFactPlanner` for read-only agents. The old classifier call and its
negative-decision handoff were removed from `SemanticFactPlanner`. A small
`LegacyTwoStagePlanner` remains under `scripts/probes/` only so historical paired
benchmarks can reproduce the former route.

For multi-intent output, `SemanticFactService` returns a fixed `FactAnswer` with
status `multi_intent_unsupported`. It never invokes `HouseholdFactEngine` or the
write dispatcher. English output asks for one instruction at a time; Chinese
output is `这句话包含多个操作，请一次吩咐一件事。`

## Read acceptance

The final paired run used the frozen 171-query suite. One excluded warm-up per
route preceded the measured pass. The final retry contract was included.

| Metric | Historical two-stage | Unified | Change |
|---|---:|---:|---:|
| LLM calls, median / p95 | 2 / 2 | **1 / 1** | −50% |
| Input tokens, median | 9,453 | 11,269 | +19.2% |
| Output tokens, median | 59 | 56 | −3 |
| Planning latency, median | 2,623.162 ms | **1,706.960 ms** | **−34.9%** |
| Planning latency, p95 | 2,852.817 ms | **1,931.136 ms** | **−32.3%** |
| Plan correctness | 154/171 | **165/171** | +11 net |
| Answer correctness | 150/171 | **162/171** | +12 net |
| Validation failures | 2 | **0** | −2 |
| Length stops | 0 | 0 | 0 |

The unified route exceeds the fixed acceptance floors of 155/171 plans and
151/171 answers. Paired comparison found 12 plan improvements, one plan
regression, 12 answer improvements, and **zero answer regressions**. The sole
plan regression, `告诉我家庭地址`, produced the same correct deterministic answer.

An initial aggregate exposed an unstable son-name validation failure. The final
generic retry rule now states the existing `resolve_reference` contract:
`property=null` and `property_source=entity`, including kinship paths. With the
integrated code, `我的儿子叫什么`, `What is my name?`, and
`我们成为夫妻多长时间了` each passed three focused runs. The final full run had
zero validation failures and no new repeat-stable identity or kinship regression.

## Mutation and compound-intent acceptance

Three passes covered 14 standalone supported writes, seven compound turns, and
eight unscored ambiguous utterances: 87 observations per route.

| Metric | Historical two-stage | Unified |
|---|---:|---:|
| Standalone write classification | 36/42 | **42/42** |
| Standalone exact payload | 24/42 | **34/42** |
| Compound → multi-intent | 0/21 | **21/21** |
| Compound partial executable plans | 18 | **0** |
| Explicit previews preserved | 3/15 | **15/15** |
| Explicit preview emitted as commit | 6 | **0** |
| Validation failures | 6 | **0** |

The compounds include fact-plus-mutation, mutation-plus-mutation, and
fact-plus-fact turns in English and Chinese. The probe never dispatched a write.
Ambiguous declarations remain observations without assigned gold labels.

## Integrated smoke and validation

The isolated integrated package produced the same evaluated prompt contract:
47,804 message-content bytes, 40,346 structured-schema bytes, and SHA-256
`760f9e46c3154b7e0b7fd90d8ea8b25d5d56386b0d3a5ea8ca71fb40c6512239`.
Planner-only smoke inputs returned all four branches in one call; the mutation
smoke retained preview mode. Both Ollama and OpenRouter adapters expose the same
unified planner surface.

Focused planner, agent, renderer, transport, and provider tests passed before the
full deterministic run: **781 passed**. The semantic transport codec version moved from 4 to 5
because the canonical envelope gained a field. Raw model observations remain at
`/tmp/hc-unified-single-v1/intents-v12.json` and
`/tmp/hc-unified-integrated-v1/reads-final.json` on the evaluation host.

Model: `qwen3.5:9b`; Ollama: 0.32.15; context/output allocation: 16,384/384;
model digest: `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`;
graph SHA-256: `a6ab9fa06584cc0a369d4e7501252c4f2402f36ec700cc68e4126d1e360150ef`;
schema SHA-256: `e192342e0df849ea589a023775ef116c4b029d2fdb8454b37b7c807c4661c156`.

These measurements cover planner latency and deterministic frozen-graph answer
comparison. They do not include reverse-proxy/network latency, successful
database mutation latency, or arbitrary-length user input.
