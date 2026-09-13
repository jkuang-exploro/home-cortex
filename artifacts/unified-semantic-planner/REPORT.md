# Unified read/write semantic planner experiment

## Decision

**REJECT — existing two-stage route retained.** The one-call shadow candidate
classified all standalone supported writes, but repeatably discarded the mutation
half of every mixed question/write case. It was not integrated or deployed.
Production additions/deletions: **0 / 0 LOC**.

## Existing architecture and unified contract

With `write_item` enabled, `SemanticFactPlanner.plan` first calls the mutation
planner. `MutationDecision` contains `requires_mutation` plus an optional canonical
named item mutation. A positive decision short-circuits fact planning; a negative
decision calls the semantic fact planner. Incomplete/unsupported writes may carry
no mutation and fall through to ordinary conversation. The current path therefore
gives supported mutations deterministic precedence over a fact branch.

The mutation union is unchanged:

| Operation | Canonical required meaning |
|---|---|
| `create` | item name, destination, bilingual display names, readable key; optional declared attributes; preview/commit mode |
| `update_attributes` | item name and one or more declared writable attributes; preview/commit mode |
| `update_location` | item name and destination; preview/commit mode |
| `delete` | item name; preview/commit mode |

The semantic planner produces `SemanticFactRequest` with an operation, subject,
property and explicit property owner, filters, projection/exclusions, optional
comparison operand, amount and mode/unit fields. Ontology concept expansion,
schema validation, entity-ID rejection, identity/location checks, resolver,
executor and renderer remain unchanged. A non-fact/non-mutation plan falls back
to conversation.

No new result type was needed. Existing `SemanticPlan` is the common envelope:

```text
fact:     requires_fact=true,  request=SemanticFactRequest, mutation=null
mutation: requires_fact=false, request=null,                mutation=NamedWriteRequest
none:     requires_fact=false, request=null,                mutation=null
```

Its Pydantic validator rejects a simultaneous fact and mutation. The candidate
used `SemanticSchemaRegistry.planner_output_schema()` plus the existing
`attribute_output_schema()` adapter, so mutation attributes stayed restricted to
canonical writable ontology properties. It reused the semantic prompt,
capabilities, examples, mutation instructions and positive mutation examples.
It introduced no V2 IR, semantic vocabulary, execution engine or write path.

## Shadow method

Both routes received identical utterances from one frozen package and graph copy.
The existing route and candidate ran sequentially on `qwen3.5:9b` at
`jkuang@192.168.68.59`. The candidate called the model directly through the
existing Ollama client, validated the existing IR, and returned its decision for
comparison. It never dispatched `write_item`. Read decisions alone were executed
against the frozen JSON graph for differential answer comparison.

Mixed turns retained the current deterministic precedence contract: if a turn
contains an explicit supported item mutation, emit that one mutation and do not
execute the question. This preserves current behavior without introducing
multi-action orchestration.

Gold-defined cases cover standalone Chinese/English create, move, attribute
update, delete, preview/commit, and three mixed question/write turns. Ambiguous
declarations and corrections are baseline-observed, not treated as gold.

## Mutation, mixed, and ambiguous results

Three passes produced 30 standalone write observations, 9 mixed-intent
observations, and 24 ambiguous observations per route.

| Metric | Existing | Unified | Change |
|---|---:|---:|---:|
| Standalone write classification | 30/30 | 30/30 | 0 |
| Standalone exact payload | 16/30 | 19/30 | +3 |
| Mixed mutation classification | 9/9 | **0/9** | **−9** |
| Mixed exact payload | 6/9 | **0/9** | −6 |
| Standalone write LLM calls | 1.00 | 1.00 | 0 |
| Standalone median input tokens | 1,209.5 | 9,580.5 | +8,371 |
| Standalone median latency | 1,076.151 ms | 1,336.543 ms | +24.2% |
| Standalone p95 latency | 2,195.793 ms | 2,063.201 ms | −6.0% |
| Validation failures / length stops | 0 / 0 | 0 / 0 | 0 |

The unified candidate emitted a fact plan on every pass for all three mixed
utterances:

- `Who am I? Delete the lamp record.`
- `我是谁，然后删除台灯的记录`
- `Where is the lamp? Move it to the kitchen.`

These are repeat-stable silent mutation losses, the ticket's highest-severity
safety error. The current route preserved a mutation in all nine observations.
No prompt/transport failure explains the difference.

Exact payload scoring intentionally includes item name, destination, explicitly
requested attributes, bilingual create names/key, and preview/commit mode. Both
routes incorrectly emitted `commit` for all preview cases. They also varied in
key transliteration, translated item names, and unsupported/invented create
attributes. The unified candidate's three-point standalone improvement does not
offset the mixed-intent safety regression.

Ambiguous sentences were recorded without assigning gold. They include plain
location declarations, questions and corrections. Their inconsistent baseline
behavior remains evidence that the planner is making semantic choices; it was not
converted into a new rule or accuracy score.

## Read evaluation

The read set contains the 12 fixed acceptance queries, existing 20-case probe,
119 generalization utterances, and 20 bilingual utterances. One measured pass per
route followed one excluded warm-up for each route. Existing failures were kept.

| Metric | Existing | Unified | Change |
|---|---:|---:|---:|
| LLM calls per read | 2 | 1 | −1 (−50%) |
| Median total input tokens | 9,453 | 9,579 | +126 (+1.3%) |
| P95 total input tokens | 9,478.5 | 9,624 | +145.5 |
| Median output tokens | 59 | 47 | −12 |
| Median planning latency | 2,598.098 ms | 1,499.721 ms | −1,098.377 ms (−42.3%) |
| P95 planning latency | 2,881.128 ms | 2,336.175 ms | −544.953 ms (−18.9%) |
| Read-plan correctness | 155/171 (90.64%) | **147/171 (85.96%)** | −8 net |
| Differential answer correctness | 151/171 (88.30%) | **149/171 (87.13%)** | −2 |
| Validation failures | 2 | **7** | +5 |
| Length stops | 0 | 0 | 0 |

Paired comparison found nine plan regressions and one improvement. Two plan
regressions changed deterministic answers, with no answer improvement:
`What is my name?` and `我们成为夫妻多长时间了`. Other new plan mismatches included
child count/identity and nested kinship name queries; several happened to produce
the same answer or were already answer failures in the existing route. The compact
read summary lists every semantic delta without committing the full 342-row raw
report.

The existing route made three calls beyond its normal two-call read path and
finished with two invalid plans. The unified route retried eight requests and
still finished with seven invalid plans. No call stopped for length. The initial
raw run recorded final validation failure type but not enough detail to subdivide
all seven into malformed versus semantic-contract failures; the focused
confirmation below records the exact error for the important answer regression.

Three additional paired passes over the two answer regressions produced existing
6/6 correct and unified **0/6**. `What is my name?` exhausted two unified attempts
with `INVALID_PLAN` on every pass. `我们成为夫妻多长时间了` returned a structurally
valid `date_difference` plan on every pass, but selected `years` instead of the
established unspecified-duration default of `days`. These are repeat-stable
semantic failures, not timing noise.

The candidate met the performance goal for reads, but failed both semantic and
stability thresholds. The speedup cannot justify the paired regressions or the
mixed-intent mutation loss.

## Overall performance comparison

| Metric | Existing | Unified | Change |
|---|---:|---:|---:|
| Read LLM calls | 2 | 1 | −1 |
| Read median input tokens | 9,453 | 9,579 | +1.3% |
| Read median latency | 2,598.098 ms | 1,499.721 ms | −42.3% |
| Read p95 latency | 2,881.128 ms | 2,336.175 ms | −18.9% |
| Standalone mutation median latency | 1,076.151 ms | 1,336.543 ms | +24.2% |
| Read-plan correctness | 155/171 | 147/171 | −8 |
| Standalone mutation classification | 30/30 | 30/30 | 0 |
| Mixed mutation preservation | 9/9 | 0/9 | −9 |

## Prompt and context

The production-catalog unified prompt contains 41,166 message-content bytes; its
structured output schema is 38,794 bytes. Native unified input was 9,581 tokens
median and 9,611 p95 across the intent suite. No call stopped for length within
the unchanged 16,384-token context and 384-token output allocation.

The prior unchanged nine-read routing probe measured 1,206 median input tokens
for mutation classification and 8,243 for fact interpretation, totaling 9,449.
The broader paired read run measured 9,453 total median for the existing route
and 9,579 for unified. Across normal one-call unified read/intent observations,
the largest input was 9,630 tokens, leaving 6,370 tokens after reserving the
384-token output budget.

The schema is a decoding constraint and is reported separately rather than added
to prompt tokens. The candidate prompt is larger than either current planner and
roughly comparable to the combined input of the two-stage read path. Context
safety passed for measured cases, while arbitrary user length remains unbounded.

## Code impact and tests

Production runtime code under `src/home_cortex`: **no changes**. Added only a
shadow planner, paired experiment harness, compact result summarizer, focused
contract tests, benchmark expectations, documentation and evidence. The candidate
exists under `scripts/` and is not reachable from API or agent routing.

Focused planner/agent/API/write/semantic suites: **276 passed**. Full deterministic
suite: **775 passed**. Tests cover all three envelope branches, mutual exclusion,
closed writable attributes, transformed canonical mutation examples, and the
single-call shadow boundary.

## Remaining issues

The unified prompt gives fact demonstrations enough weight to override explicit
mutations in compound turns despite a direct mutation-precedence instruction.
Fixing that would require another prompt-semantics experiment; this ticket's stop
condition requires retaining the current route after repeat-stable mutation loss.

Separately, both current and unified planners mishandled explicit preview wording
in this dataset. That is an existing mutation-planner correctness issue and was
not changed here.

Reproducibility evidence is in `frozen-package-summary.json`,
`frozen-confirmation-summary.json`, `intents-summary.json`, `reads-summary.json`,
and `read-confirmation-summary.json`. Model digest:
`6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`;
graph SHA-256:
`a6ab9fa06584cc0a369d4e7501252c4f2402f36ec700cc68e4126d1e360150ef`.
Raw reports remain at `/tmp/hc-unified-49619433` on the evaluation host.
