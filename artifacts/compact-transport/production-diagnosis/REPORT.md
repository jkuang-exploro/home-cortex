# Production-host diagnosis: compact identity regression

## Observed

`home-cortex-0` is running revision `adeb0674eee8a689965d6c2a1d66bc5743c858c5`
with compact serving **enabled**. The local rollback is uncommitted and is not in
that deployed revision. Actual provider: Ollama 0.32.15, model `qwen3.5:9b`,
`num_ctx=8192`, `num_predict=384`, temperature 0, seed 0. Full fingerprints and
synthetic outputs are in `diagnosis-summary.json`.

A controlled reproduction ran in separate Python processes inside the API
container, using the installed package, resident model and synthetic contract
fixture. No serving files, containers or production graph records were modified.
This exercised the real planner, retries, semantic validator, executor and renderer;
it did not replay an authenticated production HTTP conversation.

Question: **你是谁**.

| Boundary | Compact serving | Isolated expanded control |
|---|---|---|
| Identity root | assistant | assistant |
| Operation | resolve_reference | resolve_reference |
| First extra condition | entity display_name == null | none |
| First semantic validation | INVALID_PLAN | VALID |
| Retry | relationship display_name == 林青 | not needed |
| Retry validation | UNKNOWN_PROPERTY | — |
| Result | semantic_plan_unsupported | found |
| Answer | 老管家无法将这个请求转换为受支持的家庭事实查询。 | 我是老管家。 |

Both compact outputs pass codec parsing. The initial output invents an identity
filter, which canonical semantic validation rejects. The retry invents another
filter on a relationship property. The literal name is from authored prompt
examples, not household data. The factual service correctly renders its fixed
unsupported-query response after two failures. `requires_fact` is **true**: this
is not a model classification of the question as nonfactual.

Direct matched interpreter probes also reproduced the unwanted null filter for
**我是谁**. Expanded output omitted it for both identity questions. These direct
probes do not establish authenticated speaker resolution.

In the full pipeline sample, compact output consumed 12,930 prompt tokens and 93
output tokens across two attempts; expanded output consumed 6,637 and 38 in one
attempt. These are actual Ollama usage counts for this reproduction, not aggregate
accuracy or latency claims. The original offline proxy-token savings did not
establish production behavior.

## Diagnosis and limits

The observed regression is at **model interpretation under compact transport**.
The codec preserves the invented conditions, and semantic validation rejects them.
There is no evidence here that the executor lost household facts or that parsing
turned a valid identity plan into a negative result. Do not repair this by removing
filters, coercing `requires_fact`, or adding a handler for this question.

The exact model/constraint-decoder mechanism causing the unwanted optional fields
has not been isolated. This is one full pipeline comparison plus two matched direct
identity comparisons; it does not establish that every question fails. Production
household-specific capabilities and original request logs were not inspected.

## Next action

The existing local changes restore expanded JSON as the sole serving format in
both providers. They need to reach the revision/image deployed by production;
tracking master currently redeploys the failing compact code at `adeb067`.
No commit, push, restart, or deployment was performed during this diagnosis.

Keep compact output withdrawn until an isolated production-model evaluation passes
full-plan scoring and retry-rate checks. Both captured invalid outputs are now
preserved as deterministic regression tests; semantic rejection remains intact.
The full local deterministic suite passes: **600 tests**.

## Reproduce without household records

From this directory, with authorized host access:

```sh
ssh jkuang@home-cortex-0 'docker exec -i cortex-cortex-api-1 python -' < probe_metadata.py
ssh jkuang@home-cortex-0 'docker exec -i cortex-cortex-api-1 python -' < probe_pipeline.py
```

The pipeline probe issues one compact request (with its normal retry) and one
expanded control. It uses `/app/benchmarks/fixtures/semantic-contract`, fixed
resident-model settings, and emits only synthetic request results and fingerprints.
The expanded control is an in-memory adapter and does not change serving code.

Automatic approval review rejected an attempted production-log read because
error lines could contain private request data. That read did not execute; all
findings above come from source metadata and the controlled synthetic probes.
