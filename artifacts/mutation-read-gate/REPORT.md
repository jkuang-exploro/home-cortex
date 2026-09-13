# Conservative mutation-planner bypass — stopped before implementation

**No safe new fast path was found under this ticket's constraints.** Stop
conditions 1 and 4 apply: the current entry points expose no deterministic read
intent, and deriving one from unrestricted text would duplicate semantic
interpretation. Production LOC added/removed: **0 / 0**. No routing, semantic
prompt, model, context, IR, write behavior or deployment change was made.

## Current path and evidence

`AgentService.__init__` enables mutations when the configured tool list contains
`write_item`. `_prepare_request` constructs `AgentRequestContext`, then calls
`SemanticConversationService` → `SemanticFactService` → `SemanticFactPlanner.plan`.
The latter calls `OllamaService.plan_item_mutation` before semantic interpretation
whenever mutation support is enabled.

The mutation planner returns `MutationDecision(requires_mutation, mutation)` and
runtime metrics. A positive decision short-circuits read planning. It compiles
create, attribute update, location update or delete, including names, destination,
attributes and preview/commit mode. A missing write payload can return no fact
plan for downstream conversation handling. A negative decision continues with the
unchanged semantic planner and contributes one call plus latency/token metrics.

For a compiled mutation, `SemanticFactService` returns `SemanticMutationIntent`;
`AgentService._prepare_request` dispatches `write_item` with
`planned_mutation=True`. Native model-loop writes without that marker are blocked.
This ticket did not execute any compiled mutation. General relationship/person
fact mutations are not among the current supported item-mutation operations.

## Exact gate boundary considered

| Existing evidence | Why it cannot prove these requests read-only |
|---|---|
| `AgentRequestContext` | Identity, household, clock, locale, conversation and discourse; no read/write intent |
| Chat endpoints/request models | Same paths carry questions, corrections and commands; no enforced query mode |
| `SemanticFactRequest` | Created by the LLM after the mutation check, not available beforehand |
| Identity/location hints | Semantic hints, not validated mutation-routing decisions; prefix matches accept trailing write commands |
| Greeting shortcut | Already handles standalone greetings, not the requested factual queries |
| `enable_mutations=False` / no `write_item` | Already skips the call for agents without write capability; disabling it for the steward changes write behavior |

Concrete counterexamples: existing helpers produce hints for
`Who am I? Delete the lamp record.`, `我是谁，然后删除台灯的记录`, and
`Where is the lamp? Move it to the kitchen.` Absence of a write keyword, question
punctuation, prior read-only turns, or a valid read-schema output would not prove
that the original current request contains no write intent.

No always-false helper, question allowlist, regex classifier, client-supplied
intent flag or semantic-planner-first reordering was introduced. All supported
writes, corrections and ambiguous declarations retain the existing mutation path.
No unsafe-bypass candidate was sent to production.

## Measurement method

The small probe reuses `build_json_fact_service`, `SemanticFactPlanner`, existing
request tracing, latency summaries and provenance collection. It explicitly
enables mutation planning, but **never calls the fact executor or write
dispatcher**. Read, bilingual create/update/delete/location requests and the
specified ambiguous milk/fridge phrases are fixed in
`benchmarks/mutation_routing.yaml`.

The unchanged package was frozen and run on `jkuang@192.168.68.59`, in
`/tmp/hc-routing-ddaefc0f` inside `cortex-cortex-api-1`, with an explicit isolated
`PYTHONPATH` and copied graph catalog. The serving `/app` files and household facts
were untouched. Package, graph, schema and model fingerprints accompany the
summary; raw observations remain in that directory's `probe.json`.

These are **planning latency** measurements, not end-to-end API timings or
successful database mutation measurements. No accepted candidate exists, so
before/after improvement and unchanged final-answer correctness cannot be claimed.

## Baseline measurements (three passes, one excluded warm-up)

Model `qwen3.5:9b`, Ollama 0.32.15, 16,384 context and unchanged prompts.
See `baseline-summary.json` for per-query results, call counts and fingerprints.

| Metric | Before / retained route | Accepted fast path |
|---|---:|---:|
| LLM calls per obvious read | 2 (27/27 requests) | Not implemented |
| Mutation-planner calls per read | 1 | — |
| Semantic-planner calls per read | 1 | — |
| Median read planning latency | 2,536.426 ms | — |
| P95 read planning latency | 2,629.880 ms | — |
| Median mutation-call latency within reads | 933.233 ms | — |
| P95 mutation-call latency within reads | 960.440 ms | — |
| Explicit writes reaching mutation planner | 24/24 | Unchanged routing |
| Ambiguous inputs reaching mutation planner | 24/24 | Unchanged routing |

Explicit writes needed one call each (median planning 1,069.961 ms; p95
2,198.912 ms), yielding the expected create/update-location/update-attributes/delete
operation categories. Ambiguous inputs used 1–3 calls (median planning
2,178.811 ms; p95 4,146.504 ms). All 75 requests completed without a probe exception.
No final fact or write result was executed, so these are routing/plan observations,
not a final-answer correctness score or mutation-safety proof.

`冰箱里现在有牛奶`, `牛奶在冰箱里`, and `There is milk in the fridge now.`
compiled as creates. The Chinese correction compiled as a location update while
its English counterpart went to semantic read planning. `The milk is in the
fridge.` required two semantic attempts after mutation classification. These
are observed baseline behaviors, not new routing rules or changes. The kitchen
read also retained the existing incorrect household-root plan; successful parsing
is not equivalent to semantic correctness.

The extra call is a measured cost, but its removal is not a measured speedup:
there was no safe candidate and no post-change end-to-end benchmark.

## Validation and recommendation

Existing routing/API/write/semantic tests: **324 passed**. Full deterministic
suite: **768 passed**. No bypass-specific behavior tests were added because no
bypass was implemented. The pre-existing working-tree changes in
`tests/test_planner_prompt_audit.py` and `artifacts/capability-payload-compression/`
were preserved.

The only additions are the fixed routing inputs, a small planning-only probe,
usage documentation and this report/evidence. Further text-only gate work is not
justified under these constraints. Proceeding would first require an upstream,
trusted structured read intent with a defined contract; silently inferring one
from free text or disabling writes is not an acceptable optimization here.
