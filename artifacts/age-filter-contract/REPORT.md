# Age and calendar-range interpretation repair

Starting source: `021f1d2`. Date: 2026-09-07.

## Root causes

- Age requests compiled to a raw `select(birth_date)`, or even a name projection,
  instead of the existing `completed_years` transform.
- Calendar-year collection requests confused the filter property with the output
  projection. The resulting singular date lookup became ambiguous before
  filtering. Other plans compared a date to January 1 alone or to a year integer.
- The first candidate exposed a validation gap: request-level field filters
  accepted `value_from=anchor`, although the collection executor only consumes
  literal values. These conditions silently compared against null and could
  produce empty results. A zero-result fixture could accidentally score such an
  answer as correct; plan accuracy remained false.

## Changes

- Explain output shape, stored values, computed values, and half-open date
  intervals in the model-facing contract. Two illustrative compositions cover
  elapsed years for a different relative and date filtering on a residence edge;
  user/evaluation wording and household facts are not copied into the examples.
- Split collection-filter grammar from traversal-filter grammar. Collection
  field filters require an explicit operator and literal value. Traversal anchor
  comparisons remain supported. Runtime validation rejects dynamic collection
  filters before graph access; no unsupported condition is silently repaired.
- Empty filtered lists render as no matching records rather than suggesting
  that the household contains no member records.
- No new operation, ontology concept, question router, household-specific rule,
  storage mutation, or result-scoring adjustment.

## Deterministic validation

Local full suite: **364 passed**. Focused contract tests: **36 passed** after
adding the final three user paraphrases to the dataset.

Regressions cover completed-year boundaries on the birthday, raw dates versus
ages, two matching people, interval endpoints, household scope, empty results,
missing date evidence, and rejection of unsupported dynamic collection filters.
All fixtures are invented; runtime household records were not used as test data.

## Real-model experiments

Production GPU: `qwen3.5:9b`, digest
`6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`,
Ollama `0.32.13`, 100% GPU. Temperature 0, seed 0, think=false, 384 output-token
limit, keep_alive=24h. Frozen synthetic clock: 2026-09-07T12:00:00-07:00.

| Run | Cases | Plan matches | Correct answers | Planner P50 / P95 ms |
|---|---:|---:|---:|---:|
| Baseline, existing source | 21 | 5/21 | 6/21 | 2505 / 5962 |
| First candidate, before filter-grammar correction | 21 | 13/21 | 15/21 | 2134 / 3309 |
| Corrected candidate, first pass | 21 | 21/21 | 21/21 | 2152 / 2868 |
| Final candidate, expanded dataset, three passes | 72 (24 x 3) | 72/72 | 72/72 | 2132 / 2881 |

All three use identical invented data and the original 21-case dataset; each
has one excluded warm-up request. First-request timing is not verified-cold.
Each `*-summary.json` records exact model/data/schema/package/contract hashes.

Final repeated validation expands the dataset to 24 cases by adding the latest
three user utterances, without changing production code or scoring. The fixed
119-case suite is evaluated separately against read-only `/app/data`; suite
answer correctness remains unscored where expectations are absent.

Repeated synthetic validation is complete: all 24 cases passed in each of three
measured passes. There are no evaluation-only equivalence changes in these scores.
The three new cases include the user's latest verbatim age questions and the
punctuated calendar-year request.

### Existing fixed suite

**116/119 matching plans**, all scored. Planner P50/P95 **2107/2295 ms**.
Answer correctness remains unscored for all 119 cases; a `found` status is not
treated as answer correctness. Executor statuses: found 115, entity_not_found 2,
not_run 1, property_unavailable 1.

Remaining plan mismatches, outside the reported age/range failures:

- `self_identity::你知道我叫什么吗`: identity operation incorrectly includes a
  display-name property; rejected as `INVALID_PLAN`.
- `minor_count::有几个孩子`: chooses speaker children instead of the dataset's
  household-minor interpretation.
- `son_identity::哪个男孩是我儿子`: chooses a collection `select` rather than
  singular identity resolution.

These are recorded, not relabeled or silently repaired. This run does not prove
universal language generalization or absence of all interpretation regressions.

### Read-only production household check

Four reported questions, three measured passes: **12/12 matching plans and
12/12 correct answers**. Planner P50/P95 **2105/2823 ms**. Same isolated final
candidate, `/app/data` read-only, frozen 2026-09-07 clock:

- Both latest wife-age formulations return **38** completed years.
- Father-in-law age returns **64** completed years.
- The 1988 birth-year collection returns the expected two people, Jian and Pu.

This checks the semantic pipeline against production JSON data, not the live
HTTP conversation service or SurrealDB serving path. It does not deploy the fix.
The temporary `production-smoke.yaml` stays in `/tmp/hc-age-final`; its expected
values come from the reported household records, not from model output. It is
not a fixture for the local software suite.

## Reproduction and deployment boundary

Candidate code, schemas, scripts and benchmarks are staged under
`/tmp/hc-age-final` inside `cortex-cortex-api-1`. Run from that directory with
`PYTHONPATH=/tmp/hc-age-final/src`:

```sh
python scripts/tier1_latency_bench.py \
  --ollama-url http://ollama:11434 --model qwen3.5:9b \
  --data-dir benchmarks/fixtures/semantic-contract --schema-dir schemas/edge \
  --eval benchmarks/semantic_planner_age_filters.yaml \
  --warmup 1 --repeat 3 --progress --output /tmp/hc-age-final/probe-final.json

python scripts/tier1_latency_bench.py \
  --ollama-url http://ollama:11434 --model qwen3.5:9b \
  --data-dir /app/data --schema-dir schemas/edge \
  --eval benchmarks/semantic_planner_eval.yaml --suite \
  --warmup 1 --repeat 1 --progress --output /tmp/hc-age-final/suite-final.json
```

Full generated per-case outputs remain in the isolated production directory;
only summaries are retained here. The running application and live data were
not changed. Production deployment still requires the normal rebuild/restart.
