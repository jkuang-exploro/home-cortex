# Generic date intervals with explicit units

Starting revision: `2c7d151`. Validation clock: 2026-09-07.

## Diagnosis

The reported marriage duration produced 4509 without a unit, while the marriage
date lookup was correct. Source inspection showed three coupled defects:
`duration` could only choose days/seconds; the interpreter instructed duration
questions to default to days; and `completed_years` presentation assumed every
input was a person's age. Date records themselves were not the missing data.

## Architecture changes

- One model-facing operation, `date_difference`, evaluates an entity or final
  relationship date against the trusted current time. `mode` explicitly selects
  years, months, days or seconds. Property ownership remains explicit.
- Calendar years/months count full anniversaries, not days divided by 365/30.
  Intervals are signed and truncated toward zero for those calendar units.
  Invalid/missing units fail validation. Annual occurrence retains its distinct
  date-or-days contract.
- Structured `FactResult` retains numeric `value` and adds `unit`; benchmark
  serialization preserves it and new cases score the unit as well as the value.
- The renderer distinguishes age presentation from generic elapsed time by
  semantic property/ownership and carries units into both Chinese and English.
  No marriage question matching, source-code phrase routing or household identity
  mapping was added.
- The old `completed_years` and `duration` structured operations remain documented
  compatibility entry points, sharing the same generic implementation. The
  interpreter cannot choose those aliases.
- Adult/minor predicate definitions now declare the canonical transform and year
  mode, retaining a past-date requirement so future dates remain invalid inputs.

This is an interval from one resolved date to the contextual clock, not a claim
that arbitrary two-date expressions or fractional calendar durations are supported.

## Evaluation migration

Existing duration expectations now use `date_difference` with the same day mode;
age expectations use `date_difference` with year mode. Wording, expected numeric
answers and acceptable relationship references were not loosened. Historical
artifacts are unchanged. Scoring revision:
`2026-09-07.1-explicit-date-interval-units`.

`benchmarks/semantic_planner_date_intervals.yaml` adds 16 invented-household
cases across property owners and units. Its wording is separate from interpreter
examples. Real-household smoke expectations use the reported 2014-05-04 date
and remain in a temporary production-only YAML rather than local test fixtures.

## Validation

Local software tests: **386 passed**. Coverage includes anniversaries, leap dates,
short months, negative intervals, time-of-day boundaries, property ownership,
unit serialization/scoring, age versus raw dates and bilingual presentation.

Production GPU: qwen3.5:9b, original model digest, 100% GPU, temperature 0,
seed 0, think=false, 384 output tokens, keep_alive=24h. Source, data, schema,
model and interpretation-contract fingerprints are in each summary.

| Run | Plan accuracy | Answer accuracy (including units when declared) | Planner P50 / P95 ms |
|---|---:|---:|---:|
| Synthetic date intervals, 16 cases | 16/16 | 16/16 | 2093 / 2191 |
| Previous synthetic age/range probe | 24/24 | 24/24 | 2182 / 2845 |
| Read-only real household, 5 cases x 3 passes | 15/15 | 15/15 | 2141 / 2202 |
| Existing fixed suite, 119 cases | 113/119 | Unscored (no answer expectations) | 2099 / 2274 |

The repeated household check returned **12 years**, **148 months**, or **4509
days** for the same marriage start, according to the requested unit. Marriage
date remained **2014-05-04** and wife age remained **38 years**. Both numeric
values and interval units were scored. The candidate was not deployed.

The fixed suite has six plan mismatches. Four unspecified-unit marriage duration
questions produce `mode=years`, whereas both the unchanged expected unit and the
interpreter's default-unit instruction require days. These return valid interval
results, but remain failures under the declared contract; no alternative year
plans were added to the expectations. The remaining two failures concern self
identity with an incompatible property and son identity compiled as collection
selection. Explicit day requests and all relationship-date lookups pass.

Compared with the previous age/filter run's 116/119, this is a net decrease of
three plan matches: four new default-unit mismatches and one previously failing
case that now matches. This is not a claim of overall planner accuracy improvement.
Choosing the default unit for unspecified durations remains an interpreter
limitation. The suite does not score answer correctness, and its `found` statuses
must not be treated as correct answers. In particular, it also records two
`entity_not_found` results and one `property_unavailable` result.

## Isolation and reproduction

All real-LLM runs use `/tmp/hc-date-intervals` inside `cortex-cortex-api-1`, with
`PYTHONPATH=/tmp/hc-date-intervals/src`, `scripts/tier1_latency_bench.py`,
`--ollama-url http://ollama:11434 --model qwen3.5:9b --schema-dir schemas/edge`.
Every run excludes one warm-up request and records that cold state was not verified.

- Date interval and previous age/range probes use
  `--data-dir benchmarks/fixtures/semantic-contract`, their corresponding eval
  YAML, and one measured pass.
- Real-household smoke uses `--data-dir /app/data --eval production-smoke.yaml`
  and three measured passes, reading production data without modifying it.
- The existing fixed suite uses `--data-dir /app/data`,
  `--eval benchmarks/semantic_planner_eval.yaml --suite`, one measured pass.

Full generated JSON stays in the isolated production directory; only summaries
are retained here. The running application and authoritative data are unchanged.
Deployment requires the normal source sync and image rebuild/restart.
