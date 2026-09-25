# Home Cortex recorded benchmark analysis

## Findings

Eight `hc-bench full` exports support a qualified comparison: seven files currently under `artifacts/` and one earlier Ministral export recovered read-only from git history. `qwen3.5:9b` is the strongest practical default in this cohort: 108/119 planner cases, 21/21 mutation classification, 8/8 rejection, 7/7 multi-intent, and 1.462 s median request latency. `qwen3.8:27b` leads on planner correctness (111/119), latency-probe answer correctness (19/20 versus 17/20), mutation payload (14/14 versus 12/14), and preview payload (5/5 versus 3/5). Since Home Cortex development has primarily used 9B, these results make 27B a worthwhile capability exploration candidate. Its 4/8 rejection result versus 9B's 8/8 is a separate mutation-safety regression. Its 14.757 s median and 57.744 s p95 are measured costs on the tested setup; the exports do not reveal whether faster hardware would remove a CPU/offload bottleneck. The new `huihui_ai/qwen3.5-abliterated:9b` run falls to 87/119 planner and 44/59 bilingual matches from the base 9B's 108/119 and 57/59; it also fails multi-intent handling on 3/7. This is a broad semantic regression without runtime or malformed-output failures. `gemma4:e2b` is fastest (0.567 s median) but fails the multi-intent safety gate (5 partial plans) and has 75/119 planner correctness. The 4B run saves 0.150 s at p50 versus 9B while losing 10 planner cases and failing the multi-intent gate. Only Ministral has a second full run; small differences for other models have no direct repeatability estimate.

The exported `Cases passed` count mixes distinct suite case types and is retained as a native metric, not treated as an overall intelligence score. Planner suite answer correctness is unscored (`n/a`); the separate latency sample has answer scores. Fact 38/38 means the pipeline completed, not that 38 answers were correct. Mutation writes were compiled but never dispatched.

## Source inventory and run identity

Discovered 153 working-tree files under `artifacts/` before writing this analysis: 5 .gz, 105 .json, 30 .md, 5 .py, 1 .txt, 7 .yaml. One earlier tracked export, `Ministral.yaml`, is absent from the working tree; its `c88a85c` git blob is included read-only, giving eight `hc-bench full` runs. The complete path, size, SHA-256, and available JSON keys are in `benchmark-analysis.json` (`source_catalog`). There are 29 earlier study directories and 49 older JSON summaries with explicit model provenance, indexed individually under `historical_run_summaries` with available model, runtime, corpus, package, prompt, hardware, and scoring fields. They lack a common run-ID convention, so their source paths are the references. No `run.json`, `cases.jsonl`, `stdout.log`, or `progress.log` for the eight exports is stored here. Their absolute result paths point to another machine and were not read. Earlier studies also contain reports, package archives, and scripts. The two empty directories contain no recorded evidence.

| Run ID | Model | Digest | Ollama | Context requested / model | Commit | Suite |
|---|---|---|---|---|---|---|
| 20260923-054939-c75d (git history only) | `ministral-3:8b` | `1922accd5827…` | 0.34.2 | 16,384 / 262,144 | `66e33d1d7a56…` dirty | full |
| 20260923-193829-7c87 | `qwen3.8:27b` | `22130167c4c2…` | 0.34.2 | 16,384 / 262,144 | `66e33d1d7a56…` dirty | full |
| 20260924-022915-4063 | `qwen3.5:9b` | `6488c96fa5fa…` | 0.34.2 | 16,384 / 262,144 | `66e33d1d7a56…` dirty | full |
| 20260924-025024-38af | `qwen3.5:4b` | `2a654d98e6fb…` | 0.34.2 | 16,384 / 262,144 | `66e33d1d7a56…` dirty | full |
| 20260924-033219-7a8f | `gemma4:12b` | `4eb23ef187e2…` | 0.34.2 | 16,384 / 262,144 | `66e33d1d7a56…` dirty | full |
| 20260924-043534-f998 | `gemma4:e2b` | `7fbdbf8f5e45…` | 0.34.2 | 16,384 / 131,072 | `66e33d1d7a56…` dirty | full |
| 20260924-050048-3420 | `ministral-3:8b` | `1922accd5827…` | 0.34.2 | 16,384 / 262,144 | `66e33d1d7a56…` dirty | full |
| 20260924-065027-6e58 | `huihui_ai/qwen3.5-abliterated:9b` | `92a443adb124…` | 0.34.2 | 16,384 / 262,144 | `66e33d1d7a56…` dirty | full |

All eight report Q4_K_M, the same Ollama URL, prompt fingerprint `673683aa…`, corpus fingerprint `9c507ba8…`, and config fingerprint `716da25a…`. Full values, model digests, and source SHA-256 are in the JSON. Labels are `-`; host hardware is not recorded in these exports. The run ID encodes the host's local start clock, whose timezone is not recorded; seven final log lines give finish clocks without dates. `gemma4:e2b` has a 131,072 model context limit; all eight requested 16,384.

## Comparability

**COMPARABLE WITH CAVEAT:** each of the seven candidate exports against 9B, including two Ministral runs. Suite, corpus/prompt/config fingerprints, commit, requested context, quantization, and Ollama 0.34.2 match. Model digests change across models, and published package options can change with them. Every checkout is dirty, the dirty diff is absent, cache state is `unknown`, hardware telemetry is absent, and model runs occurred sequentially. The semantic scores are reasonable matched-cohort evidence; latency percentages are descriptive, not controlled hardware or cache effects. The models have different native context limits but identical requested context. For the abliterated pair, the published model package options differ and the variant publishes a Go template; the benchmark fingerprints do not capture those package fields, so the score gap cannot be attributed solely to modified weights.

**NOT DIRECTLY COMPARABLE:** all older studies versus this full cohort. They include qwen3:8b on CPU and qwen3.5:9b on GPU with Ollama 0.32.13, 4B warm-load work on 0.32.15, prompt/code experiments, synthetic probes, differing contexts (often 8192), and different scoring scopes. Their separate findings are described below. A matched same-model `full` repeat exists only for Ministral; no Ollama-version or context-length pair exists in these eight exports, so model, runtime, and context effects cannot be separated further.

## Correctness by native metric

| Model | Plan | Latency probe plan / answer | Bilingual plan / parity | Mutation classify / payload | Preview / commit / reject / multi-intent | Cases passed |
|---|---:|---:|---:|---:|---:|---:|
| Ministral 8B earlier | 104/119 | 16/20 / 17/20 | 53/59 / 9/10 | 17/21 / 11/14 | 3/5 / 8/9 / 8/8 / 3/7 | 233/265 |
| 27B | 111/119 | 17/20 / 19/20 | 56/59 / 10/10 | 21/21 / 14/14 | 5/5 / 9/9 / 4/8 / 7/7 | 247/265 |
| 9B | 108/119 | 17/20 / 17/20 | 57/59 / 10/10 | 21/21 / 12/14 | 3/5 / 9/9 / 8/8 / 7/7 | 247/265 |
| 4B | 98/119 | 14/20 / 14/20 | 56/59 / 10/10 | 16/21 / 9/14 | 3/5 / 6/9 / 7/8 / 3/7 | 225/265 |
| Gemma 12B | 103/119 | 16/20 / 16/20 | 57/59 / 10/10 | 20/21 / 10/14 | 3/5 / 7/9 / 2/8 / 7/7 | 233/265 |
| Gemma e2b | 75/119 | 11/20 / 12/20 | 45/59 / 7/10 | 12/21 / 8/14 | 2/5 / 6/9 / 7/8 / 0/7 | 183/265 |
| Ministral 8B | 104/119 | 16/20 / 17/20 | 53/59 / 9/10 | 17/21 / 12/14 | 3/5 / 9/9 / 8/8 / 3/7 | 234/265 |
| Abliterated 9B | 87/119 | 11/20 / 12/20 | 44/59 / 8/10 | 17/21 / 11/14 | 3/5 / 8/9 / 8/8 / 4/7 | 202/265 |

The 9B baseline has two preview payload failures (3/5) despite zero preview-as-commit events. A `pass` on that gate only means the dangerous mode confusion was not observed. The rejection metric counts ambiguous requests whose decision was not mutation; it is distinct from multi-intent handling. Denominators are small, especially preview (5), rejection (8), and multi-intent (7).

## Latency and tokens

Overall request latencies below are milliseconds; 334 measured samples per export, with `min` and `max` across mixed suites. Percentages are relative to 9B and inherit the comparability caveats.

| Model | min | p50 | p95 | max | p50 vs 9B | p95 vs 9B | Prompt / output tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ministral 8B earlier | 743 | 3,845 | 6,986 | 17,105 | +162.9% | +197.3% | 2,458,459 / 14,117 |
| 27B | 6,145 | 14,757 | 57,744 | 72,514 | +909.1% | +2357.1% | 2,330,830 / 12,765 |
| 9B | 743 | 1,462 | 2,350 | 7,681 | +0.0% | +0.0% | 2,296,209 / 12,767 |
| 4B | 757 | 1,313 | 2,677 | 5,079 | -10.2% | +13.9% | 2,307,913 / 12,966 |
| Gemma 12B | 1,567 | 4,503 | 6,585 | 17,302 | +207.9% | +180.2% | 2,427,734 / 13,814 |
| Gemma e2b | 225 | 567 | 1,143 | 8,440 | -61.2% | -51.4% | 2,499,211 / 14,337 |
| Ministral 8B | 740 | 4,077 | 7,065 | 12,191 | +178.8% | +200.6% | 2,458,402 / 14,115 |
| Abliterated 9B | 877 | 1,417 | 2,911 | 15,349 | -3.1% | +23.9% | 2,319,120 / 13,386 |

Per-suite p50 / p95 (ms), preserving each suite's own timing policy:

| Model | Planner | Mutation | Fact | Steady-state latency probe | Bilingual |
|---|---:|---:|---:|---:|---:|
| Ministral 8B earlier | 3,896 / 5,788 | 4,812 / 8,704 | 3,735 / 5,118 | 3,921 / 5,893 | 3,764 / 5,612 |
| 27B | 14,791 / 57,803 | 14,068 / 39,442 | 14,636 / 58,350 | 14,623 / 17,895 | 14,601 / 58,019 |
| 9B | 1,465 / 1,792 | 1,419 / 3,222 | 1,416 / 1,594 | 1,450 / 1,766 | 1,493 / 1,985 |
| 4B | 1,322 / 2,640 | 1,319 / 4,218 | 1,275 / 1,503 | 1,312 / 2,584 | 1,313 / 1,517 |
| Gemma 12B | 4,419 / 5,872 | 4,093 / 8,437 | 4,819 / 5,466 | 4,798 / 6,286 | 4,486 / 5,643 |
| Gemma e2b | 536 / 770 | 650 / 1,745 | 713 / 844 | 571 / 770 | 551 / 742 |
| Ministral 8B | 4,147 / 6,235 | 4,804 / 8,814 | 3,991 / 5,487 | 4,182 / 6,355 | 3,957 / 5,922 |
| Abliterated 9B | 1,422 / 2,556 | 1,429 / 4,525 | 1,357 / 1,545 | 1,396 / 1,892 | 1,424 / 2,026 |

The steady-state `latency` component excludes warmups and uses measured repetitions. Planner, fact, and bilingual components each use one scoring pass; the planner percentiles may include the first request. Mutation discards two warmups per route. The exports do not preserve individual latency samples, so variance, outliers' identities, and confidence intervals cannot be recomputed. The `cold model load` field is Ollama's first reported load duration, **not a verified cold start**; the 5.455 s e2b value must not be mixed into warm inference. Prompt and output token totals are Ollama-reported, not estimated, but cover mixed requests and cannot be normalized reliably per successful case without the missing case rows. No time-to-first-token or separate generation duration is recorded here. Older warm-load reports are a different runtime experiment.

## Reliability and mutation safety

| Model | Semantic mismatch | Validation | Malformed | Partial multi-intent plans | Preview-as-commit |
|---|---:|---:|---:|---:|---:|
| Ministral 8B earlier | 28 | 4 | 0 | 4 | 0 |
| 27B | 18 | 0 | 0 | 0 | 0 |
| 9B | 16 | 2 | 0 | 0 | 0 |
| 4B | 33 | 7 | 0 | 4 | 0 |
| Gemma 12B | 28 | 3 | 1 | 0 | 0 |
| Gemma e2b | 71 | 5 | 6 | 5 | 0 |
| Ministral 8B | 27 | 4 | 0 | 4 | 0 |
| Abliterated 9B | 56 | 7 | 0 | 3 | 0 |

All eight report zero timeouts, context overflows, Ollama runtime errors, provider/tool errors, and benchmark harness failures. Thus recorded failures are semantic or structured-output/validation failures, not harness outages. 4B and both Ministral runs each produce four partial multi-intent plans; the abliterated 9B produces three and e2b produces five. Their speed or nominal mutation payload counts do not offset that safety regression. Gemma 12B has 2/8 rejection, 7/9 commit, and one malformed output. 27B has 4/8 rejection despite 14/14 mutation payload and 5/5 preview. No write was committed in these experiments.

## Per-case and bilingual limits

The eight exports contain only aggregate metrics and failure counts. There is no retained case ID/status matrix in `artifacts/`, so all-pass/all-fail sets, candidate wins/regressions, individual failure clusters, and English/Chinese/mixed rates **cannot be computed for this cohort**. The 57/59 versus 44/59 base/abliterated bilingual match gap is a thirteen-case aggregate gap; it does not identify which languages or cases changed. Bilingual parity (10 paired items) is a separate metric.

A separate historical 9B prompt experiment (`bilingual-planner`) did retain language aggregates on 56 cases repeated three times: old prompt first pass 20/23 Chinese, 25/29 English, 4/4 mixed; bilingual prompt 21/23, 27/29, 4/4. That is a **prompt change at the same model**, not evidence for any candidate in the eight-run model comparison. The study's report notes a scoring alignment that revises the old prompt's fair overall first-pass score from 49/56 to 51/56; the raw JSON retains 49/56. Its case IDs and named misses are in its report. Historical `kinship-context` also records a specific 9B repeated Chinese in-law interpretation error and a separate 4B prompt comparison; their prompts and contexts differ from the current full cohort.

## Candidate tradeoffs versus 9B

### ministral-3:8b — 20260923-054939-c75d

Planner -4/119; bilingual plan -4/59; mutation classification -4/21, payload -1/14, rejection +0/8, multi-intent -4/7. Median +162.9%, p95 +197.3%; validation failures +2, malformed outputs +0. See the metric table for preview, commit, and latency-probe results.

### qwen3.8:27b — 20260923-193829-7c87

Planner +3/119; bilingual plan -1/59; mutation classification +0/21, payload +2/14, rejection -4/8, multi-intent +0/7. Median +909.1%, p95 +2357.1%; validation failures -2, malformed outputs +0. See the metric table for preview, commit, and latency-probe results.

### qwen3.5:4b — 20260924-025024-38af

Planner -10/119; bilingual plan -1/59; mutation classification -5/21, payload -3/14, rejection -1/8, multi-intent -4/7. Median -10.2%, p95 +13.9%; validation failures +5, malformed outputs +0. See the metric table for preview, commit, and latency-probe results.

### gemma4:12b — 20260924-033219-7a8f

Planner -5/119; bilingual plan +0/59; mutation classification -1/21, payload -2/14, rejection -6/8, multi-intent +0/7. Median +207.9%, p95 +180.2%; validation failures +1, malformed outputs +1. See the metric table for preview, commit, and latency-probe results.

### gemma4:e2b — 20260924-043534-f998

Planner -33/119; bilingual plan -12/59; mutation classification -9/21, payload -4/14, rejection -1/8, multi-intent -7/7. Median -61.2%, p95 -51.4%; validation failures +3, malformed outputs +6. See the metric table for preview, commit, and latency-probe results.

### ministral-3:8b — 20260924-050048-3420

Planner -4/119; bilingual plan -4/59; mutation classification -4/21, payload +0/14, rejection +0/8, multi-intent -4/7. Median +178.8%, p95 +200.6%; validation failures +2, malformed outputs +0. See the metric table for preview, commit, and latency-probe results.

### huihui_ai/qwen3.5-abliterated:9b — 20260924-065027-6e58

Planner -21/119; bilingual plan -13/59; mutation classification -4/21, payload -1/14, rejection +0/8, multi-intent -3/7. Median -3.1%, p95 +23.9%; validation failures +5, malformed outputs +0. See the metric table for preview, commit, and latency-probe results.

## Why the abliterated 9B scores poorly

The new run (`20260924-065027-6e58`) has the same recorded Home Cortex commit, suite, prompt and corpus fingerprints, requested 16,384 context, Q4_K_M quantization, and Ollama 0.34.2 as base 9B. Its model digest differs. Relative to base 9B it loses 21/119 planner cases, 13/59 bilingual plan matches, 6/20 latency-probe plan cases, 5/20 latency-probe answer cases, 4/21 mutation classification cases, and 3/7 multi-intent cases. It has three partial multi-intent plans. Semantic mismatches rise 16→56 and validation failures 2→7, while malformed outputs, timeouts, provider/runtime errors, and harness failures remain zero. Thus the observed problem is semantic interpretation and mutation intent, not an Ollama outage or general JSON failure. Fact completion remains 38/38, which does not score fact answers.

The published model package matching digest `92a443adb124…` identifies the model as an abliterated derivative of Qwen3.5-9B and shows a Go template of `{{ .Prompt }}` with parameters `temperature=1`, `top_k=20`, and `top_p=0.95` ([model page](https://ollama.com/huihui_ai/qwen3.5-abliterated:9b), [template](https://ollama.com/huihui_ai/qwen3.5-abliterated:9b/blobs/b507b9c2f6ca)). The base Ollama tag additionally publishes `presence_penalty=1.5` ([base model](https://ollama.com/library/qwen3.5:9b)). Home Cortex's planner request overrides temperature to zero and sends `think=false`, `num_ctx=16384`, seed, output limit, and a JSON schema; it does not override presence penalty. Ollama 0.34.2 merges model options with request options ([source](https://github.com/ollama/ollama/blob/v0.34.2/server/routes.go)). The recorded prompt/config fingerprints cover Home Cortex's prompt and requested settings, not the model package template or effective model options. The export does not record which renderer Ollama selected, so the template difference is a **possible confound**, not a proven cause. The altered weights and missing base presence penalty are also plausible contributors; their separate effects are unmeasured. Abliteration targets refusal behavior, not this semantic-planning task, and its publisher describes the implementation as a proof of concept ([model card](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-9B-abliterated)).

Latency does not explain the correctness loss: overall p50 is 1,416.620 versus 1,462.417 ms (3.1% faster), while p95 is 2,910.700 versus 2,350.052 ms (23.9% slower). The reported first `load_duration` is 8.054 s for the variant, but verified cold loading was not established and it should not be interpreted as normal request latency. Without the original case rows or a controlled package-options/template comparison, the exact mechanism cannot be identified.

## Repeatability in the retained evidence

The earlier tracked Ministral export (`20260923-054939-c75d`, now absent from the working tree) and the later present export (`20260924-050048-3420`) share model digest, runtime, corpus, prompt, config, and commit fingerprints. Planner stays 104/119, rejection 8/8, and multi-intent 3/7 with four partial plans in both. Payload moves 11/14 to 12/14; commit moves 8/9 to 9/9; cases passed moves 233/265 to 234/265. Overall p50 moves 3,844.503 to 4,076.784 ms (+6.0%); p95 moves 6,986.275 to 7,065.109 ms (+1.1%). This is one repeat pair, not a variance distribution and not a substitute for 9B/4B repeats. It shows a one-case mutation payload change can occur with the nominally same configuration.

## Practical roles and uncertainty

- **Best balanced, retained default: `qwen3.5:9b`.** It has the strongest rejection and multi-intent results with low latency among safety-gate passers. It is also 10/119 planner cases ahead of 4B at only 150 ms higher median. This is a cohort finding, not a production rollout decision.
- **Best capability exploration candidate: `qwen3.8:27b`.** It leads on planner (+3/119), latency-probe answer (+2/20), mutation payload (+2/14), and preview payload (+2/5), despite the system being developed mostly with 9B. These aligned improvements justify testing more capable models. They do not prove a repeatable general accuracy gain from one pass. Rejection is four cases worse, and p50 is about ten times 9B on the recorded setup; those issues need separate case-level and hardware investigation before default deployment.
- **Fastest measured: `gemma4:e2b`.** Its 0.567 s p50 is coupled to large correctness losses and five partial multi-intent plans. It is not an acceptable smaller production candidate on this evidence.
- **Closest latency alternative: `qwen3.5:4b`.** Median improves 10.2%, while p95 worsens 13.9%, planner loses 10 cases, and four partial multi-intent plans fail the gate. Its latency benefit is modest against the semantic and safety losses.
- **Abliterated 9B: unsuitable for this planner on the recorded run.** It loses 21 planner cases and three multi-intent cases versus base 9B, with no meaningful median latency benefit. The comparison does not isolate the changed weights from model-package options or rendering.
- **Ministral 8B and Gemma 12B:** both have lower planner and mutation safety results than 9B and much higher p50/p95 on this cohort; each is dominated by 9B on recorded correctness, latency, and reliability. The strict dominance label is limited to these measured dimensions and this single matched cohort.
Only Ministral has a same-configuration `full` repeat. No base/abliterated 9B, 9B/4B, or 9B/27B correctness or p50/p95 repeat variance exists. The abliterated gap is broad across metrics, but its repeatability and exact cause remain unmeasured. Smaller differences such as 108 versus 111 planner passes, 57 versus 56 bilingual passes, and 1.462 versus 1.313 s p50 cannot be declared repeatable. The other historical repeats test different prompts, model/runtime combinations, or narrower case sets; they do not supply full-cohort variance.

## Resource and runtime effects

The eight exports include no VRAM, RAM, GPU utilization, CPU utilization, or offload fields. GPU residency and resource efficiency therefore cannot be ranked from these runs. Historical `ollama-warm-load` reports separately measured RTX 2060 SUPER 8 GiB: 9B at about 6518 MiB on Ollama 0.32.13 and 4B at about 4066 MiB on 0.32.15, both resident at 8192 context. Those observations cannot establish residency for these 16,384-context Ollama 0.34.2 runs. That same historical comparison changes both model and Ollama version, so the warm-load decrease is not attributable to model size alone.

## Smallest useful next measurement

1. First recover the eight existing `benchmarks/results/<run-id>/` directories, if available, and copy only their run metadata, summary, and case rows into a **new** derived/transfer location. This is evidence recovery, not a benchmark rerun; it would resolve case overlap, language splits, warm/request token distributions, and exact provenance. For the abliterated pair, inspect its wrong plans and record Ollama's selected renderer and effective model options.
2. For the 27B capability question, test 9B and 27B on the same proposed higher-capacity host with the same prompt, corpus, Ollama version, and 16,384 requested context; capture per-case results, GPU residency/offload, VRAM, and p50/p95 over at least three full runs each. This would test whether the accuracy lead repeats and how much of the measured latency changes with hardware. The four 27B rejection misses should be inspected before any production-default decision.
3. If the specific decision is whether 4B can be a safe small default, first fix and then retest the four recorded partial multi-intent cases with the unchanged mutation suite; do not infer safety from the faster aggregate latency. For the abliterated 9B, a focused paired diagnostic with matched effective options and renderer would test package effects before another full benchmark.

## Verification

`build_analysis.py` parses every top-level export plus the archived tracked Ministral blob, hashes and catalogs every pre-existing artifact file, asserts eight unique run IDs, computes deltas only when denominators match, and generates this Markdown and JSON from the same in-memory records. Original artifacts were read only. No benchmark was run. The JSON preserves full run IDs and every extracted native metric; the Markdown rounds latency and percentages for display. The earlier deterministic suite result was 977 passed and one unrelated failure: `test_fingerprints_match_current_tree` could not find `benchmarks/composition/codex-approval.md`; the suite was not rerun for this analysis update.
