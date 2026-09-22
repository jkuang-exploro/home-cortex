# Home Cortex benchmark harness

`hc-bench` records comparable model runs. It does not score on its own.
Each suite calls the existing runner, then stores that runner's metrics.

```bash
python -m home_cortex.benchmark --help
python -m home_cortex.benchmark run --help
python -m home_cortex.benchmark compare --help
```

After `pip install -e .` the same commands are available as `hc-bench`.

## Workflows

On the production GPU host (`home-cortex-0`), from the repository root:

```bash
hc-bench run --suite standard --model qwen3.5:9b --label current-production
hc-bench run --suite standard --model <candidate> --label candidate
hc-bench compare <baseline-run-id> <candidate-run-id>
hc-bench show <run-id>
hc-bench show <run-id> --failures
```

Narrower suites:

```bash
hc-bench run --suite fact --model qwen3.5:9b
hc-bench run --suite planner --model qwen3.5:9b
hc-bench run --suite latency --model qwen3.5:9b
hc-bench run --suite mutation --model qwen3.5:9b
hc-bench run --suite unified-planner --model qwen3.5:9b
hc-bench run --suite bilingual --model qwen3.5:9b
hc-bench run --suite full --model qwen3.5:9b
```

Name a baseline only when you mean to:

```bash
hc-bench baseline set production <run-id>
hc-bench compare production <candidate-run-id>
```

A matrix file runs one model and context length at a time. Nothing is scheduled
across machines. See `benchmarks/matrix.example.yaml`.

```bash
hc-bench matrix benchmarks/matrix.example.yaml
```

Change Ollama or pull a model yourself. The harness does not upgrade runtimes
or download weights.

## Where results go

```text
benchmarks/results/<run-id>/
  run.json
  summary.json
  cases.jsonl
  stdout.log
  progress.log
```

While a run is in progress, each case is logged to stderr and to
`progress.log` in the run directory (`...` when the model call starts, then
pass or fail and the latency when it returns). The printed report is rendered
from `summary.json` and `run.json`. There is no
second scoring pass and no single aggregate score. Plan correctness, answer
correctness, mutation payload, preview, commit, rejection, and multi-intent
stay separate. A faster model that breaks preview or commit exits 3.

## Ollama address

The project default is `http://ollama:11434`, the Compose service. That name
resolves inside the API container. On the GPU host the port is not published,
so `hc-bench` checks localhost and then the running Ollama container's bridge
address. It prints the address it actually used. `--ollama-url` is never
replaced. If nothing answers, the command exits without recording model scores.

## Host policy

Real-model suites declare `requires_gpu_host`. Off `home-cortex-0` the run
refuses. `--allow-nonstandard-host` records `nonstandard_environment = true`.
Extra hostnames can be added with `HC_BENCH_GPU_HOSTS` without changing code.
Dirty git trees are allowed and printed as `git_dirty = true`.

## Timing and cache

Semantic suites take one scoring pass. `--warmup` and `--repetitions` apply to
`latency` only (default warmup 1, repetitions 3). Warmup samples are excluded
from percentiles. The harness does not unload the model or flush caches.
`--cache-state` is a label (`warm`, `cold`, `unchanged`, `unknown`), not an
action. `--verified-cold` only labels the first latency request; unload the
model yourself before using it. `cold_load_ms` is Ollama's reported
`load_duration_ms` when the runtime provides it. Token counts are recorded
only when Ollama reports them. They are never estimated.

Comparisons warn when the corpus, prompt, configuration, commit, hostname, GPU,
cache state, or a dirty tree differ. Warnings still print the table.

## What is reused

| Asset | Location | What it measures | How the harness uses it |
| --- | --- | --- | --- |
| Fact benchmark | `scripts/benchmarks/fact_benchmark.py` | JSON-graph fact path, latency, pipeline completion | `fact` calls `benchmark_json`. No gold score. Not the household database. |
| Semantic planner benchmark | `scripts/benchmarks/semantic_planner_benchmark.py` and `benchmarks/semantic_planner_eval.yaml` | Plan correctness, answer correctness, validation | `planner` and the latency probe call the existing runner and copy `summarize_scores`. |
| Tier-1 latency probe | `scripts/benchmarks/tier1_latency_bench.py` (`run_tier1_probe`) | Warmup versus steady-state latency on the probe split | `latency` calls that function. |
| Unified planner experiment | `scripts/benchmarks/unified_planner_experiment.py` and `benchmarks/mutation_routing.yaml` | Classification, payload, preview, commit, multi-intent, paired reads | `mutation` runs the intents group. `unified-planner` runs intents and reads. Writes are not dispatched. |
| Bilingual probe | `scripts/probes/bilingual_planner_probe.py` and `benchmarks/semantic_planner_bilingual.yaml` | Cross-language plan match and parity | `bilingual` calls that probe. |
| Planner prompt experiment | `scripts/benchmarks/planner_prompt_experiment.py` | Prompt-budget regression on a frozen package | Not wrapped. Still the authority for prompt-drop experiments. |
| Composition eval | `benchmarks/composition/`, `scripts/benchmarks/composition_eval.py` | Executor gold on invented households | Test gate only. Not a model suite. |
| Conversation and greeting tests | `tests/test_open_webui_conversations.py`, `tests/test_greetings.py` | API conversation behavior | Test gate only. |
| Mutation routing probe | `scripts/probes/mutation_routing_probe.py` | Historical two-stage planning cost | Superseded for model comparison by the unified experiment. |
| Profiling and contract probes | `scripts/profiling/`, `scripts/probes/` | One-off diagnostics | Not model suites. |

`standard` is planner plus mutation. `full` adds fact, latency, and bilingual.
It does not rerun the unified reads corpus; that corpus repeats the planner
evaluation and is `unified-planner`.

No new semantic cases were added for this harness.
The old benchmark commands still call the same runners.

## Safety

Fact, planner, latency, and bilingual runs use the JSON graph or the
semantic-contract fixture. Mutation runs compile writes and do not dispatch
them. Do not point a suite at the live household database.
