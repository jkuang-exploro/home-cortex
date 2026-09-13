# Engineering utilities

Runtime code lives in `src/home_cortex/`. These importable packages contain
engineering commands and their shared support; runtime modules never import them.

- `benchmarks/`: repeatable fact/planner/latency suites, composition evaluation,
  and the shared `json_graph.py` fixture adapter used by tests and probes.
- `profiling/`: token, latency, transport, and HTTP measurement runners.
- `probes/`: focused model/semantic diagnostics and failure tracing.
- `maintenance/`: package freezing, graph export, context auditing, container
  copying, and explicit composition-dataset regeneration (`emit_composition.py`).

Install with `pip install -e '.[dev]'`, then run from the repository root:

```sh
python -m scripts.benchmarks.fact_benchmark --help
python -m scripts.benchmarks.semantic_planner_benchmark --help
python -m scripts.profiling.token_latency_audit --mode replay --output /tmp/replay.json
python -m scripts.probes.ollama_prefix_reuse_probe --help
python -m scripts.maintenance.freeze_contract_candidate --output /tmp/candidate
```

The installed `home-cortex-fact-benchmark`,
`home-cortex-semantic-planner-benchmark`, and `home-cortex-db-export` commands
remain available. Wheels include `scripts` as a separate package; Docker copies
both packages and the existing schema/benchmark inputs. Use module invocation
instead of running detached script files. For an uninstalled checkout, run from
its root with `PYTHONPATH=src`; scripts never modify `sys.path`.

Default inputs resolve relative to the checkout/frozen archive or `/app` for
container installations. Installed wheels still need deployment schema/data
inputs; those are not embedded in the wheel. Explicit path flags remain supported.

`home_cortex.request_tracing` stays in production: it supplies request-scoped
instrumentation to the API, DB, and model clients, not a profiling CLI.
Historical artifact reports retain the paths used for their original runs.
The withdrawn compact semantic codec lives beside its offline profiler; it is
not an alternate runtime transport in `home_cortex`.

Planner prompt compression experiments:

```sh
python -m scripts.profiling.planner_prompt_audit --output /tmp/prompt-summary.json
python -m scripts.profiling.planner_prompt_audit --data-dir /path/to/frozen-data --ollama-url http://ollama:11434 --model qwen3.5:9b --output /tmp/native-components-summary.json
python -m scripts.benchmarks.planner_prompt_experiment --data-dir /path/to/frozen-data --repeat 3 --enforce-budgets --output /tmp/baseline-summary.json
python -m scripts.benchmarks.planner_prompt_experiment --data-dir /path/to/frozen-data --repeat 3 --drop 6 --drop 34 --enforce-budgets --output /tmp/candidate-a-summary.json
python -m scripts.benchmarks.planner_prompt_experiment --data-dir /path/to/frozen-data --regression --repeat 1 --output /tmp/regression-summary.json
```

Run live experiments only from an isolated frozen package on the GPU host, with
identical frozen data, schema, model and case fingerprints. Example indices refer
to that package's example inventory; inspect it before selecting a candidate.
The override exists only within the benchmark process and restores on failure.
Do not run competing model experiments concurrently. Each output path must be
new; its adjacent JSONL contains per-case plans, answers, timings and model calls.
Keep raw results on the evaluation host; commit concise `*-summary.json` files.

The fixed set has 12 requests; `--regression` adds the existing 20-case probe,
119 generalization utterances, and 20 bilingual utterances. Gold plans expand
through the canonical ontology. Answer comparisons execute gold plans against
the same frozen graph; they do not independently validate the executor or data.
Latency covers planner, deterministic JSON-graph execution, and rendering;
it excludes HTTP, mutation routing, conversation persistence, and live DB I/O.

Native raw component token counts exclude chat framing and are not additive.
Structured `format` schema bytes are reported separately: they constrain decoding
and are not added to message content. The live harness records every model call
and optionally fails on the measured 8,500-token normal budget, insufficient
16K context headroom for the 384-token output allocation, or a length stop.
These are observed-call guardrails, not a universal bound on arbitrary user text.
The offline 32,000-byte synthetic-fixture check is a separate prompt-creep alarm.
