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
