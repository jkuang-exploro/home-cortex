# Engineering utility consolidation

## Files moved

| Old path | New path |
|---|---|
| `scripts/tier1_latency_bench.py` | `scripts/benchmarks/tier1_latency_bench.py` |
| `scripts/token_latency_audit.py` | `scripts/profiling/token_latency_audit.py` |
| `scripts/token_component_probe.py` | `scripts/profiling/token_component_probe.py` |
| `scripts/http_latency_audit.py` | `scripts/profiling/http_latency_audit.py` |
| `scripts/profile_semantic_transport.py` | `scripts/profiling/profile_semantic_transport.py` |
| `scripts/ollama_prefix_reuse_probe.py` | `scripts/probes/ollama_prefix_reuse_probe.py` |
| `scripts/ollama_warm_load_probe.py` | `scripts/probes/ollama_warm_load_probe.py` |
| `scripts/kinship_context_probe.py` | `scripts/probes/kinship_context_probe.py` |
| `scripts/item_location_probe.py` | `scripts/probes/item_location_probe.py` |
| `scripts/bilingual_planner_probe.py` | `scripts/probes/bilingual_planner_probe.py` |
| `scripts/layer_failure_trace.py` | `scripts/probes/layer_failure_trace.py` |
| `scripts/copy_tier1_bench_into_api.sh` | `scripts/maintenance/copy_tier1_bench_into_api.sh` |
| `scripts/freeze_contract_candidate.py` | `scripts/maintenance/freeze_contract_candidate.py` |
| `scripts/context_surface_audit.py` | `scripts/maintenance/context_surface_audit.py` |
| `src/home_cortex/fact_benchmark.py` | `scripts/benchmarks/fact_benchmark.py` |
| `src/home_cortex/semantic_planner_benchmark.py` | `scripts/benchmarks/semantic_planner_benchmark.py` |
| `src/home_cortex/composition_eval.py` | `scripts/benchmarks/composition_eval.py` |
| `src/home_cortex/profiling.py` | `src/home_cortex/request_tracing.py` |
| `benchmarks/composition/_emit.py` | `scripts/maintenance/emit_composition.py` |

The graph-export CLI was split from `src/home_cortex/export.py` into
`scripts/maintenance/export_graph.py`; its reusable runtime implementation stays
in `export.py`.

## Retained runtime functionality

`request_tracing.py` is the renamed production `profiling.py`. It contains the
bounded, task-local request trace, stage decorators, model-usage capture, and
ASGI middleware used by the API, database, and model clients. Its behavior and
`CORTEX_PROFILE_REQUESTS` switch are unchanged. It is not a profiling runner.

`export.py` retains canonical graph reads, validation, serialization, and atomic
export-tree replacement used by the API. Only argument parsing, connection
lifecycle for the standalone command, and report printing moved.

## Shared logic and dependency direction

* `scripts/benchmarks/json_graph.py` owns `JsonGraphDispatcher`, extracted from
  the fact benchmark. Tests, benchmarks, and probes share this synthetic adapter
  without importing an executable harness just to build a fixture.
* `composition_eval.py` remains an importable evaluation-support module beside
  the benchmark harnesses. It has no runtime consumers and is shared by tests,
  the freezer, and dataset-generation tooling; it does not belong in production
  or exclusively in tests.
* Benchmark runners retain their importable scoring/run functions and CLI
  `main()` functions. No new framework or compatibility forwarding modules were
  introduced.
* Runtime modules have **no imports from `scripts`**. A regression test checks
  this boundary. Runtime code also has no remaining argparse/CLI main blocks.
* Cross-script imports are package imports. The warm-load probe's `sys.path`
  insertion was removed; module invocation and installation provide imports.

## Packaging, paths, and provenance

All three installed commands retain their names and arguments:
`home-cortex-fact-benchmark`, `home-cortex-semantic-planner-benchmark`, and
`home-cortex-db-export`. Their entrypoints now target the engineering package.
Wheels include both `home_cortex` and `scripts`; Docker copies the latter before
installation. Keeping the tooling in the distribution preserves these commands
without making runtime code depend on it.

Defaults locate checkout/frozen-package inputs or container `/app` inputs rather
than depending on the shell working directory. Explicit path options remain.
Deployment schemas and datasets remain external to the wheel, as before.
Planner CLI settings now load after argument parsing, so `--help` works without
a configured model. Actual benchmark runs still use the configured defaults.

The container-copy helper now copies the engineering package alongside the
runtime package and prints a module invocation. Frozen candidates include all
engineering Python/shell sources. Benchmark provenance separately fingerprints
the imported production package and benchmark harness tree; moving the runner
must not silently change a production-package hash into a scripts-only hash.

Active docs, tests, commands, imports, and the dataset generator were updated.
Historical artifact reports retain the paths of their original measured runs.
Composition fingerprints changed only for relocated import/documentation text
and the corresponding tree digest; gold cases, household fixtures, ontology,
and scoring revision are unchanged. The composition generator was moved and
syntax-checked, not executed to regenerate the frozen data.

## Validation

* Focused benchmark, profiling, composition, export, prefix, and new CLI/boundary
  tests: **69 passed**.
* Full suite: **730 passed in 15.53 seconds**.
* All 16 argparse commands pass `--help` from outside the checkout.
* Offline wheel build succeeded. An isolated installation runs all three console
  commands with `--help` from `/tmp`.
* Installed-wheel deterministic replay: **35/35 semantic matches and 35/35
  structured results**, with **zero model calls**.
* Frozen archive contains the moved modules and shared adapter; the extracted
  package's planner command runs from outside its directory.
* Shell syntax, all engineering Python syntax, and `git diff --check` pass.

No live inference, production database operation, or Docker deployment was run.
Docker packaging/copy changes were checked locally; the actual container-copy
operation was not executed against a running service.

## Resulting structure

```text
src/home_cortex/
  api.py, agent_service.py, semantic_*.py, ...
  request_tracing.py       # reusable runtime instrumentation
  export.py                # reusable runtime export
scripts/
  benchmarks/              # fact/planner/latency, composition, JSON adapter
  profiling/               # token/HTTP/transport audits
  probes/                  # focused diagnostic experiments
  maintenance/             # freezing, copying, export, context, data generation
benchmarks/                # unchanged dataset/fixture paths
tests/                    # runtime and engineering behavior coverage
```

See `scripts/README.md` for concise invocation guidance. Three engineering-only
modules leave the production package; runtime instrumentation is clearly named,
and standalone orchestration is separated from the export library. Discoverability
and dependency direction, rather than deleted LOC, are the result of this change.
