# benchmarks/results/

`hc-bench run` writes one immutable directory per run here:

```text
<run-id>/
  run.json
  summary.json
  cases.jsonl
  stdout.log
  progress.log
```

Named baselines live in `baselines.json` in this directory. Promotion is explicit
(`hc-bench baseline set`). Runs are generated output and are not committed.
The household database is never a benchmark target.

See `benchmarks/HARNESS.md` for the commands.
