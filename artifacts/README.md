# artifacts/

**This directory holds generated benchmark results and prose reports. It is NOT
active source code.**

- `qwen35-semantic-contract/` and `tier1-baseline/` contain the output of
  semantic-planner / fact-benchmark runs on real LLMs, plus the investigation
  notes in `REPORT.md`.
- The **readable, authoritative** provenance is the `*-summary.json` files and the
  `REPORT.md` / `REPORT-qwen35-9b.md` prose. Prefer these.
- The large per-case JSONs (e.g. `probe-iteration*.json`, `probe.json`,
  `suite.json`, `baseline-relevant-rows.json`) are **derived output**. They are
  gitignored (see `.gitignore`) and should **not** be read wholesale; they are
  preserved on disk and in git history only for on-demand deep inspection.
- Do not write new benchmark output here by default; keep generated results out
  of version control and summarize them in a small `*-summary.json` instead.
