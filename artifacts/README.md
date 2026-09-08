# artifacts/

**This directory holds generated benchmark results and prose reports. It is NOT
active source code.**

- `qwen35-semantic-contract/` and `tier1-baseline/` contain the output of
  semantic-planner / fact-benchmark runs on real LLMs, plus the investigation
  notes in `REPORT.md`.
- `static-prefix-reuse/` records cross-process serialization checks and isolated
  planner/chat/conversation prefill measurements (`REPORT.md` + summaries).
- `ollama-warm-load/` explains warm-request Ollama `load_duration` on a
  resident `qwen3.5:9b` runner (`REPORT.md` + `probe-summary.json`).
- The **readable, authoritative** provenance is the `*-summary.json` files and the
  `REPORT.md` / `REPORT-qwen35-9b.md` prose. Prefer these.
- The large per-case JSONs (e.g. `probe-iteration*.json`, `probe.json`,
  `suite.json`, `baseline-relevant-rows.json`) are **derived output**. They are
  **not in the working tree** (gitignored and removed). Do not read them wholesale.
  If you need one, restore it from git history:
  `git log --all -- <path>` to find a commit, then
  `git show <commit>:<path>` (e.g.
  `git show 45b8761:artifacts/tier1-baseline/probe.json`).
- Do not write new benchmark output here by default; keep generated results out
  of version control and summarize them in a small `*-summary.json` instead.
