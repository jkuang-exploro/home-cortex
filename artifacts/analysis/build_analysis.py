"""Rebuild the derived comparison from immutable files under artifacts/.

Run: python3 artifacts/analysis/build_analysis.py
Eight hc-bench exports (one archived in git history) form the cross-model cohort.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SOURCE_FILES = sorted(p for p in ROOT.rglob("*") if p.is_file() and OUT not in p.parents)
EXPORTS = sorted(ROOT.glob("*.yaml"))
BASELINE = "20260924-022915-4063"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_export(path: Path, *, raw_override: str | None = None, source_override: str | None = None) -> dict:
    raw = path.read_text() if raw_override is None else raw_override
    sections: dict[str, dict] = {}
    current = "metadata"
    sections[current] = {}
    for line in raw.splitlines():
        if line in {"Semantic results", "Latency", "Failures", "Tokens", "Gates", "Timing policy", "Notes"}:
            current = line.lower().replace(" ", "_")
            sections[current] = {}
            continue
        match = re.match(r"^\s*([^:]+):\s*(.*)$", line)
        if match and current not in {"gates", "timing_policy", "notes"}:
            sections[current][match.group(1).strip()] = match.group(2).strip()
        elif current in {"gates", "timing_policy", "notes"} and line.strip():
            sections[current].setdefault("lines", []).append(line.strip())

    meta = sections["metadata"]
    metrics = {}
    for label, value in sections["semantic_results"].items():
        if value == "n/a":
            metrics[label] = None
        elif (m := re.fullmatch(r"(\d+)/(\d+)", value)):
            metrics[label] = {"correct": int(m[1]), "scored": int(m[2])}
        else:
            metrics[label] = value
    latency = {}
    for label, value in sections["latency"].items():
        if (m := re.fullmatch(r"([\d.]+) / ([\d.]+) ms", value)):
            latency[label] = {"p50_ms": float(m[1]), "p95_ms": float(m[2])}
        else:
            try:
                latency[label] = float(value)
            except ValueError:
                latency[label] = value
    failures = {k: int(v) for k, v in sections["failures"].items()}
    tokens = {k: (v == "true" if k == "estimated" else int(v)) for k, v in sections["tokens"].items()}
    gates = {}
    for line in sections["gates"].get("lines", []):
        m = re.fullmatch(r"(pass|fail): (.+) \((\d+)\)", line)
        if m:
            gates[m[2]] = {"status": m[1], "count": int(m[3])}
    return {
        "source": source_override or str(path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "run_id": meta["Run ID"],
        "label": meta["Label"],
        "suite": meta["Suite"],
        "model": meta["Model"],
        "model_tag": meta["Model tag"],
        "model_digest": meta["Model digest"],
        "ollama_url": meta["Ollama URL"],
        "ollama_version": meta["Ollama version"],
        "quantization": meta["Quantization"],
        "model_context_length": int(meta["Context length"]),
        "requested_num_ctx": int(meta["Requested num_ctx"]),
        "home_cortex_commit": meta["Home Cortex commit"],
        "branch": meta["Branch"],
        "git_dirty": "git_dirty = true" in raw,
        "nonstandard_environment": "nonstandard_environment = true" in raw,
        "prompt_fingerprint": meta["Prompt fingerprint"],
        "corpus_fingerprint": meta["Corpus fingerprint"],
        "config_fingerprint": meta["Config fingerprint"],
        "cache_state": meta["Cache state"],
        "original_results_path": meta["Results"],
        "original_progress_path": meta["Progress"],
        "hardware": None,
        "start_clock_from_run_id_timezone_unrecorded": meta["Run ID"][:8] + "T" + meta["Run ID"][9:15],
        "metrics": metrics,
        "latency_ms": latency,
        "failures": failures,
        "tokens": tokens,
        "gates": gates,
        "timing_policy": sections.get("timing_policy", {}).get("lines", []),
        "notes": sections.get("notes", {}).get("lines", []),
        "raw_finished_line": next((line for line in reversed(raw.splitlines()) if "[hc-bench] finished" in line), None),
    }


runs = [parse_export(p) for p in EXPORTS]
archived = subprocess.run(["git", "show", "c88a85c:artifacts/Ministral.yaml"], cwd=ROOT.parent, capture_output=True, text=True, check=True).stdout
archived_run = parse_export(ROOT / "Ministral.yaml", raw_override=archived, source_override="git:c88a85c:artifacts/Ministral.yaml (absent from working tree)")
runs.append(archived_run)
runs.sort(key=lambda r: r["run_id"])
assert len(runs) == 8 and len({r["run_id"] for r in runs}) == 8
by_id = {r["run_id"]: r for r in runs}
baseline = by_id[BASELINE]
abliterated = by_id["20260924-065027-6e58"]
core_fields = ["suite", "ollama_url", "ollama_version", "quantization", "requested_num_ctx", "home_cortex_commit", "prompt_fingerprint", "corpus_fingerprint", "config_fingerprint"]


def delta(candidate: dict) -> dict:
    result = {"correctness": {}, "latency": {}, "failures": {}, "reliability": {}, "tokens": {}}
    for name, b in baseline["metrics"].items():
        c = candidate["metrics"].get(name)
        if isinstance(b, dict) and isinstance(c, dict) and b["scored"] == c["scored"]:
            result["correctness"][name] = c["correct"] - b["correct"]
    for name in ("min", "p50", "p95", "max"):
        b, c = baseline["latency_ms"][name], candidate["latency_ms"][name]
        result["latency"][name] = {"ms": round(c - b, 3), "percent": round(100 * (c / b - 1), 1)}
    for name, b in baseline["failures"].items():
        result["failures"][name] = candidate["failures"][name] - b
    result["reliability"] = {
        "failure_count_deltas": result["failures"],
        "gate_count_deltas": {name: candidate["gates"][name]["count"] - baseline["gates"][name]["count"] for name in baseline["gates"]},
    }
    for name in ("prompt", "output", "total"):
        result["tokens"][name] = candidate["tokens"][name] - baseline["tokens"][name]
    return result


catalog = []
historical_runs = []
for p in SOURCE_FILES:
    item = {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha(p)}
    if p.suffix == ".json":
        try:
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                item["top_level_keys"] = list(data)
                for key in ("mode", "model", "ollama_version", "run_id"):
                    if isinstance(data.get(key), (str, int, float)):
                        item[key] = data[key]
                provenance = data.get("provenance")
                if isinstance(provenance, dict) and provenance.get("model_name"):
                    fields = ("model_name", "model_digest", "ollama_version", "git_commit", "git_dirty", "host_git_commit", "host_git_dirty", "dataset_path", "eval_sha256", "data_tree_sha256", "schema_tree_sha256", "copied_package_sha256", "planner_contract_sha256", "scoring_revision", "hardware", "request_settings", "warmup", "repeat", "verified_cold")
                    record = {"reference": str(p.relative_to(ROOT)), "run_id": data.get("run_id"), "mode": data.get("mode"), "suite_or_group": data.get("group"), "provenance": {k: provenance[k] for k in fields if k in provenance}, "scores": data.get("scores")}
                    historical_runs.append(record)
        except (UnicodeError, json.JSONDecodeError):
            item["parse_error"] = True
    catalog.append(item)
catalog.append({"path": archived_run["source"], "bytes": len(archived.encode()), "sha256": archived_run["source_sha256"], "availability": "git commit c88a85c only; absent from working tree"})

comparisons = []
for r in runs:
    if r["run_id"] == BASELINE:
        continue
    diffs = {f: [baseline[f], r[f]] for f in core_fields if baseline[f] != r[f]}
    comparisons.append({
        "baseline_run_id": BASELINE, "candidate_run_id": r["run_id"],
        "classification": "COMPARABLE WITH CAVEAT" if not diffs else "NOT DIRECTLY COMPARABLE",
        "recorded_core_differences": diffs,
        "caveats": ["git_dirty=true for both; unrecorded dirty diff", "cache_state=unknown", "hardware telemetry absent in these exports", "single sequential scoring pass for semantic suites"],
        "deltas": delta(r),
    })

historical = []
for directory in sorted(p for p in ROOT.iterdir() if p.is_dir() and p != OUT):
    files = [p for p in SOURCE_FILES if directory in p.parents]
    historical.append({"study": directory.name, "files": [str(p.relative_to(ROOT)) for p in files], "classification_vs_full_cohort": "NOT DIRECTLY COMPARABLE", "reason": "Different study purpose, corpus, prompt, code, runtime, scoring, or incomplete matching provenance; consult study report and source summaries."})

payload = {
    "schema": "home-cortex-derived-benchmark-analysis-v1",
    "source_scope": "artifacts/ excluding analysis/",
    "working_tree_source_file_count": len(SOURCE_FILES),
    "archived_source_file_count": 1,
    "source_file_count": len(catalog),
    "source_types_working_tree": dict(Counter(p.suffix for p in SOURCE_FILES)),
    "source_catalog": catalog,
    "runs_discovered": runs,
    "historical_studies": historical,
    "historical_run_summaries": historical_runs,
    "comparability_groups": [{"name": "2026-09-23/24 hc-bench full cohort", "run_ids": [r["run_id"] for r in runs], "baseline": BASELINE, "class_vs_baseline": "COMPARABLE WITH CAVEAT", "matched_fields": core_fields}],
    "baseline": BASELINE,
    "interpretation": {
        "current_default": "qwen3.5:9b retains the stronger measured mutation rejection and far lower latency in this cohort",
        "capability_exploration_candidate": "qwen3.8:27b leads this cohort on planner correctness, latency-probe answer correctness, mutation payload, and preview correctness despite Home Cortex development primarily using qwen3.5:9b",
        "capability_caveat": "One full pass per model, absent per-case overlap and 27B rejection 4/8 versus 9B 8/8 prevent a general accuracy or production-safety claim",
        "hardware_question": "27B p50 and p95 are much slower in the recorded setup; no GPU residency or offload telemetry was retained, so the effect of better hardware is unmeasured",
        "abliterated_9b": "The huihui-ai variant is much less accurate on this matched benchmark; changed weights and published model-package defaults/template are confounded, so the exported run cannot isolate abliteration itself as the cause",
    },
    "abliterated_9b_review": {
        "run_id": abliterated["run_id"],
        "baseline_run_id": BASELINE,
        "deltas": delta(abliterated),
        "observed_pattern": "Broad semantic losses across planner, bilingual, latency-probe, and multi-intent metrics; zero malformed output, timeout, runtime, and harness errors",
        "published_package_observations": {
            "source": "https://ollama.com/huihui_ai/qwen3.5-abliterated:9b",
            "digest_matches_run": True,
            "go_template": "{{ .Prompt }}",
            "params": {"temperature": 1, "top_k": 20, "top_p": 0.95},
            "baseline_model_params_source": "https://ollama.com/library/qwen3.5:9b",
            "baseline_model_params": {"presence_penalty": 1.5, "temperature": 1, "top_k": 20, "top_p": 0.95},
            "interpretation_limit": "Benchmark exports do not record Ollama's selected renderer or the final effective options. Home Cortex overrides temperature to 0 for planners; model package differences remain potential confounders.",
        },
    },
    "candidate_metrics": {r["run_id"]: {"model": r["model"], "metrics": r["metrics"], "latency_ms": r["latency_ms"], "failures": r["failures"], "tokens": r["tokens"]} for r in runs},
    "comparisons": comparisons,
    "notable_regressions": [
        {"run_id": r["run_id"], "model": r["model"], "mutation_safety": {k: r["metrics"][k] for k in ("Rejection correctness", "Multi-intent handling")}, "gates": r["gates"]}
        for r in runs if r["run_id"] != BASELINE and (r["metrics"]["Rejection correctness"]["correct"] < 8 or r["metrics"]["Multi-intent handling"]["correct"] < 7)
    ],
    "notable_improvements": [{"run_id": r["run_id"], "model": r["model"], "planner_delta": delta(r)["correctness"]["Plan correctness"], "payload_delta": delta(r)["correctness"]["Mutation payload"]} for r in runs if r["run_id"] != BASELINE and (delta(r)["correctness"]["Plan correctness"] > 0 or delta(r)["correctness"]["Mutation payload"] > 0)],
    "repeatability": {"model": "ministral-3:8b", "run_ids": [archived_run["run_id"], "20260924-050048-3420"], "delta_later_minus_earlier": None},
    "missing_data": ["Eight original run.json, summary.json, cases.jsonl and progress.log files are referenced by absolute paths outside artifacts and are not present here", "Per-case overlap and per-language accuracy for eight full runs", "Within-run latency samples and variance", "Time to first token and generation time for eight full runs", "GPU/CPU/RAM/VRAM/offload telemetry for eight full runs", "Verified cold-start condition", "Repeated full runs for other model/config combinations", "Git dirty diff and exact host identity for eight full runs", "Ollama selected renderer and final effective model options for the abliterated/base pair"],
}

earlier = archived_run
later = by_id["20260924-050048-3420"]
payload["repeatability"]["delta_later_minus_earlier"] = {
    "correctness": {name: later["metrics"][name]["correct"] - earlier["metrics"][name]["correct"] for name, value in earlier["metrics"].items() if isinstance(value, dict)},
    "p50_ms": round(later["latency_ms"]["p50"] - earlier["latency_ms"]["p50"], 3),
    "p50_percent": round(100 * (later["latency_ms"]["p50"] / earlier["latency_ms"]["p50"] - 1), 1),
    "p95_ms": round(later["latency_ms"]["p95"] - earlier["latency_ms"]["p95"], 3),
    "p95_percent": round(100 * (later["latency_ms"]["p95"] / earlier["latency_ms"]["p95"] - 1), 1),
    "failures": {name: later["failures"][name] - earlier["failures"][name] for name in earlier["failures"]},
}

(OUT / "benchmark-analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

def ratio(run, name):
    v = run["metrics"][name]
    return f'{v["correct"]}/{v["scored"]}' if v else "n/a"


short = {"qwen3.8:27b": "27B", "qwen3.5:9b": "9B", "huihui_ai/qwen3.5-abliterated:9b": "Abliterated 9B", "qwen3.5:4b": "4B", "gemma4:12b": "Gemma 12B", "gemma4:e2b": "Gemma e2b", "ministral-3:8b": "Ministral 8B"}
lines = [
    "# Home Cortex recorded benchmark analysis", "",
    "## Findings", "",
    "Eight `hc-bench full` exports support a qualified comparison: seven files currently under `artifacts/` and one earlier Ministral export recovered read-only from git history. `qwen3.5:9b` is the strongest practical default in this cohort: 108/119 planner cases, 21/21 mutation classification, 8/8 rejection, 7/7 multi-intent, and 1.462 s median request latency. `qwen3.8:27b` leads on planner correctness (111/119), latency-probe answer correctness (19/20 versus 17/20), mutation payload (14/14 versus 12/14), and preview payload (5/5 versus 3/5). Since Home Cortex development has primarily used 9B, these results make 27B a worthwhile capability exploration candidate. Its 4/8 rejection result versus 9B's 8/8 is a separate mutation-safety regression. Its 14.757 s median and 57.744 s p95 are measured costs on the tested setup; the exports do not reveal whether faster hardware would remove a CPU/offload bottleneck. The new `huihui_ai/qwen3.5-abliterated:9b` run falls to 87/119 planner and 44/59 bilingual matches from the base 9B's 108/119 and 57/59; it also fails multi-intent handling on 3/7. This is a broad semantic regression without runtime or malformed-output failures. `gemma4:e2b` is fastest (0.567 s median) but fails the multi-intent safety gate (5 partial plans) and has 75/119 planner correctness. The 4B run saves 0.150 s at p50 versus 9B while losing 10 planner cases and failing the multi-intent gate. Only Ministral has a second full run; small differences for other models have no direct repeatability estimate.", "",
    "The exported `Cases passed` count mixes distinct suite case types and is retained as a native metric, not treated as an overall intelligence score. Planner suite answer correctness is unscored (`n/a`); the separate latency sample has answer scores. Fact 38/38 means the pipeline completed, not that 38 answers were correct. Mutation writes were compiled but never dispatched.", "",
    "## Source inventory and run identity", "",
    f"Discovered {len(SOURCE_FILES)} working-tree files under `artifacts/` before writing this analysis: {', '.join(f'{n} {s}' for s,n in sorted(payload['source_types_working_tree'].items()))}. One earlier tracked export, `Ministral.yaml`, is absent from the working tree; its `c88a85c` git blob is included read-only, giving eight `hc-bench full` runs. The complete path, size, SHA-256, and available JSON keys are in `benchmark-analysis.json` (`source_catalog`). There are {len(historical)} earlier study directories and {len(historical_runs)} older JSON summaries with explicit model provenance, indexed individually under `historical_run_summaries` with available model, runtime, corpus, package, prompt, hardware, and scoring fields. They lack a common run-ID convention, so their source paths are the references. No `run.json`, `cases.jsonl`, `stdout.log`, or `progress.log` for the eight exports is stored here. Their absolute result paths point to another machine and were not read. Earlier studies also contain reports, package archives, and scripts. The two empty directories contain no recorded evidence.", "",
    "| Run ID | Model | Digest | Ollama | Context requested / model | Commit | Suite |",
    "|---|---|---|---|---|---|---|",
]
for r in runs:
    lines.append(f"| {r['run_id']}{' (git history only)' if r is archived_run else ''} | `{r['model']}` | `{r['model_digest'][:12]}…` | {r['ollama_version']} | {r['requested_num_ctx']:,} / {r['model_context_length']:,} | `{r['home_cortex_commit'][:12]}…` dirty | {r['suite']} |")
lines += ["", "All eight report Q4_K_M, the same Ollama URL, prompt fingerprint `673683aa…`, corpus fingerprint `9c507ba8…`, and config fingerprint `716da25a…`. Full values, model digests, and source SHA-256 are in the JSON. Labels are `-`; host hardware is not recorded in these exports. The run ID encodes the host's local start clock, whose timezone is not recorded; seven final log lines give finish clocks without dates. `gemma4:e2b` has a 131,072 model context limit; all eight requested 16,384.", "",
    "## Comparability", "",
    "**COMPARABLE WITH CAVEAT:** each of the seven candidate exports against 9B, including two Ministral runs. Suite, corpus/prompt/config fingerprints, commit, requested context, quantization, and Ollama 0.34.2 match. Model digests change across models, and published package options can change with them. Every checkout is dirty, the dirty diff is absent, cache state is `unknown`, hardware telemetry is absent, and model runs occurred sequentially. The semantic scores are reasonable matched-cohort evidence; latency percentages are descriptive, not controlled hardware or cache effects. The models have different native context limits but identical requested context. For the abliterated pair, the published model package options differ and the variant publishes a Go template; the benchmark fingerprints do not capture those package fields, so the score gap cannot be attributed solely to modified weights.", "",
    "**NOT DIRECTLY COMPARABLE:** all older studies versus this full cohort. They include qwen3:8b on CPU and qwen3.5:9b on GPU with Ollama 0.32.13, 4B warm-load work on 0.32.15, prompt/code experiments, synthetic probes, differing contexts (often 8192), and different scoring scopes. Their separate findings are described below. A matched same-model `full` repeat exists only for Ministral; no Ollama-version or context-length pair exists in these eight exports, so model, runtime, and context effects cannot be separated further.", "",
    "## Correctness by native metric", "",
    "| Model | Plan | Latency probe plan / answer | Bilingual plan / parity | Mutation classify / payload | Preview / commit / reject / multi-intent | Cases passed |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for r in runs:
    lines.append(f"| {short[r['model']]}{' earlier' if r is archived_run else ''} | {ratio(r,'Plan correctness')} | {ratio(r,'Latency-sample plan correctness')} / {ratio(r,'Latency-sample answer correctness')} | {ratio(r,'Bilingual plan match')} / {ratio(r,'Bilingual parity')} | {ratio(r,'Mutation classify')} / {ratio(r,'Mutation payload')} | {ratio(r,'Preview correctness')} / {ratio(r,'Commit correctness')} / {ratio(r,'Rejection correctness')} / {ratio(r,'Multi-intent handling')} | {ratio(r,'Cases passed')} |")
lines += ["", "The 9B baseline has two preview payload failures (3/5) despite zero preview-as-commit events. A `pass` on that gate only means the dangerous mode confusion was not observed. The rejection metric counts ambiguous requests whose decision was not mutation; it is distinct from multi-intent handling. Denominators are small, especially preview (5), rejection (8), and multi-intent (7).", "",
    "## Latency and tokens", "",
    "Overall request latencies below are milliseconds; 334 measured samples per export, with `min` and `max` across mixed suites. Percentages are relative to 9B and inherit the comparability caveats.", "",
    "| Model | min | p50 | p95 | max | p50 vs 9B | p95 vs 9B | Prompt / output tokens |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for r in runs:
    d = delta(r)["latency"]
    l = r["latency_ms"]
    t = r["tokens"]
    lines.append(f"| {short[r['model']]}{' earlier' if r is archived_run else ''} | {l['min']:,.0f} | {l['p50']:,.0f} | {l['p95']:,.0f} | {l['max']:,.0f} | {d['p50']['percent']:+.1f}% | {d['p95']['percent']:+.1f}% | {t['prompt']:,} / {t['output']:,} |")
lines += ["", "Per-suite p50 / p95 (ms), preserving each suite's own timing policy:", "",
    "| Model | Planner | Mutation | Fact | Steady-state latency probe | Bilingual |",
    "|---|---:|---:|---:|---:|---:|",
]
for r in runs:
    def pair(suite):
        value = r["latency_ms"][f"{suite} p50/p95"]
        return f"{value['p50_ms']:,.0f} / {value['p95_ms']:,.0f}"
    lines.append(f"| {short[r['model']]}{' earlier' if r is archived_run else ''} | {pair('planner')} | {pair('mutation')} | {pair('fact')} | {pair('latency')} | {pair('bilingual')} |")
lines += ["", "The steady-state `latency` component excludes warmups and uses measured repetitions. Planner, fact, and bilingual components each use one scoring pass; the planner percentiles may include the first request. Mutation discards two warmups per route. The exports do not preserve individual latency samples, so variance, outliers' identities, and confidence intervals cannot be recomputed. The `cold model load` field is Ollama's first reported load duration, **not a verified cold start**; the 5.455 s e2b value must not be mixed into warm inference. Prompt and output token totals are Ollama-reported, not estimated, but cover mixed requests and cannot be normalized reliably per successful case without the missing case rows. No time-to-first-token or separate generation duration is recorded here. Older warm-load reports are a different runtime experiment.", "",
    "## Reliability and mutation safety", "",
    "| Model | Semantic mismatch | Validation | Malformed | Partial multi-intent plans | Preview-as-commit |",
    "|---|---:|---:|---:|---:|---:|",
]
for r in runs:
    f = r["failures"]
    lines.append(f"| {short[r['model']]}{' earlier' if r is archived_run else ''} | {f['semantic_mismatch']} | {f['validation_failure']} | {f['malformed_structured_output']} | {r['gates']['Multi-intent turn produced a partial plan']['count']} | {r['gates']['Preview was compiled as commit']['count']} |")
lines += ["", "All eight report zero timeouts, context overflows, Ollama runtime errors, provider/tool errors, and benchmark harness failures. Thus recorded failures are semantic or structured-output/validation failures, not harness outages. 4B and both Ministral runs each produce four partial multi-intent plans; the abliterated 9B produces three and e2b produces five. Their speed or nominal mutation payload counts do not offset that safety regression. Gemma 12B has 2/8 rejection, 7/9 commit, and one malformed output. 27B has 4/8 rejection despite 14/14 mutation payload and 5/5 preview. No write was committed in these experiments.", "",
    "## Per-case and bilingual limits", "",
    "The eight exports contain only aggregate metrics and failure counts. There is no retained case ID/status matrix in `artifacts/`, so all-pass/all-fail sets, candidate wins/regressions, individual failure clusters, and English/Chinese/mixed rates **cannot be computed for this cohort**. The 57/59 versus 44/59 base/abliterated bilingual match gap is a thirteen-case aggregate gap; it does not identify which languages or cases changed. Bilingual parity (10 paired items) is a separate metric.", "",
    "A separate historical 9B prompt experiment (`bilingual-planner`) did retain language aggregates on 56 cases repeated three times: old prompt first pass 20/23 Chinese, 25/29 English, 4/4 mixed; bilingual prompt 21/23, 27/29, 4/4. That is a **prompt change at the same model**, not evidence for any candidate in the eight-run model comparison. The study's report notes a scoring alignment that revises the old prompt's fair overall first-pass score from 49/56 to 51/56; the raw JSON retains 49/56. Its case IDs and named misses are in its report. Historical `kinship-context` also records a specific 9B repeated Chinese in-law interpretation error and a separate 4B prompt comparison; their prompts and contexts differ from the current full cohort.", "",
    "## Candidate tradeoffs versus 9B", "",
]
for c in comparisons:
    r = by_id[c["candidate_run_id"]]
    d = c["deltas"]
    lines.append(f"### {r['model']} — {r['run_id']}")
    lines.append("")
    lines.append(f"Planner {d['correctness']['Plan correctness']:+d}/119; bilingual plan {d['correctness']['Bilingual plan match']:+d}/59; mutation classification {d['correctness']['Mutation classify']:+d}/21, payload {d['correctness']['Mutation payload']:+d}/14, rejection {d['correctness']['Rejection correctness']:+d}/8, multi-intent {d['correctness']['Multi-intent handling']:+d}/7. Median {d['latency']['p50']['percent']:+.1f}%, p95 {d['latency']['p95']['percent']:+.1f}%; validation failures {d['failures']['validation_failure']:+d}, malformed outputs {d['failures']['malformed_structured_output']:+d}. See the metric table for preview, commit, and latency-probe results.")
    lines.append("")
lines += [
    "## Why the abliterated 9B scores poorly", "",
    "The new run (`20260924-065027-6e58`) has the same recorded Home Cortex commit, suite, prompt and corpus fingerprints, requested 16,384 context, Q4_K_M quantization, and Ollama 0.34.2 as base 9B. Its model digest differs. Relative to base 9B it loses 21/119 planner cases, 13/59 bilingual plan matches, 6/20 latency-probe plan cases, 5/20 latency-probe answer cases, 4/21 mutation classification cases, and 3/7 multi-intent cases. It has three partial multi-intent plans. Semantic mismatches rise 16→56 and validation failures 2→7, while malformed outputs, timeouts, provider/runtime errors, and harness failures remain zero. Thus the observed problem is semantic interpretation and mutation intent, not an Ollama outage or general JSON failure. Fact completion remains 38/38, which does not score fact answers.", "",
    "The published model package matching digest `92a443adb124…` identifies the model as an abliterated derivative of Qwen3.5-9B and shows a Go template of `{{ .Prompt }}` with parameters `temperature=1`, `top_k=20`, and `top_p=0.95` ([model page](https://ollama.com/huihui_ai/qwen3.5-abliterated:9b), [template](https://ollama.com/huihui_ai/qwen3.5-abliterated:9b/blobs/b507b9c2f6ca)). The base Ollama tag additionally publishes `presence_penalty=1.5` ([base model](https://ollama.com/library/qwen3.5:9b)). Home Cortex's planner request overrides temperature to zero and sends `think=false`, `num_ctx=16384`, seed, output limit, and a JSON schema; it does not override presence penalty. Ollama 0.34.2 merges model options with request options ([source](https://github.com/ollama/ollama/blob/v0.34.2/server/routes.go)). The recorded prompt/config fingerprints cover Home Cortex's prompt and requested settings, not the model package template or effective model options. The export does not record which renderer Ollama selected, so the template difference is a **possible confound**, not a proven cause. The altered weights and missing base presence penalty are also plausible contributors; their separate effects are unmeasured. Abliteration targets refusal behavior, not this semantic-planning task, and its publisher describes the implementation as a proof of concept ([model card](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-9B-abliterated)).", "",
    "Latency does not explain the correctness loss: overall p50 is 1,416.620 versus 1,462.417 ms (3.1% faster), while p95 is 2,910.700 versus 2,350.052 ms (23.9% slower). The reported first `load_duration` is 8.054 s for the variant, but verified cold loading was not established and it should not be interpreted as normal request latency. Without the original case rows or a controlled package-options/template comparison, the exact mechanism cannot be identified.", "",
    "## Repeatability in the retained evidence", "",
    "The earlier tracked Ministral export (`20260923-054939-c75d`, now absent from the working tree) and the later present export (`20260924-050048-3420`) share model digest, runtime, corpus, prompt, config, and commit fingerprints. Planner stays 104/119, rejection 8/8, and multi-intent 3/7 with four partial plans in both. Payload moves 11/14 to 12/14; commit moves 8/9 to 9/9; cases passed moves 233/265 to 234/265. Overall p50 moves 3,844.503 to 4,076.784 ms (+6.0%); p95 moves 6,986.275 to 7,065.109 ms (+1.1%). This is one repeat pair, not a variance distribution and not a substitute for 9B/4B repeats. It shows a one-case mutation payload change can occur with the nominally same configuration.", "",
    "## Practical roles and uncertainty", "",
    "- **Best balanced, retained default: `qwen3.5:9b`.** It has the strongest rejection and multi-intent results with low latency among safety-gate passers. It is also 10/119 planner cases ahead of 4B at only 150 ms higher median. This is a cohort finding, not a production rollout decision.",
    "- **Best capability exploration candidate: `qwen3.8:27b`.** It leads on planner (+3/119), latency-probe answer (+2/20), mutation payload (+2/14), and preview payload (+2/5), despite the system being developed mostly with 9B. These aligned improvements justify testing more capable models. They do not prove a repeatable general accuracy gain from one pass. Rejection is four cases worse, and p50 is about ten times 9B on the recorded setup; those issues need separate case-level and hardware investigation before default deployment.",
    "- **Fastest measured: `gemma4:e2b`.** Its 0.567 s p50 is coupled to large correctness losses and five partial multi-intent plans. It is not an acceptable smaller production candidate on this evidence.",
    "- **Closest latency alternative: `qwen3.5:4b`.** Median improves 10.2%, while p95 worsens 13.9%, planner loses 10 cases, and four partial multi-intent plans fail the gate. Its latency benefit is modest against the semantic and safety losses.",
    "- **Abliterated 9B: unsuitable for this planner on the recorded run.** It loses 21 planner cases and three multi-intent cases versus base 9B, with no meaningful median latency benefit. The comparison does not isolate the changed weights from model-package options or rendering.",
    "- **Ministral 8B and Gemma 12B:** both have lower planner and mutation safety results than 9B and much higher p50/p95 on this cohort; each is dominated by 9B on recorded correctness, latency, and reliability. The strict dominance label is limited to these measured dimensions and this single matched cohort.",
    "Only Ministral has a same-configuration `full` repeat. No base/abliterated 9B, 9B/4B, or 9B/27B correctness or p50/p95 repeat variance exists. The abliterated gap is broad across metrics, but its repeatability and exact cause remain unmeasured. Smaller differences such as 108 versus 111 planner passes, 57 versus 56 bilingual passes, and 1.462 versus 1.313 s p50 cannot be declared repeatable. The other historical repeats test different prompts, model/runtime combinations, or narrower case sets; they do not supply full-cohort variance.", "",
    "## Resource and runtime effects", "",
    "The eight exports include no VRAM, RAM, GPU utilization, CPU utilization, or offload fields. GPU residency and resource efficiency therefore cannot be ranked from these runs. Historical `ollama-warm-load` reports separately measured RTX 2060 SUPER 8 GiB: 9B at about 6518 MiB on Ollama 0.32.13 and 4B at about 4066 MiB on 0.32.15, both resident at 8192 context. Those observations cannot establish residency for these 16,384-context Ollama 0.34.2 runs. That same historical comparison changes both model and Ollama version, so the warm-load decrease is not attributable to model size alone.", "",
    "## Smallest useful next measurement", "",
    "1. First recover the eight existing `benchmarks/results/<run-id>/` directories, if available, and copy only their run metadata, summary, and case rows into a **new** derived/transfer location. This is evidence recovery, not a benchmark rerun; it would resolve case overlap, language splits, warm/request token distributions, and exact provenance. For the abliterated pair, inspect its wrong plans and record Ollama's selected renderer and effective model options.",
    "2. For the 27B capability question, test 9B and 27B on the same proposed higher-capacity host with the same prompt, corpus, Ollama version, and 16,384 requested context; capture per-case results, GPU residency/offload, VRAM, and p50/p95 over at least three full runs each. This would test whether the accuracy lead repeats and how much of the measured latency changes with hardware. The four 27B rejection misses should be inspected before any production-default decision.",
    "3. If the specific decision is whether 4B can be a safe small default, first fix and then retest the four recorded partial multi-intent cases with the unchanged mutation suite; do not infer safety from the faster aggregate latency. For the abliterated 9B, a focused paired diagnostic with matched effective options and renderer would test package effects before another full benchmark.", "",
    "## Verification", "",
    "`build_analysis.py` parses every top-level export plus the archived tracked Ministral blob, hashes and catalogs every pre-existing artifact file, asserts eight unique run IDs, computes deltas only when denominators match, and generates this Markdown and JSON from the same in-memory records. Original artifacts were read only. No benchmark was run. The JSON preserves full run IDs and every extracted native metric; the Markdown rounds latency and percentages for display. The earlier deterministic suite result was 977 passed and one unrelated failure: `test_fingerprints_match_current_tree` could not find `benchmarks/composition/codex-approval.md`; the suite was not rerun for this analysis update.", "",
]
(OUT / "benchmark-analysis.md").write_text("\n".join(lines))
