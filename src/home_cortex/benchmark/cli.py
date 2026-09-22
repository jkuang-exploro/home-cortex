"""``hc-bench`` command line.

Scoring is not implemented here. ``run`` delegates to registered suite adapters,
then prints the summary that was written to disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .compare import build_comparison
from .environment import CACHE_STATES, DEFAULT_OLLAMA_URL
from .present import format_comparison, format_failures, format_run_report
from .records import load_baselines, load_cases, load_run, resolve_run, results_root, write_baseline
from .registry import registry
from .runner import execute, plan_matrix
from .types import RunRequest

_EXAMPLES = """
examples:
  hc-bench list
  hc-bench run --suite standard --model qwen3.5:9b
  hc-bench run --suite standard --model qwen3.5:9b --label production-baseline
  hc-bench run --suite fact --model qwen3.5:9b
  hc-bench run --suite planner --model qwen3.5:9b
  hc-bench run --suite latency --model qwen3.5:9b
  hc-bench show RUN_ID
  hc-bench show RUN_ID --failures
  hc-bench compare BASELINE_RUN CANDIDATE_RUN
  hc-bench baseline set production RUN_ID
  hc-bench compare production CANDIDATE_RUN
  hc-bench matrix benchmarks/matrix.example.yaml

Real-model suites run on the production GPU host (home-cortex-0). On any other
machine the command refuses unless --allow-nonstandard-host is set. That override
records nonstandard_environment = true. The harness does not upgrade Ollama,
download models, or write to the household database.

Exit codes: 0 recorded, 1 harness failure, 2 usage or wrong host, 3 safety gate failed.
""".strip()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    return int(handler(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hc-bench",
        description=(
            "Record and compare Home Cortex model benchmarks. "
            "Suites reuse the existing fact, planner, mutation, and latency runners."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EXAMPLES,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    listed = sub.add_parser("list", help="Show registered benchmark suites")
    listed.set_defaults(handler=_list)

    run = sub.add_parser(
        "run",
        help="Run one suite and write an immutable result directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Semantic suites score once. --warmup and --repetitions apply to the "
            "latency suite (default warmup 1, repetitions 3). Other suites record "
            "the request but still take one scoring pass.\n\n"
            "Latency policy: warmup samples are excluded from percentiles. The "
            "harness does not unload the model. --verified-cold only labels the "
            "first request; unload the model yourself before using it.\n\n"
            "--cache-state is recorded and is not acted on. The harness does not "
            "flush caches."
        ),
    )
    run.add_argument("--suite", required=True, help="Suite name from `hc-bench list`")
    run.add_argument("--model", required=True, help="Ollama model name, including tag")
    run.add_argument("--label", help="Human label stored with the run. Not a run id")
    run.add_argument(
        "--ollama-url",
        default=None,
        help=(
            "Ollama base URL. The default is the project OLLAMA_URL "
            f"({DEFAULT_OLLAMA_URL}). If that name is not reachable from this "
            "host, hc-bench uses localhost or the Compose container address. "
            "An explicit value is never replaced."
        ),
    )
    run.add_argument("--results-dir", type=Path, help="Directory that holds run ids")
    run.add_argument("--data-dir", type=Path, help="Household JSON graph used by fact and planner suites")
    run.add_argument("--schema-dir", type=Path, help="Edge schema directory")
    run.add_argument("--cache-state", choices=CACHE_STATES, default="unknown")
    run.add_argument("--warmup", type=int, help="Latency suite only. Default 1")
    run.add_argument("--repetitions", type=int, help="Latency suite only. Default 3")
    run.add_argument("--num-ctx", type=int, help="Override planner context length for this process")
    run.add_argument(
        "--limit",
        type=int,
        help="Smoke cap on cases. Changes the corpus fingerprint; do not promote a limited run",
    )
    run.add_argument(
        "--verified-cold",
        action="store_true",
        help="Label the first latency request as a verified cold start. Does not unload the model",
    )
    run.add_argument(
        "--allow-nonstandard-host",
        action="store_true",
        help="Run on a machine other than the production GPU host and record that fact",
    )
    run.set_defaults(handler=_run)

    show = sub.add_parser("show", help="Print a stored run")
    show.add_argument("run_id", help="Run id, baseline name, or result directory")
    show.add_argument("--failures", action="store_true", help="List cases that did not pass")
    show.add_argument("--results-dir", type=Path)
    show.set_defaults(handler=_show)

    compare = sub.add_parser(
        "compare",
        help="Compare two stored runs without combining them into one score",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Warnings are printed when corpus, prompt, config, commit, hardware, "
            "or cache state differ. Warnings do not by themselves fail the command. "
            "Safety regressions and absolute preview/commit gates exit 3."
        ),
    )
    compare.add_argument("baseline", help="Baseline run id, baseline name, or directory")
    compare.add_argument("candidate", help="Candidate run id, baseline name, or directory")
    compare.add_argument("--results-dir", type=Path)
    compare.set_defaults(handler=_compare)

    baseline = sub.add_parser("baseline", help="Name a stored run. Promotion is explicit")
    baseline_sub = baseline.add_subparsers(dest="baseline_command", required=True)
    set_baseline = baseline_sub.add_parser("set", help="Point a name at a run id")
    set_baseline.add_argument("name")
    set_baseline.add_argument("run_id")
    set_baseline.add_argument("--results-dir", type=Path)
    set_baseline.set_defaults(handler=_baseline_set)
    list_baselines = baseline_sub.add_parser("list", help="Show named baselines")
    list_baselines.add_argument("--results-dir", type=Path)
    list_baselines.set_defaults(handler=_baseline_list)

    matrix = sub.add_parser(
        "matrix",
        help="Run the models and context lengths in a YAML file, one after another",
    )
    matrix.add_argument("path", type=Path, help="Matrix YAML. See benchmarks/matrix.example.yaml")
    matrix.add_argument("--ollama-url", default=None)
    matrix.add_argument("--results-dir", type=Path)
    matrix.add_argument("--data-dir", type=Path)
    matrix.add_argument("--schema-dir", type=Path)
    matrix.add_argument("--cache-state", choices=CACHE_STATES, default="unknown")
    matrix.add_argument("--warmup", type=int)
    matrix.add_argument("--repetitions", type=int)
    matrix.add_argument("--limit", type=int)
    matrix.add_argument("--verified-cold", action="store_true")
    matrix.add_argument("--allow-nonstandard-host", action="store_true")
    matrix.set_defaults(handler=_matrix)
    return parser


def _list(_args: argparse.Namespace) -> int:
    suites = registry().suites()
    if not suites:
        print("No benchmark suites are registered.", file=sys.stderr)
        return 2
    width = max(len(suite.name) for suite in suites)
    for suite in suites:
        print(f"{suite.name:<{width}}  {suite.description}")
    return 0


def _run(args: argparse.Namespace) -> int:
    if args.warmup is not None and args.warmup < 0:
        print("--warmup must be >= 0", file=sys.stderr)
        return 2
    if args.repetitions is not None and args.repetitions < 1:
        print("--repetitions must be >= 1", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 1:
        print("--limit must be >= 1", file=sys.stderr)
        return 2
    if args.num_ctx is not None and args.num_ctx < 1:
        print("--num-ctx must be >= 1", file=sys.stderr)
        return 2
    return execute(_request_from_args(args, suite=args.suite, model=args.model, label=args.label))


def _show(args: argparse.Namespace) -> int:
    try:
        directory = resolve_run(results_root(args.results_dir), args.run_id)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 2
    run, summary = load_run(directory)
    print(format_run_report(run, summary), end="")
    if args.failures:
        print(format_failures(load_cases(directory)))
    return 0


def _compare(args: argparse.Namespace) -> int:
    root = results_root(args.results_dir)
    try:
        baseline = resolve_run(root, args.baseline)
        candidate = resolve_run(root, args.candidate)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 2
    baseline_run, baseline_summary = load_run(baseline)
    candidate_run, candidate_summary = load_run(candidate)
    comparison = build_comparison(
        baseline_run, baseline_summary, candidate_run, candidate_summary
    )
    print(format_comparison(comparison, baseline_run, candidate_run), end="")
    return comparison.exit_code


def _baseline_set(args: argparse.Namespace) -> int:
    root = results_root(args.results_dir)
    try:
        directory = resolve_run(root, args.run_id)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 2
    run_id = load_run(directory)[0]["run_id"]
    write_baseline(root, args.name, str(run_id))
    print(f"Baseline {args.name} now points at {run_id}")
    return 0


def _baseline_list(args: argparse.Namespace) -> int:
    baselines = load_baselines(results_root(args.results_dir))
    if not baselines:
        print("No named baselines.")
        return 0
    for name, run_id in sorted(baselines.items()):
        print(f"{name}  {run_id}")
    return 0


def _matrix(args: argparse.Namespace) -> int:
    try:
        spec = yaml.safe_load(args.path.read_text(encoding="utf-8"))
        requests = plan_matrix(spec, _request_from_args(args, suite="", model="", label=None))
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Could not read matrix: {error}", file=sys.stderr)
        return 2
    worst = 0
    for request in requests:
        print(f"Matrix run: suite={request.suite} model={request.model} num_ctx={request.num_ctx}")
        code = execute(request)
        if code == 2:
            return 2
        worst = max(worst, code)
    return worst


def _request_from_args(args: argparse.Namespace, *, suite: str, model: str, label: str | None) -> RunRequest:
    return RunRequest(
        suite=suite,
        model=model,
        ollama_url=args.ollama_url,
        ollama_url_explicit=args.ollama_url is not None,
        label=label,
        results_dir=args.results_dir,
        data_dir=getattr(args, "data_dir", None),
        schema_dir=getattr(args, "schema_dir", None),
        cache_state=args.cache_state,
        warmup=args.warmup,
        repetitions=args.repetitions,
        num_ctx=getattr(args, "num_ctx", None),
        limit=args.limit,
        verified_cold=args.verified_cold,
        allow_nonstandard_host=args.allow_nonstandard_host,
    )
