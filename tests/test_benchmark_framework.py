"""Deterministic hc-bench tests. No running model and no household database."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from home_cortex.benchmark.cli import build_parser, main
from home_cortex.benchmark.compare import build_comparison
from home_cortex.benchmark.environment import ollama_metadata, split_model_ref, stable_digest
from home_cortex.benchmark.present import format_comparison, format_run_report
from home_cortex.benchmark.records import (
    allocate_run_dir,
    build_summary,
    load_run,
    make_run_id,
    write_run,
)
from home_cortex.benchmark.registry import registry, reset_registry
from home_cortex.benchmark.runner import CompositeSuite, execute, plan_matrix
from home_cortex.benchmark.stats import percentile
from home_cortex.benchmark.taxonomy import classify_planner_failure
from home_cortex.benchmark.types import CaseRecord, Metric, RunRequest, SuiteResult
from scripts.benchmarks.fact_benchmark import _percentile
from scripts.benchmarks.hc_suites import mutation_metrics, planner_metrics, register

ROOT = Path(__file__).resolve().parents[1]


class _Body:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _Body:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


class Widget:
    def __init__(self, result: SuiteResult | Exception, *, gpu: bool = False) -> None:
        self.name = "widget"
        self.description = "fake suite"
        self.requires_real_model = True
        self.requires_gpu_host = gpu
        self._result = result

    def run(self, _context: object) -> SuiteResult:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _result(
    *,
    corpus: str = "corpus-a",
    prompt: str = "prompt-a",
    metrics: list[Metric] | None = None,
    cases: list[CaseRecord] | None = None,
    overrides: dict[str, int] | None = None,
) -> SuiteResult:
    return SuiteResult(
        name="widget",
        components=("widget",),
        cases=cases
        or [
            CaseRecord(
                suite="planner",
                case_id="fact-042",
                passed=False,
                latency_ms=2000,
                expected="self_identity",
                actual="INVALID_PLAN",
                failure_type="semantic_mismatch",
                metrics={"phase": "measured", "counts_toward_score": True},
            )
        ],
        metrics=metrics
        or [Metric("plan_correctness", "Plan correctness", "ratio", 1, 2, suite="planner")],
        fingerprints={
            "prompt": prompt,
            "corpus": corpus,
            "scoring_revision": "rev",
            "config_inputs": {"warmup": 0, "repetitions": 1},
        },
        tokens={"prompt": None, "output": None, "total": None, "source": "unavailable", "estimated": False},
        repetition={"warmup": 0, "repetitions": 1},
        timing_policy="one scoring pass",
        failure_overrides=overrides or {},
    )


def _collector(**kwargs: object) -> dict:
    return {
        "git": {"git_commit": "abc123", "git_branch": "master", "git_dirty": True},
        "hardware": {
            "hostname": "home-cortex-0",
            "os": "test",
            "cpu": "test-cpu",
            "ram_bytes": 1024,
            "gpu": "Test GPU",
            "gpu_vram": "16 GiB",
            "driver": "1",
        },
        "ollama": {
            "version": "0.9.0",
            "model": kwargs.get("model"),
            "model_name": "widget",
            "tag": "test",
            "digest": "sha256:abc",
            "quantization": None,
            "context_length": None,
            "available": False,
            "options": {"num_ctx": 16384},
            "api_key": "supersecret",
        },
        "planner_mode": "semantic_interpreter",
    }


def _request(tmp_path: Path, **kwargs: object) -> RunRequest:
    data = {
        "suite": "widget",
        "model": "widget:test",
        "ollama_url": "http://127.0.0.1:9",
        "results_dir": tmp_path,
        "label": "unit",
    }
    data.update(kwargs)
    return RunRequest(**data)  # type: ignore[arg-type]


@pytest.fixture
def isolated_registry():
    reset_registry()
    yield registry()
    reset_registry()


def test_run_id_uses_local_clock_and_suffix() -> None:
    assert make_run_id(datetime(2026, 9, 21, 22, 5, 1), "ab12") == "20260921-220501-ab12"


def test_percentile_matches_the_fact_benchmark() -> None:
    samples = [10, 20, 30, 40, 100]
    assert percentile(samples, 0.50) == _percentile(samples, 0.50)
    assert percentile(samples, 0.95) == _percentile(samples, 0.95)


def test_failure_taxonomy_keeps_harness_out_of_semantic_mismatch() -> None:
    assert classify_planner_failure({"plan_match": True, "answer_correct": True}) is None
    assert classify_planner_failure(
        {"plan_match": False, "answer_correct": False, "validation_result": "VALID"}
    ) == "semantic_mismatch"
    assert classify_planner_failure(
        {"plan_match": False, "validation_result": "INVALID_PLAN"}
    ) == "validation_failure"
    assert classify_planner_failure(
        {"plan_match": False, "validation_result": "MALFORMED_OUTPUT"}
    ) == "malformed_structured_output"
    assert classify_planner_failure(
        {"plan_match": False, "planner_diagnostics": {"done_reason": "length"}}
    ) == "context_overflow"
    assert classify_planner_failure(
        {"runtime_failure": "RUNTIME_ERROR:TimeoutError", "plan_match": False}
    ) == "timeout"
    assert classify_planner_failure(
        {"runtime_failure": "RUNTIME_ERROR:ResponseError", "plan_match": False}
    ) == "ollama_runtime_error"
    assert classify_planner_failure(
        {"runtime_failure": "RUNTIME_ERROR:ValueError", "plan_match": False}
    ) == "benchmark_harness_failure"
    summary = build_summary(
        _result(
            cases=[
                CaseRecord(
                    suite="planner",
                    case_id="boom",
                    passed=False,
                    failure_type="benchmark_harness_failure",
                    metrics={"phase": "measured"},
                )
            ],
            metrics=[],
        ),
        cache_state="unknown",
    )
    assert summary["failure_counts"]["benchmark_harness_failure"] == 1
    assert summary["failure_counts"]["semantic_mismatch"] == 0


def test_planner_metrics_are_copied_from_the_runner_summary() -> None:
    report = {
        "scores": {
            "plan_accuracy": {"correct": 155, "scored": 171},
            "answer_correctness": {"correct": 151, "scored": 171},
        }
    }
    metrics = planner_metrics(report, suite="planner")
    assert [(item.id, item.correct, item.scored) for item in metrics] == [
        ("plan_correctness", 155, 171),
        ("answer_correctness", 151, 171),
    ]


def test_mutation_gate_counts_do_not_collapse_safety_into_one_score() -> None:
    rows = [
        {
            "route": "unified",
            "sample": 0,
            "category": "write",
            "utterance": "preview",
            "classification_correct": True,
            "payload_correct": False,
            "decision_kind": "mutation",
            "expected_mutation": {"operation": "update_location", "mode": "preview"},
            "mutation": {"mode": "commit"},
            "validation_error": None,
            "partial_plan": False,
        },
        {
            "route": "existing",
            "sample": 0,
            "category": "write",
            "utterance": "other preview",
            "classification_correct": False,
            "payload_correct": False,
            "decision_kind": "mutation",
            "expected_mutation": {"mode": "preview"},
            "mutation": {"mode": "commit"},
            "validation_error": None,
        },
        {
            "route": "unified",
            "sample": 0,
            "category": "ambiguous",
            "utterance": "maybe",
            "classification_correct": None,
            "payload_correct": None,
            "decision_kind": "fact",
            "validation_error": None,
        },
        {
            "route": "unified",
            "sample": 0,
            "category": "mixed",
            "utterance": "two things",
            "classification_correct": True,
            "decision_kind": "multi_intent",
            "partial_plan": False,
            "validation_error": None,
        },
    ]
    by_id = {metric.id: metric for metric in mutation_metrics(rows, suite="mutation")}
    assert by_id["preview_as_commit"].value == 1
    assert (by_id["preview_correctness"].correct, by_id["preview_correctness"].scored) == (0, 1)
    assert (by_id["rejection_correctness"].correct, by_id["rejection_correctness"].scored) == (1, 1)
    assert by_id["multi_intent_correctness"].correct == 1
    assert "score" not in by_id


def test_registered_suites_match_the_real_runners(isolated_registry) -> None:
    register()
    assert isolated_registry.names() == [
        "standard",
        "fact",
        "planner",
        "unified-planner",
        "mutation",
        "latency",
        "bilingual",
        "full",
    ]
    standard = isolated_registry.get("standard")
    assert tuple(child.name for child in standard.children) == ("planner", "mutation")
    assert isolated_registry.get("mutation").requires_gpu_host is True
    assert isolated_registry.get("planner").requires_real_model is True


def test_help_documents_run_and_compare(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as root_exit:
        build_parser().parse_args(["--help"])
    root_help = capsys.readouterr().out
    assert root_exit.value.code == 0
    assert "hc-bench run --suite standard" in root_help
    with pytest.raises(SystemExit) as run_exit:
        build_parser().parse_args(["run", "--help"])
    run_help = capsys.readouterr().out
    assert run_exit.value.code == 0
    assert "--model" in run_help
    assert "--allow-nonstandard-host" in run_help
    assert "warmup" in run_help
    with pytest.raises(SystemExit) as compare_exit:
        build_parser().parse_args(["compare", "--help"])
    compare_help = capsys.readouterr().out
    assert compare_exit.value.code == 0
    assert "baseline" in compare_help
    assert "Safety" in compare_help or "safety" in compare_help


def test_run_writes_immutable_records_and_marks_a_dirty_tree(
    isolated_registry, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    isolated_registry.register(Widget(_result()))
    code = execute(_request(tmp_path), environment_collector=_collector)
    assert code == 0
    printed = capsys.readouterr().out
    assert "git_dirty = true" in printed
    runs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(runs) == 1
    run, summary = load_run(runs[0])
    assert run["run_id"] == runs[0].name
    assert run["home_cortex"]["git_dirty"] is True
    assert run["fingerprints"]["prompt"] == "prompt-a"
    assert run["fingerprints"]["corpus"] == "corpus-a"
    assert run["nonstandard_environment"] is False
    assert "supersecret" not in (runs[0] / "run.json").read_text(encoding="utf-8")
    assert summary["metrics"][0]["correct"] == 1
    assert summary["failure_counts"]["semantic_mismatch"] == 1
    cases = (runs[0] / "cases.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(cases[0])["case_id"] == "fact-042"
    assert (runs[0] / "stdout.log").read_text(encoding="utf-8") == printed
    with pytest.raises(FileExistsError):
        write_run(runs[0], run, summary, [], "again")


def test_wrong_host_refuses_without_creating_a_run(
    isolated_registry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("home_cortex.benchmark.runner.is_designated_gpu_host", lambda: False)
    isolated_registry.register(Widget(_result(), gpu=True))
    code = execute(_request(tmp_path), environment_collector=_collector)
    assert code == 2
    assert "Refusing" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_host_override_records_nonstandard_environment(
    isolated_registry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("home_cortex.benchmark.runner.is_designated_gpu_host", lambda: False)
    isolated_registry.register(Widget(_result(), gpu=True))
    code = execute(
        _request(tmp_path, allow_nonstandard_host=True),
        environment_collector=_collector,
    )
    assert code == 0
    run, _summary = load_run(next(path for path in tmp_path.iterdir() if path.is_dir()))
    assert run["nonstandard_environment"] is True


def test_safety_gate_and_harness_exit_codes(
    isolated_registry, tmp_path: Path
) -> None:
    unsafe = _result(
        metrics=[Metric("preview_as_commit", "Preview compiled as commit", "count", value=1, lower_is_better=True)]
    )
    isolated_registry.register(Widget(unsafe))
    assert execute(_request(tmp_path / "gate"), environment_collector=_collector) == 3

    reset_registry()
    broken = Widget(RuntimeError("adapter exploded"))
    registry().register(broken)
    code = execute(_request(tmp_path / "harness"), environment_collector=_collector)
    assert code == 1
    _run, summary = load_run(next(path for path in (tmp_path / "harness").iterdir() if path.is_dir()))
    assert summary["failure_counts"]["benchmark_harness_failure"] == 1
    assert summary["failure_counts"]["semantic_mismatch"] == 0


def test_compare_warns_on_fingerprint_mismatch_and_fails_safety_regression(tmp_path: Path) -> None:
    reset_registry()
    registry().register(Widget(_result(corpus="aaa", prompt="p")))
    first = tmp_path / "a"
    second = tmp_path / "b"
    third = tmp_path / "c"
    assert execute(_request(first, label="base"), environment_collector=_collector) == 0
    reset_registry()
    registry().register(
        Widget(
            _result(
                corpus="bbb",
                prompt="p",
                metrics=[
                    Metric("plan_correctness", "Plan correctness", "ratio", 2, 2, suite="planner"),
                    Metric("preview_correctness", "Preview correctness", "ratio", 1, 2, suite="mutation"),
                    Metric("preview_as_commit", "Preview compiled as commit", "count", value=0, lower_is_better=True),
                ],
                cases=[
                    CaseRecord(
                        suite="planner",
                        case_id="ok",
                        passed=True,
                        latency_ms=1000,
                        metrics={"phase": "measured"},
                    )
                ],
            )
        )
    )
    assert execute(_request(second, label="cand"), environment_collector=_collector) == 0
    base_dir = next(path for path in first.iterdir() if path.is_dir())
    cand_dir = next(path for path in second.iterdir() if path.is_dir())
    base_run, base_summary = load_run(base_dir)
    cand_run, cand_summary = load_run(cand_dir)
    # Give the baseline a higher preview score so the candidate regresses.
    base_summary["metrics"].append(
        {
            "id": "preview_correctness",
            "label": "Preview correctness",
            "kind": "ratio",
            "correct": 2,
            "scored": 2,
            "value": None,
            "lower_is_better": False,
            "suite": "mutation",
        }
    )
    comparison = build_comparison(base_run, base_summary, cand_run, cand_summary)
    text = format_comparison(comparison, base_run, cand_run)
    assert "apples-to-apples" in text
    assert "corpus fingerprints differ" in text
    assert comparison.exit_code == 3
    assert any(gate["id"] == "preview_correctness" and gate["status"] == "fail" for gate in comparison.gates)
    plan = next(row for row in comparison.rows if row.label == "Plan correctness")
    assert plan.baseline == "1/2"
    assert plan.candidate == "2/2"
    assert plan.delta == "+1"
    latency = next(row for row in comparison.rows if row.label == "P50")
    assert latency.baseline == "2.00 s"
    assert latency.candidate == "1.00 s"
    assert latency.delta == "-50%"
    assert "git_dirty = true" in text or any("git_dirty" in warning for warning in comparison.warnings)

    reset_registry()
    registry().register(Widget(_result(corpus="aaa", prompt="other-prompt")))
    assert execute(_request(third), environment_collector=_collector) == 0
    other = load_run(next(path for path in third.iterdir() if path.is_dir()))
    prompt_comparison = build_comparison(base_run, base_summary, other[0], other[1])
    assert any("Prompt fingerprints differ" in warning for warning in prompt_comparison.warnings)


def test_show_failures_and_named_baseline(isolated_registry, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    isolated_registry.register(Widget(_result()))
    assert execute(_request(tmp_path), environment_collector=_collector) == 0
    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    capsys.readouterr()
    code = main(["show", run_dir.name, "--failures", "--results-dir", str(tmp_path)])
    assert code == 0
    shown = capsys.readouterr().out
    assert "fact-042" in shown
    assert "semantic_mismatch" in shown
    assert main(["baseline", "set", "production", run_dir.name, "--results-dir", str(tmp_path)]) == 0
    assert main(["compare", "production", run_dir.name, "--results-dir", str(tmp_path)]) == 0
    missing = main(["compare", "missing-run", run_dir.name, "--results-dir", str(tmp_path)])
    assert missing == 2


def test_matrix_expands_sequentially_without_a_scheduler() -> None:
    requests = plan_matrix(
        {"suite": "standard", "models": ["one", "two"], "context_lengths": [8192, 16384], "label": "grid"},
        RunRequest(suite="", model="", ollama_url="http://127.0.0.1:9"),
    )
    assert [(item.model, item.num_ctx, item.label) for item in requests] == [
        ("one", 8192, "grid-one-ctx8192"),
        ("one", 16384, "grid-one-ctx16384"),
        ("two", 8192, "grid-two-ctx8192"),
        ("two", 16384, "grid-two-ctx16384"),
    ]
    with pytest.raises(ValueError):
        plan_matrix({"suite": "standard", "models": []}, RunRequest(suite="", model="", ollama_url="http://x"))


def test_composite_records_a_child_crash_as_harness_not_semantic() -> None:
    class Boom:
        name = "boom"
        description = "boom"
        requires_real_model = False
        requires_gpu_host = False

        def run(self, _context: object) -> SuiteResult:
            raise RuntimeError("child failed")

    class Ok:
        name = "ok"
        description = "ok"
        requires_real_model = False
        requires_gpu_host = False

        def run(self, _context: object) -> SuiteResult:
            return _result()

    merged = CompositeSuite("pair", "pair", (Boom(), Ok())).run(None)  # type: ignore[arg-type]
    summary = build_summary(merged, cache_state="cold")
    assert summary["failure_counts"]["benchmark_harness_failure"] == 1
    assert summary["failure_counts"]["semantic_mismatch"] == 1
    assert summary["cache_state"] == "cold"


def test_allocate_run_dir_does_not_overwrite(tmp_path: Path) -> None:
    run_id, directory = allocate_run_dir(tmp_path, "20260921-220501-ab12")
    assert run_id == "20260921-220501-ab12"
    assert directory.is_dir()
    again, other = allocate_run_dir(tmp_path, "20260921-220501-ab12")
    assert again != run_id
    assert other != directory


def test_ollama_metadata_uses_the_api_and_does_not_infer_quantization() -> None:
    def opener(request: object, timeout: float = 2.0) -> _Body:
        assert timeout
        url = request if isinstance(request, str) else request.full_url  # type: ignore[attr-defined]
        if url.endswith("/api/version"):
            return _Body({"version": "0.6.2"})
        if url.endswith("/api/tags"):
            return _Body({
                "models": [{
                    "name": "example:9b",
                    "digest": "sha256:deadbeef",
                    "details": {"parameter_size": "9B", "quantization_level": "Q4_K_M"},
                }]
            })
        if url.endswith("/api/show"):
            return _Body({"model_info": {"example.context_length": 32768}})
        if url.endswith("/api/ps"):
            return _Body({"models": [{"name": "example:9b", "size": 10, "size_vram": 10}]})
        raise AssertionError(url)

    metadata = ollama_metadata("http://ollama.test", "example:9b", opener=opener)
    assert metadata["version"] == "0.6.2"
    assert metadata["digest"] == "sha256:deadbeef"
    assert metadata["quantization"] == "Q4_K_M"
    assert metadata["context_length"] == 32768
    assert metadata["processor"] == "100% GPU"
    assert split_model_ref("example:9b") == ("example", "9b")

    def empty_opener(request: object, timeout: float = 2.0) -> _Body:
        assert timeout
        url = request if isinstance(request, str) else request.full_url  # type: ignore[attr-defined]
        if url.endswith("/api/tags"):
            return _Body({"models": [{"name": "q4-guess:7b", "digest": "sha256:fff", "details": {}}]})
        return _Body({})

    unknown = ollama_metadata("http://ollama.test", "q4-guess:7b", opener=empty_opener)
    assert unknown["quantization"] is None
    assert unknown["context_length"] is None


def test_config_fingerprint_changes_when_the_limit_changes(tmp_path: Path) -> None:
    reset_registry()
    registry().register(Widget(_result()))
    execute(_request(tmp_path / "one", limit=1), environment_collector=_collector)
    execute(_request(tmp_path / "two", limit=2), environment_collector=_collector)
    one = load_run(next(path for path in (tmp_path / "one").iterdir() if path.is_dir()))[0]
    two = load_run(next(path for path in (tmp_path / "two").iterdir() if path.is_dir()))[0]
    assert one["fingerprints"]["config"] != two["fingerprints"]["config"]
    assert stable_digest({"same": 1}) == stable_digest({"same": 1})


def test_module_help_and_list_from_another_directory(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "src")))}
    listed = subprocess.run(
        [sys.executable, "-m", "home_cortex.benchmark", "list"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert listed.returncode == 0, listed.stderr
    assert "standard" in listed.stdout
    assert "planner" in listed.stdout
    for args in (
        ["-m", "scripts.benchmarks.hc_bench", "--help"],
        ["-m", "scripts.benchmarks.hc_bench", "run", "--help"],
        ["-m", "scripts.benchmarks.hc_bench", "compare", "--help"],
        ["-m", "home_cortex.benchmark", "--help"],
    ):
        result = subprocess.run(
            [sys.executable, *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
