"""Engineering commands are importable; the application never imports them."""
import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_has_no_engineering_dependencies():
    for path in (ROOT / "src/home_cortex").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([node.module or ""] if isinstance(node, ast.ImportFrom) else
                     [item.name for item in node.names] if isinstance(node, ast.Import) else [])
            assert not any(name == "scripts" or name.startswith("scripts.") for name in names), path


@pytest.mark.parametrize("module", [
    "benchmarks.fact_benchmark", "benchmarks.semantic_planner_benchmark",
    "benchmarks.tier1_latency_bench", "profiling.token_latency_audit",
    "profiling.token_component_probe", "profiling.http_latency_audit",
    "profiling.profile_semantic_transport", "probes.ollama_prefix_reuse_probe",
    "probes.ollama_warm_load_probe", "probes.kinship_context_probe",
    "probes.item_location_probe", "probes.bilingual_planner_probe",
    "probes.layer_failure_trace", "maintenance.freeze_contract_candidate",
    "maintenance.context_surface_audit", "maintenance.export_graph",
])
def test_command_help_from_another_directory(module, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "scripts." + module, "--help"], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "src")))},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
