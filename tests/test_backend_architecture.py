"""Regression guards for the consolidated backend boundaries."""

import ast
import json
from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "home_cortex"


def _loaded_after_import(module: str) -> set[str]:
    script = (
        "import importlib,json,sys;"
        f"importlib.import_module({module!r});"
        "print(json.dumps(sorted(sys.modules)))"
    )
    return set(json.loads(subprocess.check_output(
        [sys.executable, "-c", script], text=True, cwd=ROOT,
    )))


def test_chat_api_startup_excludes_optional_and_maintenance_subsystems() -> None:
    loaded = _loaded_after_import("home_cortex.api")
    forbidden = {
        "home_cortex.ingestion",
        "home_cortex.export",
        "home_cortex.spatial.localize",
        "home_cortex.spatial.observation",
    }
    assert forbidden.isdisjoint(loaded)
    assert not any(name.startswith("home_cortex.vision") for name in loaded)


def test_provider_contract_import_does_not_choose_or_load_an_adapter() -> None:
    loaded = _loaded_after_import("home_cortex.model_provider")
    assert "home_cortex.ollama" not in loaded
    assert "home_cortex.openrouter" not in loaded


def test_ollama_adapter_does_not_import_openrouter_adapter() -> None:
    loaded = _loaded_after_import("home_cortex.ollama")
    assert "home_cortex.openrouter" not in loaded


def test_openrouter_adapter_does_not_import_ollama_adapter() -> None:
    loaded = _loaded_after_import("home_cortex.openrouter")
    assert "home_cortex.ollama" not in loaded


def test_provider_adapters_implement_the_shared_surface() -> None:
    from home_cortex.ollama import OllamaService
    from home_cortex.openrouter import OpenRouterService

    required = {
        "chat",
        "stream_chat",
        "chat_with_tools",
        "stream_chat_with_tools",
        "plan_item_mutation",
        "plan_semantic_fact",
        "plan_unified_semantic",
        "close",
    }
    for adapter in (OllamaService, OpenRouterService):
        assert not (required - set(dir(adapter)))


def test_agent_service_does_not_call_model_loop_private_dispatch() -> None:
    tree = ast.parse((PACKAGE / "agent_service.py").read_text())
    private_dispatches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "_dispatch"
    ]
    assert private_dispatches == []


def test_test_frameworks_are_dev_dependencies_only() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    production = {item.split("<", 1)[0].split(">", 1)[0] for item in project["dependencies"]}
    assert {"pytest", "pytest-asyncio", "pytest-cov"}.isdisjoint(production)


def test_legacy_api_module_is_only_a_compatibility_surface() -> None:
    api_lines = (PACKAGE / "api.py").read_text().splitlines()
    assert len(api_lines) < 100
