"""Regression guards for the consolidated backend boundaries."""

import ast
from graphlib import CycleError, TopologicalSorter
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


def _production_import_graph() -> dict[str, set[str]]:
    modules: dict[str, Path] = {}
    for path in PACKAGE.rglob("*.py"):
        parts = list(path.relative_to(PACKAGE).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        name = "home_cortex" + (f".{'.'.join(parts)}" if parts else "")
        modules[name] = path

    graph = {name: set() for name in modules}
    for name, path in modules.items():
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package_parts = package.split(".")
                    prefix = ".".join(
                        package_parts[:len(package_parts) - node.level + 1]
                    )
                    base = ".".join(filter(None, (prefix, node.module or "")))
                else:
                    base = node.module or ""
                candidates.append(base)
                candidates.extend(
                    ".".join(filter(None, (base, alias.name)))
                    for alias in node.names
                )
            for candidate in candidates:
                while candidate:
                    if candidate in modules and candidate != name:
                        graph[name].add(candidate)
                        break
                    candidate = candidate.rpartition(".")[0]
    return graph


def test_chat_api_startup_excludes_optional_and_maintenance_subsystems() -> None:
    loaded = _loaded_after_import("home_cortex.api")
    forbidden = {
        "home_cortex.persistence.ingestion",
        "home_cortex.persistence.export",
        "home_cortex.spatial.localization.solver",
        "home_cortex.spatial.localization.observation",
    }
    assert forbidden.isdisjoint(loaded)
    assert not any(name.startswith("home_cortex.vision") for name in loaded)


def test_production_import_graph_is_acyclic() -> None:
    try:
        tuple(TopologicalSorter(_production_import_graph()).static_order())
    except CycleError as error:
        raise AssertionError(f"production import cycle: {error.args[1]}") from error


def test_root_namespace_stays_package_oriented() -> None:
    root_modules = {path.name for path in PACKAGE.glob("*.py")}
    assert root_modules == {"__init__.py", "config.py"}


def test_provider_contract_import_does_not_choose_or_load_an_adapter() -> None:
    loaded = _loaded_after_import("home_cortex.providers.base")
    assert "home_cortex.providers.ollama" not in loaded
    assert "home_cortex.providers.openrouter" not in loaded


def test_ollama_adapter_does_not_import_openrouter_adapter() -> None:
    loaded = _loaded_after_import("home_cortex.providers.ollama")
    assert "home_cortex.providers.openrouter" not in loaded


def test_openrouter_adapter_does_not_import_ollama_adapter() -> None:
    loaded = _loaded_after_import("home_cortex.providers.openrouter")
    assert "home_cortex.providers.ollama" not in loaded


def test_provider_adapters_implement_the_shared_surface() -> None:
    from home_cortex.providers.ollama import OllamaService
    from home_cortex.providers.openrouter import OpenRouterService

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
    tree = ast.parse((PACKAGE / "runtime" / "agent.py").read_text())
    private_dispatches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "_dispatch"
    ]
    assert private_dispatches == []


def test_test_frameworks_are_dev_dependencies_only() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    production = {item.split("<", 1)[0].split(">", 1)[0] for item in project["dependencies"]}
    assert {"pytest", "pytest-asyncio", "pytest-cov"}.isdisjoint(production)


def test_api_package_initializer_is_only_the_public_entrypoint_surface() -> None:
    assert not (PACKAGE / "api.py").exists()
    api_lines = (PACKAGE / "api" / "__init__.py").read_text().splitlines()
    assert len(api_lines) < 100
