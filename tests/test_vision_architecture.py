"""Import and dependency guards for the backend/client Vision boundary."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _loaded_after_import(module: str) -> set[str]:
    script = (
        "import importlib,json,sys;"
        f"importlib.import_module({module!r});"
        "print(json.dumps(sorted(sys.modules)))"
    )
    return set(json.loads(subprocess.check_output(
        [sys.executable, "-c", script], text=True, cwd=ROOT,
    )))


@pytest.mark.parametrize(
    "module",
    (
        "home_cortex.vision",
        "home_cortex.vision.contracts",
        "home_cortex.vision.ports",
    ),
)
def test_vision_domain_import_does_not_load_edge_or_household_adapters(module) -> None:
    loaded = _loaded_after_import(module)
    forbidden = {
        "cv2",
        "ultralytics",
        "surrealdb",
        "home_cortex.persistence.db",
        "home_cortex.persistence.retrieval",
        "home_cortex.mutation.writing",
        "home_cortex_client",
    }
    assert forbidden.isdisjoint(loaded)


def test_chat_api_import_does_not_load_vision() -> None:
    loaded = _loaded_after_import("home_cortex.api")
    assert not any(name.startswith("home_cortex.vision") for name in loaded)
    assert "cv2" not in loaded
    assert "ultralytics" not in loaded


def test_backend_has_no_device_runtime_dependencies() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    groups = [project["project"]["dependencies"]]
    groups.extend(project["project"].get("optional-dependencies", {}).values())
    dependencies = [item.lower() for group in groups for item in group]
    assert not any(
        item.startswith(("opencv-python", "ultralytics", "cryptography"))
        for item in dependencies
    )


def test_backend_has_no_device_runtime_modules() -> None:
    vision = ROOT / "src" / "home_cortex" / "vision"
    assert not (vision / "edge").exists()
    assert not (vision / "camera").exists()
    assert not (vision / "relay.py").exists()
    assert not (vision / "web").exists()


def test_backend_does_not_import_client_package() -> None:
    imports: list[tuple[Path, str]] = []
    for path in (ROOT / "src" / "home_cortex").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend((path, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append((path, node.module))
    assert not [
        (path, name)
        for path, name in imports
        if name == "home_cortex_client" or name.startswith("home_cortex_client.")
    ]


def test_gui_does_not_depend_on_client_package() -> None:
    gui = ROOT / "src" / "home_gui"
    checked = [gui / "package.json", *(gui / "src").rglob("*")]
    assert not any(
        "home_cortex_client" in path.read_text(errors="ignore")
        for path in checked
        if path.is_file()
    )
