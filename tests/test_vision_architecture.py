"""Import and dependency guards for the edge/backend vision boundary."""
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
        "home_cortex.db",
        "home_cortex.retrieval",
        "home_cortex.writing",
        "home_cortex.vision.edge",
    }
    assert forbidden.isdisjoint(loaded)


def test_edge_package_import_does_not_eagerly_load_opencv() -> None:
    loaded = _loaded_after_import("home_cortex.vision.edge")
    assert "cv2" not in loaded
    assert "ultralytics" not in loaded


def test_semantic_api_import_does_not_load_edge_runtime() -> None:
    loaded = _loaded_after_import("home_cortex.api")
    assert not any(name.startswith("home_cortex.vision.edge") for name in loaded)
    assert "cv2" not in loaded
    assert "ultralytics" not in loaded


def test_edge_camera_dependency_remains_optional() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]
    vision = project["project"]["optional-dependencies"]["vision"]

    assert not any(name.startswith("opencv-python") for name in dependencies)
    assert not any(name.startswith("ultralytics") for name in dependencies)
    assert any(name.startswith("opencv-python") for name in vision)
