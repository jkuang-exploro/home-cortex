"""Provenance for a benchmark run: git, hardware, and Ollama identity.

Values that the runtime does not report stay null or ``unavailable``. This module
does not guess quantization, context length, or token counts.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

DESIGNATED_GPU_HOSTS = frozenset({"home-cortex-0"})
CACHE_STATES = ("warm", "cold", "unchanged", "unknown")
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

Opener = Callable[..., Any]


def repo_root() -> Path:
    """Checkout that contains benchmarks and pyproject, else the working directory."""

    here = Path(__file__).resolve()
    candidates = list(here.parents)
    candidates.append(Path.cwd())
    for parent in candidates:
        if (parent / "benchmarks").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def designated_gpu_hosts() -> set[str]:
    """Production GPU host, plus any extra names in ``HC_BENCH_GPU_HOSTS``."""

    hosts = set(DESIGNATED_GPU_HOSTS)
    extra = os.environ.get("HC_BENCH_GPU_HOSTS", "")
    hosts.update(item.strip() for item in extra.split(",") if item.strip())
    return hosts


def hostname() -> str:
    return socket.gethostname()


def is_designated_gpu_host(name: str | None = None) -> bool:
    short = (name if name is not None else hostname()).split(".")[0]
    return short in designated_gpu_hosts()


def stable_digest(payload: Any) -> str:
    """SHA-256 of a JSON-stable payload. Used for prompt, corpus, and config."""

    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path) -> str:
    """Hash regular files under ``root``. Skips dotfiles, caches, and secret names."""

    if not root.is_dir():
        return "missing"
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and not _skip_hash_path(path)
    )
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def cached_tree_hash(root: Path, cache: dict[str, str]) -> str:
    key = str(root.resolve()) if root.exists() else str(root)
    if key not in cache:
        cache[key] = hash_tree(root)
    return cache[key]


def instruction_prompt_fingerprint() -> str:
    """Fingerprint interpreter instructions and examples, without the schema payload.

    Schema and ontology changes are part of the corpus fingerprint. A failed run
    that never builds a schema still records this instruction fingerprint.
    """

    from home_cortex.semantic import unified_planner as unified
    from home_cortex.semantic.prompt import (
        _PLANNER_INSTRUCTIONS,
        _semantic_planner_examples,
    )

    return stable_digest(
        {
            "instructions": _PLANNER_INSTRUCTIONS,
            "examples": _semantic_planner_examples(),
            "unified_rules": unified._UNIFIED_RULES,
            "unified_examples": unified._MULTI_INTENT_EXAMPLES,
        }
    )


def semantic_prompt_fingerprint(schema: Any) -> str:
    """Fingerprint the fact-planner system prompt actually built for this schema."""

    from home_cortex.semantic import unified_planner as unified
    from home_cortex.semantic.prompt import (
        _semantic_planner_examples,
        planner_system_prompt,
    )

    return stable_digest(
        {
            "fact_system": planner_system_prompt(schema.planner_capability_payload()),
            "examples": _semantic_planner_examples(),
            "unified_rules": unified._UNIFIED_RULES,
            "unified_examples": unified._MULTI_INTENT_EXAMPLES,
        }
    )


def git_metadata(root: Path | None = None) -> dict[str, Any]:
    checkout = root or repo_root()

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=checkout,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(run("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_commit": "unavailable",
            "git_branch": "unavailable",
            "git_dirty": None,
        }
    return {"git_commit": commit, "git_branch": branch, "git_dirty": dirty}


def hardware_metadata() -> dict[str, Any]:
    """Host facts needed to interpret latency. No user identity is collected."""

    gpu = _gpu_metadata()
    memory = _ram_bytes()
    return {
        "hostname": hostname(),
        "os": platform.platform(),
        "cpu": _cpu_name(),
        "ram_bytes": memory,
        "gpu": gpu["gpu"],
        "gpu_vram": gpu["gpu_vram"],
        "driver": gpu["driver"],
    }


def planner_options(num_ctx: int | None = None) -> dict[str, Any]:
    from home_cortex.providers.ollama import (
        PLANNER_KEEP_ALIVE,
        PLANNER_NUM_CTX,
        PLANNER_NUM_PREDICT,
        PLANNER_SEED,
    )

    return {
        "temperature": 0,
        "num_ctx": num_ctx if num_ctx is not None else PLANNER_NUM_CTX,
        "num_predict": PLANNER_NUM_PREDICT,
        "seed": PLANNER_SEED,
        "keep_alive": PLANNER_KEEP_ALIVE,
    }


def ollama_metadata(
    base_url: str,
    model: str,
    *,
    opener: Opener = urllib.request.urlopen,
    timeout: float = 2.0,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    """Read Ollama's own version, tag, and show payload. Do not infer missing fields."""

    model_name, tag = split_model_ref(model)
    base = base_url.rstrip("/")
    version_payload = _get_json(f"{base}/api/version", opener=opener, timeout=timeout)
    tags_payload = _get_json(f"{base}/api/tags", opener=opener, timeout=timeout)
    show_payload = _post_json(
        f"{base}/api/show",
        {"name": model},
        opener=opener,
        timeout=timeout,
    )
    ps_payload = _get_json(f"{base}/api/ps", opener=opener, timeout=timeout)
    listed = _matching_model(tags_payload, model)
    details = {}
    if isinstance(listed, Mapping):
        raw_details = listed.get("details")
        if isinstance(raw_details, Mapping):
            details = dict(raw_details)
    if isinstance(show_payload, Mapping) and isinstance(show_payload.get("details"), Mapping):
        for key, value in show_payload["details"].items():
            details.setdefault(key, value)
    digest = None
    if isinstance(listed, Mapping):
        digest = listed.get("digest") or listed.get("id")
    processor = _processor(ps_payload, model)
    quantization = details.get("quantization_level")
    return {
        "available": version_payload is not None or listed is not None,
        "version": (
            version_payload.get("version")
            if isinstance(version_payload, Mapping)
            else None
        ),
        "model": model,
        "model_name": model_name,
        "tag": tag,
        "digest": digest if isinstance(digest, str) and digest else None,
        "quantization": quantization if isinstance(quantization, str) and quantization else None,
        "parameter_size": details.get("parameter_size")
        if isinstance(details.get("parameter_size"), str)
        else None,
        "context_length": _context_length(show_payload),
        "processor": processor,
        "options": planner_options(num_ctx),
        "requested_num_ctx": (num_ctx if num_ctx is not None else planner_options()["num_ctx"]),
    }


def collect_environment(
    *,
    ollama_url: str,
    model: str,
    num_ctx: int | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Snapshot used by a real run. Network and GPU probes fail soft."""

    return {
        "git": git_metadata(root),
        "hardware": hardware_metadata(),
        "ollama": ollama_metadata(ollama_url, model, num_ctx=num_ctx),
        "planner_mode": "semantic_interpreter",
    }


def split_model_ref(model: str) -> tuple[str, str | None]:
    if ":" not in model:
        return model, None
    name, tag = model.split(":", 1)
    return name, tag or None


def _context_length(show_payload: Any) -> int | None:
    if not isinstance(show_payload, Mapping):
        return None
    info = show_payload.get("model_info")
    if not isinstance(info, Mapping):
        return None
    found: list[int] = []
    for key, value in info.items():
        if str(key).endswith("context_length") and isinstance(value, int) and not isinstance(value, bool):
            found.append(value)
    if not found:
        return None
    return max(found)


def _matching_model(tags_payload: Any, model: str) -> Mapping[str, Any] | None:
    if not isinstance(tags_payload, Mapping):
        return None
    models = tags_payload.get("models")
    if not isinstance(models, list):
        return None
    for item in models:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "")
        if name == model or name.endswith("/" + model):
            return item
    return None


def _processor(ps_payload: Any, model: str) -> str | None:
    if not isinstance(ps_payload, Mapping):
        return None
    models = ps_payload.get("models")
    if not isinstance(models, list):
        return None
    for item in models:
        if not isinstance(item, Mapping):
            continue
        if item.get("name") != model and not str(item.get("name") or "").endswith("/" + model):
            continue
        processor = item.get("processor")
        size = item.get("size")
        size_vram = item.get("size_vram")
        if isinstance(size, int) and isinstance(size_vram, int) and size > 0:
            if size_vram == size:
                return "100% GPU"
            if size_vram:
                return f"GPU {size_vram}/{size}"
            return "CPU"
        if isinstance(processor, str) and processor:
            return processor
    return None


def _get_json(url: str, *, opener: Opener, timeout: float) -> Any | None:
    try:
        with opener(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None


def _post_json(url: str, payload: Mapping[str, Any], *, opener: Opener, timeout: float) -> Any | None:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with opener(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=1,
            ).strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    return platform.processor() or platform.machine() or "unavailable"


def _ram_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError, AttributeError):
        return None


def _gpu_metadata() -> dict[str, str]:
    unavailable = {
        "gpu": "unavailable",
        "gpu_vram": "unavailable",
        "driver": "unavailable",
    }
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return unavailable
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if not rows:
        return unavailable
    names: list[str] = []
    vrams: list[str] = []
    drivers: list[str] = []
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        names.append(parts[0] if parts else "unavailable")
        vrams.append(parts[1] if len(parts) > 1 else "unavailable")
        drivers.append(parts[2] if len(parts) > 2 else "unavailable")
    return {
        "gpu": "; ".join(names),
        "gpu_vram": "; ".join(vrams),
        "driver": "; ".join(drivers),
    }


def _skip_hash_path(path: Path) -> bool:
    if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
        return True
    name = path.name
    return name == ".env" or name.endswith(".secret")


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
