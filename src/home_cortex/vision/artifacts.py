"""Local content-addressed media store. Bytes never live in household records."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .contracts import VisionContractError

_REF = re.compile(r"^sha256:([0-9a-f]{64})(\.[A-Za-z0-9]+)?$")
_SUFFIX = re.compile(r"^\.[A-Za-z0-9]+$")


class ArtifactStore:
    """Filesystem store keyed by SHA-256. Layout: <root>/<aa>/<rest>[suffix]."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes | Path, *, suffix: str | None = None) -> str:
        if isinstance(data, Path):
            payload = data.read_bytes()
            if suffix is None:
                suffix = data.suffix
        else:
            payload = data
        digest = hashlib.sha256(payload).hexdigest()
        suffix = _canonical_suffix(suffix)
        ref = f"sha256:{digest}{suffix}"
        dest = self.path(ref)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(payload)
        elif dest.read_bytes() != payload:
            raise VisionContractError(f"artifact collision at {ref}")
        return ref

    def get(self, reference: str) -> bytes:
        path = self.path(reference)
        if not path.is_file():
            raise FileNotFoundError(reference)
        return path.read_bytes()

    def exists(self, reference: str) -> bool:
        return self.path(reference).is_file()

    def path(self, reference: str) -> Path:
        match = _REF.fullmatch(reference)
        if match is None:
            raise VisionContractError(
                "artifact_ref must be sha256:<hex> with optional suffix"
            )
        digest, suffix = match.group(1), match.group(2) or ""
        return self.root / digest[:2] / f"{digest[2:]}{suffix}"


def _canonical_suffix(suffix: str | None) -> str:
    if not suffix:
        return ""
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    if _SUFFIX.fullmatch(suffix) is None:
        raise VisionContractError("artifact suffix must be a simple extension")
    return suffix.lower()
