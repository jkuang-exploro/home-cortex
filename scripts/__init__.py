"""Engineering utilities, importable independently of the application package."""
from pathlib import Path

# Source checkouts and frozen archives keep inputs beside scripts/. Installed
# container wheels use /app, where Docker copies schemas and benchmark inputs.
_source_root = Path(__file__).resolve().parent.parent
PROJECT_ROOT = next(
    (root for root in (_source_root, _source_root.parent, Path("/app"))
     if (root / "schemas").is_dir() and (root / "benchmarks").is_dir()),
    _source_root,
)
