#!/usr/bin/env bash
# Copy the Tier-1 probe into a running cortex-api container with distinct paths.
# Usage (from repo root or docker/cortex):
#   ./scripts/maintenance/copy_tier1_bench_into_api.sh [compose-dir]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
COMPOSE_DIR="${1:-$ROOT/docker/cortex}"
SERVICE="${CORTEX_API_SERVICE:-cortex-api}"

if [[ ! -f "$ROOT/scripts/benchmarks/tier1_latency_bench.py" ]]; then
  echo "missing $ROOT/scripts/benchmarks/tier1_latency_bench.py" >&2
  exit 1
fi
if [[ ! -f "$ROOT/benchmarks/semantic_planner_eval.yaml" ]]; then
  echo "missing eval YAML" >&2
  exit 1
fi

cd "$COMPOSE_DIR"
docker compose exec -T "$SERVICE" mkdir -p /tmp/tier1-bench /tmp/tier1-bench/pkg /tmp/tier1-artifacts
# Replace copied trees; docker compose cp into an existing dir nests a second copy.
docker compose exec -T "$SERVICE" rm -rf \
  /tmp/tier1-bench/pkg/home_cortex \
  /tmp/tier1-bench/pkg/scripts \
  /tmp/tier1-bench/schemas \
  /tmp/tier1-bench/run_probe.py \
  /tmp/tier1-bench/semantic_planner_eval.yaml
docker compose cp "$ROOT/benchmarks/semantic_planner_eval.yaml" "$SERVICE":/tmp/tier1-bench/semantic_planner_eval.yaml
docker compose cp "$ROOT/schemas" "$SERVICE":/tmp/tier1-bench/schemas
docker compose cp "$ROOT/scripts" "$SERVICE":/tmp/tier1-bench/pkg/scripts
docker compose cp "$ROOT/src/home_cortex" "$SERVICE":/tmp/tier1-bench/pkg/home_cortex
docker compose exec -T "$SERVICE" find /tmp/tier1-bench -name '._*' -delete
docker compose exec -T "$SERVICE" find /tmp/tier1-bench -name '.DS_Store' -delete

docker compose exec -T "$SERVICE" python - <<'PY'
from pathlib import Path
p = Path("/tmp/tier1-bench/pkg/scripts/benchmarks/tier1_latency_bench.py")
text = p.read_text(encoding="utf-8")
if not text.startswith("#!/usr/bin/env python3"):
    raise SystemExit(f"{p} is not the probe runner (got {text[:40]!r})")
if "utterance:" in text.splitlines()[0:5]:
    raise SystemExit(f"{p} looks like YAML, not Python")
print("copied ok:", p, "lines", text.count(chr(10)) + 1)
PY

cat <<'EOF'
Run:

  docker compose exec -e PYTHONPATH=/tmp/tier1-bench/pkg cortex-api \
    python -m scripts.benchmarks.tier1_latency_bench \
      --ollama-url http://ollama:11434 \
      --model qwen3.5:9b \
      --data-dir /app/data \
      --schema-dir /tmp/tier1-bench/schemas/edge \
      --eval /tmp/tier1-bench/semantic_planner_eval.yaml \
      --warmup 1 --repeat 5 \
      --output /tmp/tier1-artifacts/probe.json
EOF
