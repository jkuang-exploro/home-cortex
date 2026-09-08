"""Cross-process prefix regression using the synthetic schema and real builder."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_probe_prefixes_are_stable_across_hash_seeds(tmp_path):
    root = Path(__file__).resolve().parents[1]
    reports = []
    for seed in ('0', '1'):
        output = tmp_path / f'{seed}.json'
        subprocess.run([
            sys.executable, str(root / 'scripts/ollama_prefix_reuse_probe.py'),
            '--root', str(root), '--output', str(output), '--dry-run',
        ], env={**os.environ, 'PYTHONHASHSEED': seed, 'PYTHONPATH': str(root / 'src')},
           check=True, capture_output=True, text=True)
        reports.append(json.loads(output.read_text()))
    canonical = [[row for row in report['experiments'] if row['ordering'] == 'canonical'] for report in reports]
    assert canonical[0] == canonical[1]
    assert len({row["static_prefix_sha256"] for row in canonical[0]}) == 1
    # Conversation histories share the complete grammar/examples; ordinary chat does not.
    rows = {row['traffic']: row for row in canonical[0]}
    assert all(item['identical_leading_messages'] > 40 for item in rows['two_conversations']['adjacent_prefixes'])
    assert all(item['identical_leading_messages'] == 0 for item in rows['alternating_chat']['adjacent_prefixes'])
