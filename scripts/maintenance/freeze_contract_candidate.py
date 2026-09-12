#!/usr/bin/env python3
"""Freeze source + synthetic evaluation inputs; never reads runtime data or secrets."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import tarfile
from pathlib import Path
from scripts import PROJECT_ROOT

ROOT = PROJECT_ROOT


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def freeze(output: Path):
    from scripts.benchmarks.composition_eval import household_engine, composition_fingerprint_payload
    from home_cortex.semantic_schema import SemanticSchemaRegistry
    from home_cortex.semantic_ontology import SemanticOntology

    payload = {}
    # Deliberate allowlist. In particular, exclude data/, docker envs, .git,
    # generated production probes, local configuration and credentials.
    for directory in ('src/home_cortex', 'scripts', 'schemas', 'benchmarks', 'tests'):
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts and path.suffix in {'.py', '.sh', '.yaml', '.json', '.md'}:
                payload[str(path.relative_to(ROOT))] = path.read_bytes()
    for name in ('pyproject.toml', 'AGENTS.md', 'scripts/maintenance/freeze_contract_candidate.py',
                 'artifacts/generic-contracts/REPORT.md'):
        payload[name] = (ROOT / name).read_bytes()
    views = {}
    for household in ('alpha', 'beta', 'gamma'):
        engine, _ = household_engine(household)
        for version, ontology in [('v1', engine.schema.ontology), ('v2', SemanticOntology.from_file(ROOT / 'schemas/semantic/ontology-v2.yaml'))]:
            schema = SemanticSchemaRegistry(engine.schema.catalog, ontology)
            for kind, view in [('capabilities', schema.planner_capability_payload()), ('output-schema', schema.planner_output_schema())]:
                name = f'contract-views/{household}-{version}-{kind}.json'
                payload[name] = encoded(view)
                views[name] = {'sha256': sha(payload[name]), 'bytes': len(payload[name])}
            if schema.contracts:
                views[f'{household}-resolved-contract'] = {'sha256': schema.contracts.fingerprint}
    manifest = {
        'format': 1, 'profile': 'explicit ontology-v2.yaml; default ontology.yaml remains V1',
        'git_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'files': {name: sha(data) for name, data in sorted(payload.items())},
        'views': views, 'composition': composition_fingerprint_payload(),
        'local_python': platform.python_version(),
        'local_packages': {name: importlib.metadata.version(name) for name in ('pydantic', 'pyyaml', 'ollama', 'pytest', 'fastapi')},
        'evaluation': 'No live model measurements; see included deterministic verification report.',
    }
    payload_id = sha(encoded(manifest))
    payload['CANDIDATE-MANIFEST.json'] = encoded(manifest)
    archive = io.BytesIO()
    with gzip.GzipFile(fileobj=archive, mode='wb', mtime=0, filename='') as compressed:
        with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as tar:
            for name, data in sorted(payload.items()):
                info = tarfile.TarInfo(f'home-cortex-contracts/{name}')
                info.size, info.mode, info.mtime = len(data), 0o444, 0
                tar.addfile(info, io.BytesIO(data))
    output.mkdir(parents=True, exist_ok=True)
    filename = f'candidate-{payload_id[:16]}.tar.gz'
    data = archive.getvalue()
    target = output / filename
    if target.exists() and target.read_bytes() != data:
        raise RuntimeError('Refusing to replace a different frozen package')
    if not target.exists():
        target.write_bytes(data)
        target.chmod(0o444)
    summary = {'archive': filename, 'archive_sha256': sha(data), 'archive_bytes': len(data),
               'payload_id': payload_id, 'manifest_sha256': sha(payload['CANDIDATE-MANIFEST.json']),
               'file_count': len(manifest['files']), 'git_base': manifest['git_base'],
               'views': views, 'composition': manifest['composition'],
               'local_python': manifest['local_python'], 'local_packages': manifest['local_packages']}
    (output / 'candidate-summary.json').write_bytes(encoded(summary))
    print(target)
    print(summary['archive_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/generic-contracts')
    freeze(parser.parse_args().output)
