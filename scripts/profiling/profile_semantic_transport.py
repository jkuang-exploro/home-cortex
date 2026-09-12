#!/usr/bin/env python3
"""Offline synthetic transport profile. Never contacts a model or reads data/."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from scripts import PROJECT_ROOT
import statistics
import subprocess
import time

from home_cortex.ollama import planner_chat_messages, _semantic_planner_examples, _PLANNER_INSTRUCTIONS
from scripts.benchmarks.semantic_planner_benchmark import build_json_fact_service
from home_cortex.semantic_transport import canonical_json, pack_capabilities, transport_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer-json', type=Path, help='Optional deployed tokenizer.json (requires tokenizers package)')
    parser.add_argument('--baseline-snapshot', type=Path, help='Optional frozen, synthetic-only pre-change capture')
    args = parser.parse_args()
    root = PROJECT_ROOT
    if args.tokenizer_json:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
        tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
        tokenizer_name = 'tokenizer.json sha256:' + hashlib.sha256(args.tokenizer_json.read_bytes()).hexdigest()
    else:
        import tiktoken
        tokenizer = tiktoken.get_encoding('cl100k_base')
        tokens = lambda text: len(tokenizer.encode(text, disallowed_special=()))
        tokenizer_name = 'cl100k_base (proxy; NOT deployed model tokenizer)'
    service, context = build_json_fact_service(root / 'benchmarks/fixtures/semantic-contract', root / 'schemas/edge', None)
    schema = service.engine.schema
    capabilities, output_schema = schema.planner_capability_payload(), schema.planner_output_schema()
    users = [{'role': 'user', 'content': 'List the members of this home.'}]
    now = context.current_time.isoformat()
    baseline = planner_chat_messages(users, capabilities, household_now=now)
    examples = _semantic_planner_examples()
    baseline_label = 'Expanded diagnostic representation; same grammar and examples, explicit eq defaults'
    if args.baseline_snapshot:
        snapshot = json.loads(args.baseline_snapshot.read_text())
        capabilities, output_schema = snapshot['capabilities'], snapshot['schema']
        baseline, examples = snapshot['messages'], snapshot['examples']
        now = '2026-09-09T12:00:00Z'
        baseline_label = 'Pre-change frozen synthetic capture at ' + snapshot['revision']
    codec = transport_for(output_schema)
    compact = [dict(message) for message in planner_chat_messages(users, capabilities, household_now=now)]
    compact[0]['content'] = (
        _PLANNER_INSTRUCTIONS + codec.instructions() + '\nCapabilities:\n'
        + canonical_json(pack_capabilities(capabilities))
    )
    for index, message in enumerate(examples):
        if message['role'] == 'assistant':
            compact[1 + index]['content'] = codec.encode(
                json.loads(message['content']), validate=False
            )

    def sizes(text):
        return {'bytes': len(text.encode()), 'tokens': tokens(text), 'sha256': hashlib.sha256(text.encode()).hexdigest()}
    def pair(before, after):
        a, b = sizes(before), sizes(after)
        return {'expanded': a, 'compact': b, 'token_reduction_pct': round(100 * (1 - b['tokens'] / a['tokens']), 2)}
    assistant_before = [item['content'] for item in examples if item['role'] == 'assistant']
    assistant_after = [item['content'] for item in compact[1:1 + len(examples)] if item['role'] == 'assistant']
    components = {
        'capabilities': pair(canonical_json(capabilities), canonical_json(pack_capabilities(capabilities))),
        'output_schema': pair(canonical_json(output_schema), canonical_json(codec.schema)),
        'assistant_examples': pair('\n'.join(assistant_before), '\n'.join(assistant_after)),
        'message_content_total': pair('\n'.join(m['content'] for m in baseline), '\n'.join(m['content'] for m in compact)),
        'structured_content_including_schema': pair(canonical_json(capabilities) + '\n' + '\n'.join(assistant_before) + '\n' + canonical_json(output_schema), canonical_json(pack_capabilities(capabilities)) + '\n' + '\n'.join(assistant_after) + '\n' + canonical_json(codec.schema)),
    }
    outputs = []
    decode_times, encode_times, json_times = [], [], []
    # Synthetic examples carry only demonstration names, never deployed identities.
    for index, before in enumerate(assistant_before):
        payload = json.loads(before)
        for condition in payload['request'].get('filters', []):
            if 'property' in condition:
                condition.setdefault('operator', 'eq')
        encoded = codec.encode(payload, validate=False)
        row = {'example_index': index, 'operation': payload['request']['operation'], **pair(before, encoded)}
        try:
            decoded = codec.decode(encoded)
            assert decoded == payload
            row['strict_parse'] = True
            for _ in range(10):
                start = time.perf_counter(); codec.decode(encoded); decode_times.append((time.perf_counter()-start)*1000)
                start = time.perf_counter(); codec.encode(payload); encode_times.append((time.perf_counter()-start)*1000)
                start = time.perf_counter(); json.loads(before); json_times.append((time.perf_counter()-start)*1000)
        except ValueError:
            row['strict_parse'] = False
            row['reason'] = 'Demonstration vocabulary unavailable in this synthetic catalog'
        outputs.append(row)
    components['structured_with_transport_instructions'] = pair(
        canonical_json(capabilities) + '\n' + '\n'.join(assistant_before) + '\n' + canonical_json(output_schema),
        canonical_json(pack_capabilities(capabilities)) + '\n' + '\n'.join(assistant_after) + '\n' + canonical_json(codec.schema) + codec.instructions(),
    )
    fixture_pattern = 'tests/static_test_data/**/*.json' if args.baseline_snapshot else 'benchmarks/fixtures/semantic-contract/**/*.json'
    paths = sorted({*root.glob('src/home_cortex/**/*.py'), *root.glob('schemas/**/*.yaml'), *root.glob(fixture_pattern), root / 'pyproject.toml', Path(__file__).resolve()})
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        revision = 'unversioned-package; see handoff-summary.json'
    report = {
        'codec_version': codec.version, 'schema_sha256': codec.schema_fingerprint,
        'field_dictionary_sha256': hashlib.sha256(canonical_json(codec.aliases).encode()).hexdigest(),
        'source_revision': revision,
        'baseline': baseline_label, 'tokenizer': tokenizer_name,
        'token_scope': 'Content strings only; excludes provider chat templates and hidden grammar compilation. Schema is measured separately.',
        'components': components, 'synthetic_outputs': outputs,
        'local_median_ms': {'strict_decode': statistics.median(decode_times) if decode_times else None, 'validated_encode': statistics.median(encode_times) if encode_times else None, 'json_parse_only': statistics.median(json_times) if json_times else None},
        'production': {'prompt_tokens': None, 'output_tokens': None, 'latency_ms': None, 'retry_rate': None, 'acceptance': 'Pending Grok in-situ profiling; no LLM calls made'},
        'fingerprints': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'components': {k: v['token_reduction_pct'] for k,v in components.items()}, 'strict_parse': sum(row['strict_parse'] for row in outputs), 'examples': len(outputs)}))


if __name__ == '__main__':
    main()
