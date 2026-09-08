#!/usr/bin/env python3
"""Synthetic kinship composition regression probe.

Prepare once from each isolated baseline/candidate checkout, then run the two
payload files on the GPU host. Only the interpreter is measured, without retries;
this is not a production household accuracy benchmark.
"""
import argparse

def prepare(output):
    import json, sys, hashlib
    from pathlib import Path
    from home_cortex.ollama import planner_chat_messages
    from home_cortex.semantic_planner_benchmark import build_json_fact_service
    root = Path.cwd()
    service, ctx = build_json_fact_service(root / 'benchmarks/fixtures/semantic-contract', root / 'schemas/edge', None)
    schema = service.engine.schema
    sequence = [('我是谁', []), ('你是谁', []), ('我父亲是谁', ['father']), ('我岳父是谁', ['father_in_law']), ('我岳父是谁', ['father_in_law']), ('我老婆是谁', ['wife']), ('我岳父的生日是哪天', ['father_in_law']), ('我岳父是谁', ['father_in_law'])]
    held = [('我岳父是谁', ['father_in_law']), ('公公的出生日期是什么？', ['father_in_law']), ('请告诉我岳母是哪位。', ['mother_in_law']), ('Who is my father-in-law?', ['father_in_law']), ('我的妻子的父亲是哪位？', ['wife', 'father']), ('我配偶的岳父是哪位？', ['spouse', 'father_in_law']), ('我女儿的岳父是哪位？', ['daughter', 'father_in_law']), ('我外公是哪位？', ['maternal_grandfather']), ('我母亲的外公是哪位？', ['mother', 'maternal_grandfather'])]
    rows = []
    users = []
    for group, cases in [('sequence', sequence), ('minimal_pair', sequence[2:4]), ('heldout', held)]:
        users = []
        for i, (text, concepts) in enumerate(cases):
            if group in {'sequence', 'minimal_pair'}:
                users = users[-7:] + [{'role': 'user', 'content': text}]
            else:
                users = [{'role': 'user', 'content': text}]
            raw = {'request': {'subject': {'kind': 'assistant' if text == '你是谁' else 'self', 'entity_type': 'person', 'path': [{'concept': c} for c in concepts]}}}
            expected = schema.expand_planner_concepts(raw)['request']['subject']
            body = {'model': 'qwen3.5:9b', 'messages': planner_chat_messages(users, schema.planner_capability_payload(), household_now=ctx.current_time.isoformat()), 'format': schema.planner_output_schema(), 'stream': False, 'think': False, 'keep_alive': '24h', 'options': {'num_ctx': 8192, 'temperature': 0, 'seed': 0, 'num_predict': 384}}
            rows.append({'case': f'{group}-{i}', 'expected_subject': expected, 'body': body})
    Path(output).write_text(json.dumps({'rows': rows, 'source_sha256': hashlib.sha256((root / 'src/home_cortex/ollama.py').read_bytes()).hexdigest()}, ensure_ascii=False))

def run(root_path):
    import json, time, urllib.request, hashlib
    from pathlib import Path
    root = Path(root_path)

    def req(path, body=None):
        r = urllib.request.Request('http://ollama:11434' + path, data=json.dumps(body).encode() if body else None, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(r, timeout=180) as f:
            return json.load(f)
    env = {'version': req('/api/version'), 'ps': req('/api/ps')}
    resident = next((x for x in env['ps']['models'] if x['name'] == 'qwen3.5:9b'), None)
    assert resident and resident.get('context_length') == 8192, 'Requires resident 8192 model'
    report = {'environment': env, 'rows': [], 'payload_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.json')}}
    try:
        for repeat in range(3):
            for variant in ['baseline', 'candidate'] if repeat % 2 == 0 else ['candidate', 'baseline']:
                data = json.loads((root / f'kinship-{variant}.json').read_text())
                req('/api/chat', data['rows'][0]['body'])
                for case in data['rows']:
                    start = time.perf_counter()
                    out = req('/api/chat', case['body'])
                    report['rows'].append({'variant': variant, 'repeat': repeat, 'case': case['case'], 'expected_subject': case['expected_subject'], 'content': out['message']['content'], 'wall_ms': (time.perf_counter() - start) * 1000})
                print(variant, repeat, 'done', flush=True)
    finally:
        report['after'] = req('/api/ps')
        (root / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare').add_argument('--output', required=True)
    commands.add_parser('run').add_argument('--root', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output)
    else:
        run(args.root)
