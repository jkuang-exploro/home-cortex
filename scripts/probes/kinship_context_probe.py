#!/usr/bin/env python3
"""Synthetic kinship composition regression probe.

Prepare once from each isolated baseline/candidate checkout, then run the two
payload files on the GPU host. Only the interpreter is measured, without retries;
this is not a production household accuracy benchmark.
"""
import argparse

def prepare(output, model):
    import json, sys, hashlib
    from pathlib import Path
    from home_cortex.ollama import planner_chat_messages
    from scripts.benchmarks.semantic_planner_benchmark import build_json_fact_service
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
            body = {'model': model, 'messages': planner_chat_messages(users, schema.planner_capability_payload(), household_now=ctx.current_time.isoformat()), 'format': schema.planner_output_schema(), 'stream': False, 'think': False, 'keep_alive': '24h', 'options': {'num_ctx': 8192, 'temperature': 0, 'seed': 0, 'num_predict': 384}}
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
    models = {row['body']['model'] for variant in ('baseline', 'candidate')
              for row in json.loads((root / f'kinship-{variant}.json').read_text())['rows']}
    if len(models) != 1:
        raise ValueError('Both variants must use the same model')
    model = models.pop()
    resident = next((x for x in env['ps']['models'] if x['name'] == model), None)
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
def score(results_path, output):
    import json
    import statistics
    from pathlib import Path
    from home_cortex.semantic_ir import SemanticFactRequest, SemanticReference
    from scripts.benchmarks.semantic_planner_benchmark import build_json_fact_service

    root = Path.cwd()
    service, _ = build_json_fact_service(root / 'benchmarks/fixtures/semantic-contract', root / 'schemas/edge', None)
    schema = service.engine.schema
    data = json.loads(Path(results_path).read_text())
    summary = {key: data[key] for key in ('environment', 'after', 'payload_hashes')}
    summary['scoring'] = 'Exact expanded subject and schema validity; not household answer accuracy'
    summary['variants'] = {}
    for row in data['rows']:
        variant = summary['variants'].setdefault(row['variant'], {'cases': {}, 'wall_ms': []})
        case = variant['cases'].setdefault(row['case'], {'correct': 0, 'total': 0, 'failures': []})
        case['total'] += 1
        variant['wall_ms'].append(row['wall_ms'])
        try:
            raw = json.loads(row['content'])
            request = SemanticFactRequest.model_validate(schema.expand_planner_concepts(raw)['request'])
            expected = SemanticReference.model_validate(row['expected_subject'])
            correct = raw.get('requires_fact') is True and schema.validates(request) and request.subject == expected
            if not correct:
                case['failures'].append({'repeat': row['repeat'], 'subject': raw.get('request', {}).get('subject')})
        except (ValueError, KeyError, TypeError) as error:
            correct = False
            case['failures'].append({'repeat': row['repeat'], 'error': str(error)})
        case['correct'] += int(correct)
    for variant in summary['variants'].values():
        variant['mean_wall_ms'] = statistics.mean(variant.pop('wall_ms'))
        variant['correct'] = sum(case['correct'] for case in variant['cases'].values())
        variant['total'] = sum(case['total'] for case in variant['cases'].values())
    Path(output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({name: {k: v[k] for k in ('correct', 'total', 'mean_wall_ms')} for name, v in summary['variants'].items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare')
    prepare_parser.add_argument('--output', required=True)
    prepare_parser.add_argument('--model', required=True)
    commands.add_parser('run').add_argument('--root', required=True)
    score_parser = commands.add_parser('score')
    score_parser.add_argument('--results', required=True)
    score_parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output, args.model)
    elif args.command == 'run':
        run(args.root)
    else:
        score(args.results, args.output)
