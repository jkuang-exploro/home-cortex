#!/usr/bin/env python3
"""Matched synthetic traffic probe; no unload, upgrade, or production data reads.

Common prefix bytes describe message serialization, NOT actual KV cache hits.
Planner bodies use the real message builder/schema. Chat bodies use OllamaService.
Direct transport excludes validation retries and deterministic execution.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from scripts.probes.ollama_warm_load_probe import (
    OllamaHttp, snapshot, assert_safe_resident, metrics, summarize, compact,
    MODEL, NUM_CTX, KEEP_ALIVE,
)


def digest(value):
    return hashlib.sha256(compact(value).encode()).hexdigest()


def prefix_info(previous, current):
    a, b = compact(previous).encode(), compact(current).encode()
    count = next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), min(len(a), len(b)))
    messages = next((i for i, pair in enumerate(zip(previous, current)) if pair[0] != pair[1]), min(len(previous), len(current)))
    return {'common_message_bytes': count, 'identical_leading_messages': messages}


async def build_traffic(root, ordering):
    import home_cortex.ollama as module
    from scripts.benchmarks.semantic_planner_benchmark import build_json_fact_service
    service, context = build_json_fact_service(
        root / 'benchmarks/fixtures/semantic-contract', root / 'schemas/edge', None,
    )
    capabilities = service.engine.schema.planner_capability_payload()
    original = module.planner_system_prompt
    if ordering == 'native':
        module.planner_system_prompt = lambda caps: module._PLANNER_INSTRUCTIONS + '\nCapabilities:\n' + compact(caps)

    class Capture:
        async def chat(self, **kwargs):
            self.body = kwargs
            return module.ChatResponse(message={'role': 'assistant', 'content': '{}'})

    capture = Capture()
    model = module.OllamaService('http://unused', MODEL, client=capture)
    now = context.current_time.isoformat()
    async def plan(users, clock=now):
        await model.plan_semantic_fact(
            [{'role': 'user', 'content': text} for text in users], capabilities,
            service.engine.schema.planner_output_schema(), household_now=clock,
        )
        return capture.body
    try:
        first = await plan(['Who am I?'])
        clock_only = await plan(['Who am I?'], now[:-6] + '+00:00')
        query_only = await plan(['How many members live in this household?'])
        varied = [await plan([question], now[:-6] + '+00:00') for question in
                  ['How many members live in this household?', 'Who is the youngest household member?']]
        # Interleaved growing histories; no resolved identities are sent to the model.
        conversations = [
            await plan(['Who is my daughter?']), await plan(['Who is my spouse?']),
            await plan(['Who is my daughter?', 'When is her next birthday?']),
            await plan(['Who is my spouse?', 'How old are they?']),
        ]
        from home_cortex.agents.registry import get_agent
        from home_cortex.agent_service import _clock_context
        definition = get_agent('steward')
        await model.chat_with_tools([{'role': 'system', 'content': definition.prompt},
                          *_clock_context('America/Los_Angeles', context.current_time),
                          {'role': 'user', 'content': 'Say hello in one short sentence.'}],
                          definition.tool_definitions)
        chat = capture.body
        return {
            'repeat': [('planner', first)] * 4,
            'vary_clock_only': [('planner', body) for body in [first, clock_only, first, clock_only]],
            'vary_query_only': [('planner', body) for body in [first, query_only, first, query_only]],
            'vary_query_clock': [('planner', body) for body in [first, *varied, first]],
            'two_conversations': [('planner', body) for body in conversations],
            'alternating_chat': [('planner', first), ('chat', chat), ('planner', first), ('chat', chat)],
            'alternating_chat_varied': [('planner', first), ('chat', chat), ('planner', varied[0]), ('chat', chat)],
        }
    finally:
        module.planner_system_prompt = original


def run(client, traffic, repeats, warmup):
    rows = []
    previous = []
    for cycle in range(warmup + repeats):
        for position, (kind, body) in enumerate(traffic):
            assert body['options']['num_ctx'] == NUM_CTX and body['keep_alive'] == KEEP_ALIVE
            response = client.request('/api/chat', body)
            row = {'cycle': cycle, 'position': position, 'kind': kind,
                   'request_sha256': digest(body), 'format_sha256': digest(body.get('format')),
                   **prefix_info(previous, body['messages']), **metrics(response)}
            content = (response.get('message') or {}).get('content', '')
            row['output_sha256'] = digest(content)
            if kind == 'planner':
                try:
                    parsed = json.loads(content)
                    row['json_object'] = isinstance(parsed, dict)
                    row['parsed_output_sha256'] = hashlib.sha256(json.dumps(parsed, sort_keys=True).encode()).hexdigest()
                except ValueError:
                    row['json_object'] = False
            previous = body['messages']
            if cycle >= warmup:
                rows.append(row)
    return {'rows': rows, 'by_kind': {
        kind: {key: summarize([row[key] for row in rows if row['kind'] == kind and row.get(key) is not None])
               for key in ['wall_ms', 'load_ms', 'prefill_ms', 'generation_ms', 'eval_count']}
        for kind in sorted({row['kind'] for row in rows})}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ollama-url', default='http://ollama:11434')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--reverse', action='store_true', help='Reverse variant order to check time/order effects')
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 1:
        parser.error('repeat and warmup must be positive')
    report = {'hash_seed': os.environ.get('PYTHONHASHSEED'), 'experiments': [],
              'files': {str(p.relative_to(args.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for folder in ['src/home_cortex', 'schemas', 'benchmarks/fixtures/semantic-contract', 'scripts']
                        for p in sorted((args.root / folder).rglob('*'))
                        if p.is_file() and p.suffix in {'.py', '.yaml', '.json', '.md'}}}
    client = OllamaHttp(args.ollama_url, reuse=True)
    try:
        if not args.dry_run:
            report['before'] = snapshot(client)
            resident = report['before']['resident']
            if not resident.get('name') or resident.get('context') != NUM_CTX:
                raise RuntimeError('Requires an already resident model with verified 8192 context')
        for ordering in (['canonical', 'native'] if args.reverse else ['native', 'canonical']):
            traffic = asyncio.run(build_traffic(args.root, ordering))
            from home_cortex.ollama import _semantic_planner_examples
            static_count = 1 + len(_semantic_planner_examples())
            for name, sequence in traffic.items():
                entry = {'ordering': ordering, 'traffic': name,
                         'static_prefix_sha256': digest(sequence[0][1]['messages'][:static_count]),
                         'request_hashes': [digest(body) for _, body in sequence],
                         'adjacent_prefixes': [prefix_info(a[1]['messages'], b[1]['messages'])
                                              for a, b in zip(sequence, sequence[1:])]}
                if not args.dry_run:
                    entry.update(run(client, sequence, args.repeat, args.warmup))
                    after = snapshot(client)
                    assert_safe_resident(report['before'], after, name)
                    entry['resident_after'] = after['resident']
                report['experiments'].append(entry)
                print(compact({key: value for key, value in entry.items() if key in {'ordering', 'traffic', 'by_kind'}}), flush=True)
        if not args.dry_run:
            report['after'] = snapshot(client)
    finally:
        client.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
