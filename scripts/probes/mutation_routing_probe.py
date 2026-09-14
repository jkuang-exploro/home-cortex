"""Observe mutation-enabled planning; never execute any read or write plan."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from time import perf_counter

import yaml

from scripts import PROJECT_ROOT
from scripts.benchmarks.semantic_planner_benchmark import build_json_fact_service, collect_provenance, summarize_latencies
from home_cortex.ollama import OllamaService
from home_cortex.request_tracing import trace_request
from scripts.probes.two_stage_semantic_planner import LegacyTwoStagePlanner


class ObservedOllama(OllamaService):
    async def plan_item_mutation(self, messages):
        self.mutation_calls += 1
        decision, metrics = await super().plan_item_mutation(messages)
        self.decision = decision.model_dump(mode='json')
        return decision, metrics

    async def plan_semantic_fact(self, *args, **kwargs):
        self.semantic_calls += 1
        return await super().plan_semantic_fact(*args, **kwargs)


async def run(args):
    cases = yaml.safe_load(args.eval.read_text())
    client = ObservedOllama(args.ollama_url, args.model)
    service, context = build_json_fact_service(args.data_dir, PROJECT_ROOT / 'schemas/edge', client)
    service.planner = LegacyTwoStagePlanner(client, service.engine.schema)
    rows = []
    try:
        for sample in range(-1, args.repeat):
            selected = [('read', cases['read'][:1])] if sample == -1 else [
                (key, cases[key]) for key in ('read', 'write', 'ambiguous', 'mixed')
            ]
            for category, utterances in selected:
                for utterance in utterances:
                    client.mutation_calls = client.semantic_calls = 0
                    client.decision = None
                    start = perf_counter()
                    result = None
                    error = None
                    with trace_request() as trace:
                        try:
                            outcome = await service.planner.plan([{'role': 'user', 'content': utterance}], context)
                            result = outcome.plan.model_dump(mode='json', exclude_none=True)
                        except Exception as exc:
                            error = type(exc).__name__
                    if sample >= 0:
                        rows.append({'category': category, 'utterance': utterance, 'sample': sample,
                            'latency_ms': (perf_counter() - start) * 1000,
                            'mutation_calls': client.mutation_calls, 'semantic_calls': client.semantic_calls,
                            'llm_calls': sum(e['stage'].startswith('llm.') for e in trace.events),
                            'mutation_decision': client.decision, 'plan': result, 'error': error,
                            'model_timings': [e for e in trace.events if e['stage'].startswith('llm.')]})
    finally:
        await client.close()
    report = {'scope': 'Planner only; no executor, write dispatch, HTTP or live DB I/O',
        'repeat': args.repeat, 'rows': rows,
        'by_category': {key: {'requests': len(group := [r for r in rows if r['category'] == key]),
            'llm_calls': summarize_latencies([r['llm_calls'] for r in group]),
            'latency_ms': summarize_latencies([r['latency_ms'] for r in group])}
            for key in ('read', 'write', 'ambiguous', 'mixed')},
        'manifest_sha256': hashlib.sha256((PROJECT_ROOT / 'CANDIDATE-MANIFEST.json').read_bytes()).hexdigest(),
        'provenance': collect_provenance(root=PROJECT_ROOT, eval_path=args.eval,
            ollama_url=args.ollama_url, ollama_model=args.model, backend='planning-only-frozen-json',
            frozen_time=context.current_time, warmup=1, repeat=args.repeat, verified_cold=False,
            data_dir=args.data_dir, schema_dir=PROJECT_ROOT / 'schemas')}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['by_category'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval', type=Path, default=PROJECT_ROOT / 'benchmarks/mutation_routing.yaml')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--ollama-url', default='http://ollama:11434')
    parser.add_argument('--model', default='qwen3.5:9b')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1 or args.output.exists():
        parser.error('Require positive repeats and a new output path')
    asyncio.run(run(args))
