"""Isolated, example-only prompt experiments. Never modifies serving defaults.

Run from a frozen package with a frozen graph. Compare identical case sets and
repeat counts. Gold-plan execution is a differential answer oracle, not an
independent verification of the executor or household data.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

from home_cortex import ollama as prompts
from home_cortex.semantic_ir import SemanticFactRequest
from scripts import PROJECT_ROOT
from scripts.profiling.planner_prompt_audit import compact, prompt_components
from scripts.benchmarks.semantic_planner_benchmark import (
    SemanticEvalCase, build_json_fact_service, collect_provenance, evaluate_planner_case,
    load_semantic_eval_cases, load_probe_dataset, load_bilingual_dataset, serialize_fact_result, summarize_latencies, summarize_scores,
)


@contextmanager
def reduced_examples(indices):
    original = prompts._semantic_planner_examples
    baseline = original()
    if any(i < 0 or i >= len(baseline) // 2 for i in indices):
        raise ValueError('Example index outside frozen baseline')
    selected = [m for i, m in enumerate(baseline) if i // 2 not in indices]
    prompts._semantic_planner_examples = lambda: [dict(m) for m in selected]
    try:
        yield
    finally:
        prompts._semantic_planner_examples = original


def fixed_cases(path, schema):
    import yaml
    payload = yaml.safe_load(path.read_text())
    return [SemanticEvalCase(utterance=row['utterance'], speaker_id='person:jian_kuang',
        category=row['category'], plan_id=row['id'], case_id=row['id'],
        expected=SemanticFactRequest.model_validate(schema.expand_planner_concepts({'requires_fact': True, 'request': row['request']})['request'])) for row in payload['cases']]


def budget_checks(rows, normal_token_budget):
    """Measured calls only; this does not bound arbitrary future user input."""
    calls = [call for row in rows for call in row.get('model_calls', [])]
    normal = [call for row in rows if row['planner_attempt_count'] == 1 for call in row.get('model_calls', [])]
    return {
        'normal_token_budget': normal_token_budget,
        'all_counts_available': bool(calls) and all(call['prompt_tokens'] is not None for call in calls),
        'normal_exceeded': sum(call['prompt_tokens'] > normal_token_budget for call in normal if call['prompt_tokens'] is not None),
        'context_exceeded': sum(call['prompt_tokens'] + prompts.PLANNER_NUM_PREDICT > prompts.OLLAMA_NUM_CTX for call in calls if call['prompt_tokens'] is not None),
        'length_stops': sum(call['done_reason'] == 'length' for call in calls),
    }


class MeasuredOllama(prompts.OllamaService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    async def _chat(self, **kwargs):
        response = await super()._chat(**kwargs)
        self.calls.append({'prompt_tokens': response.prompt_eval_count,
                           'output_tokens': response.eval_count,
                           'done_reason': response.done_reason})
        return response


def answer_signature(result):
    if result is None:
        return None
    return {k: result[k] for k in ('status', 'value', 'unit', 'shape', 'rows', 'primary_entity_ids', 'missing_requirements')}


async def run(args):
    client = MeasuredOllama(args.ollama_url, args.model)
    service, context = build_json_fact_service(args.data_dir, PROJECT_ROOT / 'schemas/edge', client)
    cases = fixed_cases(args.eval, service.engine.schema)
    if args.regression:
        cases += list(load_probe_dataset().cases)
        cases += list(load_semantic_eval_cases())
        dataset = load_bilingual_dataset()
        for pair in dataset['pairs']:
            for language in ('zh', 'en'):
                payload = json.loads(json.dumps(pair['expected']))
                if language == 'en' and pair.get('ignore_named_value'):
                    payload['subject']['value'] = 'storage box'
                expected = SemanticFactRequest.model_validate(service.engine.schema.expand_planner_concepts({'request': payload})['request'])
                cases.append(SemanticEvalCase(utterance=pair[language], speaker_id=context.caller_entity_id,
                    category=pair['category'], plan_id=pair['id'], case_id=f"{pair['id']}:{language}", expected=expected))
        cases = [replace(c, case_id=f'{i}:{c.case_id or c.plan_id}') for i, c in enumerate(cases)]
    if args.utterance:
        missing = set(args.utterance) - {case.utterance for case in cases}
        if missing:
            raise ValueError(f'Utterances absent from fixed evaluation: {sorted(missing)}')
        cases = [case for case in cases if case.utterance in args.utterance]
    gold = {}
    for case in cases:
        result, _, _, _ = await service.engine.execute(case.expected, replace(context, caller_entity_id=case.speaker_id))
        gold[case.case_id] = serialize_fact_result(result)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.with_suffix('.jsonl')
    if checkpoint.exists():
        raise FileExistsError(f'Refusing to mix runs: {checkpoint}')
    try:
        with reduced_examples(set(args.drop)):
            parts, messages = prompt_components([{'role': 'user', 'content': cases[0].utterance}],
                service.engine.schema.planner_capability_payload(), household_now=context.current_time.isoformat())
            prompt = {'messages_sha256': hashlib.sha256(compact(messages).encode()).hexdigest(),
                      'example_pairs': len(prompts._semantic_planner_examples()) // 2,
                      'component_bytes': {k: len(v.encode()) for k, v in parts.items()}}
            await evaluate_planner_case(service, context, cases[0], phase='warmup')
            for repeat in range(args.repeat):
                for case in cases:
                    call_offset = len(client.calls)
                    start = perf_counter()
                    row = await evaluate_planner_case(service, context, case, sample_index=repeat)
                    row['total_request_ms'] = (perf_counter() - start) * 1000
                    row['model_calls'] = client.calls[call_offset:]
                    row['gold_executor'] = gold[case.case_id]
                    row['answer_correct'] = answer_signature(row['executor']) == answer_signature(gold[case.case_id])
                    rows.append(row)
                    with checkpoint.open('a') as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    finally:
        await client.close()
    metrics = {}
    for key, values in {
        'prompt_tokens': [r['planner_diagnostics'].get('prompt_eval_count') for r in rows],
        'output_tokens': [r['planner_diagnostics'].get('eval_count') for r in rows],
        'attempts': [r['planner_attempt_count'] for r in rows],
        'planner_ms': [r['planner_latency_ms'] for r in rows],
        'total_ms': [r['total_request_ms'] for r in rows],
    }.items():
        metrics[key] = summarize_latencies([v for v in values if v is not None])
    report = {'drop_indices': args.drop, 'cases': len(cases), 'repeat': args.repeat, 'prompt': prompt,
              'scores': summarize_scores(rows), 'metrics': metrics,
              'retries': sum(r['planner_attempt_count'] > 1 for r in rows),
              'budgets': budget_checks(rows, args.normal_token_budget),
              'failures': [{k: r[k] for k in ('case_id', 'utterance', 'sample_index', 'plan_match', 'answer_correct', 'validation_result', 'executor_status')} for r in rows if not r['plan_match'] or not r['answer_correct']],
              'evaluation_cases_sha256': hashlib.sha256(compact([{'utterance': c.utterance, 'speaker': c.speaker_id, 'expected': c.expected.model_dump(mode='json'), 'alternatives': [a.model_dump(mode='json') for a in c.acceptable_alternatives]} for c in cases]).encode()).hexdigest(),
              'frozen_manifest_sha256': hashlib.sha256((PROJECT_ROOT / 'CANDIDATE-MANIFEST.json').read_bytes()).hexdigest() if (PROJECT_ROOT / 'CANDIDATE-MANIFEST.json').exists() else None,
              'provenance': collect_provenance(root=PROJECT_ROOT, eval_path=args.eval,
                  ollama_url=args.ollama_url, ollama_model=args.model, backend='frozen-json',
                  frozen_time=context.current_time, warmup=1, repeat=args.repeat, verified_cold=False,
                  data_dir=args.data_dir, schema_dir=PROJECT_ROOT / 'schemas')}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.enforce_budgets and (not report['budgets']['all_counts_available'] or any(report['budgets'][key] for key in ('normal_exceeded', 'context_exceeded', 'length_stops'))):
        raise SystemExit('Measured prompt budget failed; see summary')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--eval', type=Path, default=PROJECT_ROOT / 'benchmarks/planner_prompt_compression.yaml')
    parser.add_argument('--ollama-url', default='http://ollama:11434')
    parser.add_argument('--model', default='qwen3.5:9b')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--drop', type=int, action='append', default=[])
    parser.add_argument('--regression', action='store_true')
    parser.add_argument('--utterance', action='append', default=[], help='Repeatable exact evaluation selector for regression confirmation')
    parser.add_argument('--normal-token-budget', type=int, default=8500, help='Measured qwen3.5:9b baseline guardrail, separate from 16K context')
    parser.add_argument('--enforce-budgets', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    asyncio.run(run(args))
