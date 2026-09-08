#!/usr/bin/env python3
"""Isolated semantic-pipeline audit; replay is NOT a model accuracy benchmark."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from ollama import ChatResponse
from home_cortex import ollama as prompts
from home_cortex.ollama import OllamaService
from home_cortex.profiling import trace_request
from home_cortex.semantic_conversation import SemanticConversationService
from home_cortex.semantic_facts import SemanticFactRequest
from home_cortex.semantic_planner_benchmark import (
    build_json_fact_service, collect_provenance, load_probe_dataset, SemanticEvalCase,
    score_structured_result, normalize_semantic_request, summarize_latencies,
)

ROOT = Path(__file__).resolve().parents[1]


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


class ReplayClient:
    """Replays declared gold IR; exercises serialization but performs no inference."""
    payload = None

    async def chat(self, **kwargs):
        json.loads(compact(kwargs))
        return ChatResponse(message={'role': 'assistant', 'content': compact(self.payload)})


def components(service, messages, now):
    capabilities = service.engine.schema.planner_capability_payload()
    built = prompts.planner_chat_messages(messages, capabilities, household_now=now)
    examples = prompts._semantic_planner_examples()
    values = {
        'instructions': prompts._PLANNER_INSTRUCTIONS,
        'capabilities': compact(capabilities),
        'examples': ''.join(item['content'] for item in examples),
        'reminder_history_query_notes': ''.join(item['content'] for item in built[1+len(examples):]),
        'system_separator': '\nCapabilities:\n',
    }
    total = sum(len(value.encode()) for value in values.values())
    result = {key: {'utf8_bytes': len(value.encode()),
                    'approx_tokens_utf8_div_4': len(value.encode()) / 4,
                    'content_byte_percent': len(value.encode()) * 100 / total}
              for key, value in values.items()}
    schema = compact(service.engine.schema.planner_output_schema())
    return {'message_content': result, 'output_schema_bytes_separate': len(schema.encode()),
            'wire_messages_bytes': len(compact(built).encode()),
            'example_pairs': len(examples)//2,
            'content_sha256': hashlib.sha256(compact(built).encode()).hexdigest()}


def exclusive(events):
    """Subtract union of contained child intervals, so nested stages don't double-count."""
    totals = defaultdict(float)
    for event in events:
        start, end = event['start_ms'], event['start_ms'] + event['duration_ms']
        intervals = sorted((other['start_ms'], other['start_ms'] + other['duration_ms'])
                           for other in events if other is not event and
                           start <= other['start_ms'] and
                           other['start_ms'] + other['duration_ms'] <= end)
        covered, cursor = 0.0, start
        for a, b in intervals:
            covered += max(0, b-max(cursor, a))
            cursor = max(cursor, b)
        totals[event['stage']] += max(0, end-start-covered)
    return dict(totals)


def summarize_report(report):
    summary = {key:value for key,value in report.items() if key != 'rows'}
    rows = report['rows']
    calls = [event for row in rows for event in row['trace'] if event['stage'].startswith('llm.')]
    summary['model_metrics'] = {}
    for key in ('input_tokens','output_tokens','ttft_ms','prefill_ms','generation_ms','load_ms','duration_ms'):
        values = [event[key] for event in calls if event.get(key) is not None]
        summary['model_metrics'][key] = {
            'observed_calls':len(values), 'total':sum(values) if values else None,
            'mean_per_call':sum(values)/len(values) if values else None,
            'mean_per_request':sum(values)/len(rows) if values else None,
        }
    summary['followups'] = [{key:row[key] for key in ('case_id','total_ms','llm_calls','structured_correct')}
                           for row in rows if row['case_id'].startswith('followup-')]
    summary['failures'] = [{key:row[key] for key in ('case_id','semantic_match','structured_correct','status')}
                          for row in rows if not(row['semantic_match'] and row['structured_correct'])]
    return summary


async def run(args):
    paths = [ROOT/'benchmarks'/name for name in (
        'semantic_planner_synthetic.yaml', 'semantic_planner_age_filters.yaml',
        'semantic_planner_date_intervals.yaml')]
    datasets = [load_probe_dataset(path) for path in paths]
    clocks = {id(case):dataset.frozen_time for dataset in datasets for case in dataset.cases}
    cases = [case for dataset in datasets for case in dataset.cases]
    # First paraphrase per plan keeps broad grammar coverage and bounded GPU time.
    cases = list({case.plan_id: case for case in reversed(cases)}.values())[::-1]
    for label, utterance, operation, subject, prop, expected in (
        ('identity', 'Who am I?', 'resolve_reference', {'kind':'self','entity_type':'person'}, None, ('person:a',)),
        ('household_count', 'How many people are in my household?', 'count', {'kind':'current_household','entity_type':'address','path':[{'relation':'member'}]}, None, 8),
        ('youngest', 'Who is the youngest?', 'argmax', {'kind':'current_household','entity_type':'address','path':[{'relation':'member'}]}, 'birth_date', ('person:grandson',)),
    ):
        case = SemanticEvalCase(utterance, 'person:a', 'simple_or_derived', label,
            SemanticFactRequest.model_validate({'operation':operation,'subject':subject,'property':prop}),
            expected_status='found', expected_entity_ids=expected if isinstance(expected,tuple) else None,
            expected_value=expected if isinstance(expected,int) else None)
        cases.append(case)
        clocks[id(case)] = datasets[0].frozen_time
    replay = ReplayClient() if args.mode == 'replay' else None
    model = OllamaService(args.ollama_url, args.model, client=replay)
    service, context = build_json_fact_service(args.data_dir, ROOT/'schemas/edge', model)
    context = replace(context, caller_entity_id=datasets[0].default_speaker_id,
                      household_id=datasets[0].household_id,
                      current_time=datasets[0].frozen_time, locale='en')
    conversations = SemanticConversationService(service)
    rows = []
    prompt_sizes = []
    try:
        for iteration in range(args.repeat + 1):
            for case in cases:
                messages = [{'role': 'user', 'content': case.utterance}]
                if replay:
                    replay.payload = {'requires_fact': True, 'request': case.expected.model_dump(mode='json')}
                ctx = replace(context, caller_entity_id=case.speaker_id, current_time=clocks[id(case)])
                with trace_request() as trace:
                    started = perf_counter()
                    answer = await conversations.try_answer(messages, context=ctx)
                    elapsed = (perf_counter()-started)*1000
                if iteration == 0:
                    if not prompt_sizes:
                        prompt_sizes.append(components(service, messages, ctx.current_time.isoformat()))
                    continue
                calls = [e for e in trace.events if e['stage'].startswith('llm.')]
                matches = answer is not None and any(
                    normalize_semantic_request(answer.request) == normalize_semantic_request(expected)
                    for expected in (case.expected, *case.acceptable_alternatives))
                rows.append({'case_id':case.case_id, 'category':case.category,
                             'total_ms':elapsed, 'llm_calls':len(calls) if not replay else 0,
                             'replayed_calls':len(calls) if replay else 0,
                             'semantic_match':matches,
                             'structured_correct':score_structured_result(answer.result,case) if answer else False,
                             'status':answer.result.status if answer else None,
                             'ir':answer.request.model_dump(mode='json') if answer else None,
                             'timings':exclusive(trace.events), 'trace':trace.events,
                             'trace_dropped':trace.dropped})
            # Synthetic follow-up: stateless reconstruction versus persisted focus.
            for persistent in (False, True):
                ctx = replace(context, conversation_id=f'audit-{iteration}' if persistent else None)
                base = {'operation':'select','subject':{'kind':'named_entity','value':'daughter','entity_type':'person'},'property':'birth_date'}
                follow = {'operation':'date_difference','mode':'years','subject':{'kind':'discourse','entity_type':'person','turn_offset':1,'cardinality':'single'},'property':'birth_date'}
                history = [{'role':'user','content':"When is daughter's birthday?"}]
                if replay:
                    replay.payload = {'requires_fact':True,'request':base}
                await conversations.try_answer(history, context=ctx)
                history.append({'role':'user','content':'How old is she?'})
                if replay:
                    # Replay must select the prior-turn IR when the coordinator re-grounds it.
                    async def replay_chat(**kwargs):
                        json.loads(compact(kwargs))
                        users = [m['content'] for m in kwargs['messages'] if m['role']=='user']
                        payload = follow if users[-1]=='How old is she?' else base
                        return ChatResponse(message={'role':'assistant','content':compact({'requires_fact':True,'request':payload})})
                    original = replay.chat
                    replay.chat = replay_chat
                with trace_request() as trace:
                    started = perf_counter()
                    answer = await conversations.try_answer(history, context=ctx)
                    elapsed = (perf_counter()-started)*1000
                if replay:
                    replay.chat = original
                if iteration:
                    calls = [e for e in trace.events if e['stage'].startswith('llm.')]
                    rows.append({'case_id':'followup-persistent' if persistent else 'followup-stateless',
                                 'total_ms':elapsed,'llm_calls':len(calls) if not replay else 0,
                                 'replayed_calls':len(calls) if replay else 0,
                                 'semantic_match':answer is not None and answer.request.subject.kind=='discourse',
                                 'structured_correct':answer is not None and answer.result.status=='found' and answer.result.value==16,
                                 'status':answer.result.status if answer else None,
                                 'ir':answer.request.model_dump(mode='json') if answer else None,
                                 'timings':exclusive(trace.events),'trace':trace.events,'trace_dropped':trace.dropped})
    finally:
        await model.close()
    metrics = defaultdict(list)
    for row in rows:
        for key, value in row['timings'].items():
            metrics[key].append(value)
    report = {'mode':args.mode,'scope':'semantic conversation pipeline; excludes HTTP/auth and production SurrealDB',
              'sample_count':len(rows),'repeat':args.repeat,'warmup':'one full suite pass excluded',
              'latency_ms':summarize_latencies([r['total_ms'] for r in rows]),
              'mean_exclusive_ms':{k:sum(v)/len(rows) for k,v in sorted(metrics.items(),key=lambda item:-sum(item[1]))},
              'semantic_matches':sum(r['semantic_match'] for r in rows),
              'structured_correct':sum(r['structured_correct'] is True for r in rows),
              'structured_unscored':sum(r['structured_correct'] is None for r in rows),
              'llm_calls':sum(r['llm_calls'] for r in rows), 'prompt_components':prompt_sizes,
              'provenance':collect_provenance(root=ROOT,eval_path=paths[0],ollama_url=args.ollama_url,
                 ollama_model=args.model,backend='json',frozen_time=context.current_time,warmup=1,
                 repeat=args.repeat,verified_cold=False,data_dir=args.data_dir,schema_dir=ROOT/'schemas/edge'),
              'python_hash_seed':os.environ.get('PYTHONHASHSEED'),
              'frozen_clocks':{str(d.path.name):d.frozen_time.isoformat() for d in datasets},
              'ontology_sha256':hashlib.sha256((ROOT/'schemas/semantic/ontology.yaml').read_bytes()).hexdigest(),
              'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'datasets_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    summary = summarize_report(report)
    args.output.with_name(args.output.stem+'-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['replay','ollama'],default='replay')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'benchmarks/fixtures/semantic-contract')
    parser.add_argument('--ollama-url',default='http://ollama:11434')
    parser.add_argument('--model',default='qwen3.5:9b')
    parser.add_argument('--repeat',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.mode == 'ollama' and os.environ.get('PYTHONHASHSEED') is None:
        parser.error('Set PYTHONHASHSEED to the same value for before/after runs')
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    asyncio.run(run(args))
