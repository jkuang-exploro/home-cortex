"""Account for the exact planner message content; optional native Ollama counts.

Raw component counts exclude chat framing and are not additive. The format
schema constrains decoding separately and must not be counted as message text.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from ollama import AsyncClient
from home_cortex import ollama as prompts
from home_cortex.mutation_ir import read_plan_schema
from scripts.benchmarks.semantic_planner_benchmark import build_json_fact_service
from scripts import PROJECT_ROOT


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def prompt_components(messages, capabilities, *, household_now):
    built = prompts.planner_chat_messages(messages, capabilities, household_now=household_now)
    examples = prompts._semantic_planner_examples()
    tail = built[2 + len(examples):]
    latest = max((i for i, m in enumerate(tail) if m['role'] == 'user'), default=-1)
    parts = {
        'instructions': prompts._PLANNER_INSTRUCTIONS,
        'capabilities': built[0]['content'][len(prompts._PLANNER_INSTRUCTIONS):],
        'few_shot_examples': ''.join(m['content'] for m in examples),
        'context_reminder': built[1 + len(examples)]['content'],
        'history': ''.join(m['content'] for i, m in enumerate(tail) if i < latest),
        'user_input': tail[latest]['content'] if latest >= 0 else '',
        'supplemental_notes_including_retry': ''.join(m['content'] for i, m in enumerate(tail) if i > latest),
    }
    assert sum(len(v.encode()) for v in parts.values()) == sum(len(m['content'].encode()) for m in built)
    return parts, built


def example_audit():
    examples = prompts._semantic_planner_examples()
    rows, seen, shapes = [], {}, {}
    for i in range(0, len(examples), 2):
        plan = json.loads(examples[i + 1]['content'])
        signature = compact(plan)
        shape = json.loads(signature)
        def hide_literal_names(value):
            if isinstance(value, dict):
                if value.get('kind') == 'named_entity':
                    value['value'] = '<literal name>'
                for child in value.values():
                    hide_literal_names(child)
            elif isinstance(value, list):
                for child in value:
                    hide_literal_names(child)
        hide_literal_names(shape)
        shape_signature = compact(shape)
        request = plan.get('request') or {}
        row = {'index': i // 2, 'utterance': examples[i]['content'],
               'operation': request.get('operation', 'chat'),
               'bytes': sum(len(m['content'].encode()) for m in examples[i:i + 2]),
               'same_plan_as': seen.get(signature),
               'same_shape_ignoring_literal_names_as': shapes.get(shape_signature)}
        seen.setdefault(signature, i // 2)
        shapes.setdefault(shape_signature, i // 2)
        rows.append(row)
    return rows


async def run(args):
    service, context = build_json_fact_service(args.data_dir, PROJECT_ROOT / 'schemas/edge', None)
    schema = service.engine.schema
    messages = [{'role': 'user', 'content': text} for text in [*args.history, args.query]]
    if args.retry_hint:
        messages.append({'role': 'system', 'content': args.retry_hint})
    parts, built = prompt_components(messages, schema.planner_capability_payload(), household_now=context.current_time.isoformat())
    total = sum(len(v.encode()) for v in parts.values())
    rows = {key: {'bytes': len(value.encode()), 'content_byte_percent': round(100 * len(value.encode()) / total, 2),
                  'raw_tokens': None, 'sha256': hashlib.sha256(value.encode()).hexdigest()}
            for key, value in parts.items()}
    if args.ollama_url:
        client = AsyncClient(host=args.ollama_url)
        try:
            for key, value in parts.items():
                if not value:
                    rows[key]['raw_tokens'] = 0
                    continue
                response = await client.generate(model=args.model, prompt=value, raw=True, think=False,
                    keep_alive=prompts.OLLAMA_KEEP_ALIVE,
                    options={'num_ctx': prompts.OLLAMA_NUM_CTX, 'num_predict': 1, 'temperature': 0, 'seed': 0})
                rows[key]['raw_tokens'] = response.prompt_eval_count
        finally:
            await client.close()
    report = {'method': 'Exact UTF-8 message content; optional model-native raw generate counts exclude chat framing and are not additive.',
              'model': args.model, 'num_ctx': prompts.OLLAMA_NUM_CTX, 'output_budget': prompts.PLANNER_NUM_PREDICT,
              'content_bytes': total, 'wire_messages_bytes': len(compact(built).encode()),
              'messages_sha256': hashlib.sha256(compact(built).encode()).hexdigest(),
              'schema_format_bytes_separate': len(compact(read_plan_schema(schema.planner_output_schema())).encode()),
              'components': rows, 'examples': example_audit(), 'example_pairs': len(prompts._semantic_planner_examples()) // 2}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'examples'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'benchmarks/fixtures/semantic-contract')
    parser.add_argument('--query', default='家里有几个人')
    parser.add_argument('--history', action='append', default=[])
    parser.add_argument('--retry-hint', default='')
    parser.add_argument('--ollama-url')
    parser.add_argument('--model', default='qwen3.5:9b')
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
