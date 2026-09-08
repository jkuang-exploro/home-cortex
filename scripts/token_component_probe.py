#!/usr/bin/env python3
"""Measure isolated prompt components with the deployed model's own tokenizer.

These are raw-prompt eval counts, not additive chat-template token accounting.
One generated token per component is discarded. Run separately from latency tests.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from ollama import AsyncClient
from home_cortex.ollama import _PLANNER_INSTRUCTIONS, _semantic_planner_examples
from home_cortex.semantic_planner_benchmark import build_json_fact_service

ROOT = Path(__file__).resolve().parents[1]


def compact(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':'))


async def run(args):
    service, _ = build_json_fact_service(ROOT/'benchmarks/fixtures/semantic-contract', ROOT/'schemas/edge', None)
    parts = {
        'instructions':_PLANNER_INSTRUCTIONS,
        'capabilities':compact(service.engine.schema.planner_capability_payload()),
        'examples':''.join(m['content'] for m in _semantic_planner_examples()),
        'output_schema_separate':compact(service.engine.schema.planner_output_schema()),
    }
    client=AsyncClient(host=args.ollama_url)
    rows=[]
    try:
        for name, text in parts.items():
            response=await client.generate(model=args.model,prompt=text,raw=True,
                stream=False,think=False,keep_alive='24h',
                options={'num_predict':1,'num_ctx':8192,'temperature':0,'seed':0})
            rows.append({'component':name,'utf8_bytes':len(text.encode()),
                         'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                         'raw_prompt_tokens':response.prompt_eval_count,
                         'generated_tokens':response.eval_count})
    finally:
        await client.close()
    args.output.write_text(json.dumps({'method':'Ollama raw generate; model-native prompt_eval_count; separate probes, excludes chat framing',
                                      'model':args.model,'components':rows},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ollama-url',default='http://ollama:11434')
    p.add_argument('--model',default='qwen3.5:9b')
    p.add_argument('--output',type=Path,required=True)
    asyncio.run(run(p.parse_args()))
