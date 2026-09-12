#!/usr/bin/env python3
"""Supplemental full ASGI/SurrealDB audit against a disposable synthetic namespace.

Uses the container's DB credentials without exporting them. Never selects the
configured production namespace. Removes only its own UUID namespace in finally.
"""
import argparse
import asyncio
import json
from pathlib import Path
from scripts import PROJECT_ROOT
from time import perf_counter
from types import SimpleNamespace
from uuid import uuid4

import httpx
from surrealdb import RecordID
from home_cortex.agent_service import AgentService
from home_cortex.api import app
from home_cortex.config import Settings
from home_cortex.db import Database
from home_cortex.ollama import OllamaService
from home_cortex.request_tracing import trace_request
from home_cortex.retrieval import RetrievalService
from home_cortex.tools import ToolDispatcher, get_tool_definitions
from scripts.benchmarks.semantic_planner_benchmark import build_json_fact_service, load_probe_dataset, summarize_latencies
from scripts.profiling.token_latency_audit import exclusive

ROOT=PROJECT_ROOT


def rid(value):
    return RecordID(*value.split(':',1))


async def run(args):
    fixture=ROOT/'benchmarks/fixtures/semantic-contract'
    namespace='hc_latency_audit_'+uuid4().hex
    settings=Settings().model_copy(update={'surreal_namespace':namespace,'surreal_database':'fixture'})
    database=Database(settings)
    model=OllamaService('http://ollama:11434','qwen3.5:9b')
    await database.connect()
    try:
        for path in sorted((fixture/'nodes').glob('*.json')):
            for row in json.loads(path.read_text()):
                await database.upsert(rid(row['id']),{k:v for k,v in row.items() if k!='id'})
        for path in sorted((fixture/'edges').glob('*.json')):
            for index,row in enumerate(json.loads(path.read_text())):
                await database.query('RELATE $source->$edge->$target CONTENT $content;',{
                    'source':rid(row['from']),'target':rid(row['to']),
                    'edge':RecordID(path.stem,str(index)),
                    'content':{k:v for k,v in row.items() if k not in ('from','to','id')}})
        service,_=build_json_fact_service(fixture,ROOT/'schemas/edge',model)
        schema=service.engine.schema
        retrieval=RetrievalService(database,data_dir=fixture,edge_registry=schema.edge_registry)
        dispatcher=ToolDispatcher(retrieval,("calculate",))
        agent=AgentService(model,dispatcher,system_prompt='Household steward',tools=get_tool_definitions(('calculate',)),
                           schema_catalog=schema.catalog,home_entity_id='address:fictional')
        app.state.agent=agent
        app.state.agents={'steward':agent}
        app.state.retrieval=retrieval
        cases=[]
        for file, indices in (('synthetic',(0,2,3,4,10,12)),('age_filters',(2,6,12,15))):
            dataset=load_probe_dataset(ROOT/f'benchmarks/semantic_planner_{file}.yaml')
            for index in indices:
                cases.append((dataset.cases[index],dataset.frozen_time))
        rows=[]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://isolated') as client:
            for iteration in range(2):
                for case,now in cases:
                    agent._clock=lambda now=now:now
                    app.state.settings=SimpleNamespace(cortex_api_key=None,cortex_identity_map={'id:audit':case.speaker_id})
                    with trace_request() as trace:
                        started=perf_counter()
                        response=await client.post('/v1/chat',json={'message':case.utterance},
                                                   headers={'X-OpenWebUI-User-Id':'audit'})
                        response.raise_for_status()
                        elapsed=(perf_counter()-started)*1000
                    if iteration:
                        rows.append({'case_id':case.case_id,'total_ms':elapsed,'timings':exclusive(trace.events),
                                     'llm_calls':sum(e['stage'].startswith('llm.') for e in trace.events),
                                     'db_queries':sum(e['stage']=='surreal.query' for e in trace.events),
                                     'response_sha256':__import__('hashlib').sha256(response.json()['answer'].encode()).hexdigest(),
                                     'trace':trace.events})
        totals={key:sum(row['timings'].get(key,0) for row in rows)/len(rows)
                for key in {k for row in rows for k in row['timings']}}
        output={'scope':'Full ASGI request/response with real SurrealDB on a temporary synthetic namespace; excludes browser, TCP ingress and proxy',
                'sample_count':len(rows),'latency_ms':summarize_latencies([r['total_ms'] for r in rows]),
                'mean_exclusive_ms':totals,'llm_calls':sum(r['llm_calls'] for r in rows),
                'db_queries':sum(r['db_queries'] for r in rows),'rows':rows}
    finally:
        try:
            # The identifier is generated above, never read from settings/user input.
            await database.query('REMOVE NAMESPACE '+namespace+';')
        finally:
            await model.close()
            await database.close()
    output["temporary_namespace_removed"] = True
    args.output.write_text(json.dumps(output,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    asyncio.run(run(parser.parse_args()))
