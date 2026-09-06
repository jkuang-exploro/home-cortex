import json,sys,runpy
import home_cortex.ollama as ollama
from home_cortex.semantic_planner_benchmark import build_json_fact_service
from pathlib import Path
s,_=build_json_fact_service(Path('data'),Path('schemas/edge'),None)
original=ollama.OllamaService._planner_system_prompt
schema=s.engine.schema.planner_output_schema()
ollama.OllamaService._planner_system_prompt=lambda self,capabilities: original(self,capabilities)+'\nJSON output schema:\n'+json.dumps(schema,ensure_ascii=False,separators=(',',':'))
ollama._semantic_planner_examples=lambda:[]
original_plan=ollama.OllamaService.plan_semantic_fact
# Set context on this isolated client's HTTP call only; preserve the model digest.
original_init=ollama.OllamaService.__init__
def init(self,*args,**kwargs):
 original_init(self,*args,**kwargs)
 chat=self.client.chat
 async def with_context(**request):
  request['options']={**request.get('options',{}),'num_ctx':8192}
  return await chat(**request)
 self.client.chat=with_context
ollama.OllamaService.__init__=init
sys.argv=['scripts/tier1_latency_bench.py','--ollama-url','http://ollama:11434','--model','qwen3.5:9b','--data-dir','/tmp/qwen35-semantic-contract/data','--schema-dir','/tmp/qwen35-semantic-contract/schemas/edge','--eval','/tmp/qwen35-semantic-contract/benchmarks/semantic_planner_eval.yaml','--warmup','1','--repeat','1','--progress','--output','/tmp/qwen35-semantic-contract/probe-schema-grounded.json']
runpy.run_path('scripts/tier1_latency_bench.py',run_name='__main__')
