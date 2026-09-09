import asyncio,json,hashlib
from pathlib import Path
from home_cortex.config import Settings
from home_cortex.ollama import OllamaService,planner_chat_messages,PLANNER_NUM_CTX,PLANNER_NUM_PREDICT,PLANNER_SEED,PLANNER_KEEP_ALIVE,_ollama_runtime_metrics
from home_cortex.semantic_planner_benchmark import build_json_fact_service
from home_cortex.semantic_facts import SemanticFactPlanner,SemanticFactService
class Capture(OllamaService):
 async def _chat(self,**kwargs):
  response=await super()._chat(**kwargs)
  self.captured.append({'raw':response.message.content,'prompt_tokens':response.prompt_eval_count,'output_tokens':response.eval_count,'prompt_ms':(response.prompt_eval_duration or 0)/1e6,'generation_ms':(response.eval_duration or 0)/1e6})
  return response
class Expanded(Capture):
 async def plan_semantic_fact(self,messages,capabilities,output_schema,*,household_now):
  response=await self._chat(model=self.model,messages=planner_chat_messages(messages,capabilities,household_now=household_now),format=dict(output_schema),stream=False,think=False,keep_alive=PLANNER_KEEP_ALIVE,options={'temperature':0,'num_ctx':PLANNER_NUM_CTX,'num_predict':PLANNER_NUM_PREDICT,'seed':PLANNER_SEED})
  self.last_planner_runtime=_ollama_runtime_metrics(response)
  return json.loads(response.message.content)
async def main():
 settings=Settings(); base,ctx=build_json_fact_service(Path('/app/benchmarks/fixtures/semantic-contract'),Path('/app/schemas/edge'),None)
 for mode,klass in [('compact',Capture),('expanded',Expanded)]:
  model=klass(settings.ollama_url,settings.ollama_model);model.captured=[]
  if mode=='compact':
   tags=await model.client.list()
   print(json.dumps({'model_metadata':[{'name':m.model,'digest':m.digest} for m in tags.models if m.model==settings.ollama_model],'ontology_hash':hashlib.sha256(Path('/app/schemas/semantic/ontology.yaml').read_bytes()).hexdigest()}),flush=True)
  service=SemanticFactService(base.engine,planner=SemanticFactPlanner(model,base.engine.schema))
  answer=await service.try_answer([{'role':'user','content':'你是谁'}],context=ctx)
  diag=answer.planner_diagnostics
  print(json.dumps({'mode':mode,'question':'你是谁','answer':answer.text,'status':answer.result.status,'validation':diag.validation_result,'attempts':diag.attempt_count,'calls':model.captured},ensure_ascii=False),flush=True)
  await model.close()
asyncio.run(main())
