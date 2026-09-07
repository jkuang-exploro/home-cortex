"""Grammar and execution regressions use invented households, never an LLM oracle."""
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.fact_benchmark import _JsonGraphDispatcher
from home_cortex.ollama import _semantic_planner_examples
from home_cortex.schema_catalog import RuntimeSchemaCatalog
from home_cortex.semantic_facts import (
    AgentRequestContext, HouseholdFactEngine, FactRenderer, SemanticFactPlanner,
    SemanticFactRequest, SemanticReference, SemanticRelationStep, SemanticFilter,
    SemanticSchemaRegistry, SemanticPlannerFailure,
)
from home_cortex.semantic_planner_benchmark import load_semantic_eval_cases, normalize_semantic_request


@pytest.fixture
def household(tmp_path):
    (tmp_path / 'nodes').mkdir()
    (tmp_path / 'edges').mkdir()
    people = [
        ('a', 'male', '1980-03-02'), ('b', 'female', '1982-06-01'),
        ('son1', 'male', '2008-09-03'), ('son2', 'male', '2012-10-15'),
        ('daughter', 'female', '2010-04-09'), ('father', 'male', '1955-01-01'),
        ('mother', 'female', '1957-02-02'), ('grandson', 'male', '2025-01-01'),
    ]
    def write(kind, name, rows):
        (tmp_path / kind / (name + '.json')).write_text(json.dumps(rows))
    write('nodes', 'person', [dict(id='person:'+i, name=i, gender=g, dob=d) for i,g,d in people])
    write('nodes', 'address', [dict(id='address:fictional', full_address='Invented home')])
    write('edges', 'spouse_of', [{'from':'person:a','to':'person:b','start':'2005-05-06','end':None}])
    write('edges', 'parent_of', [
        {'from':'person:'+p,'to':'person:'+c}
        for p,c in [('a','son1'),('a','son2'),('a','daughter'),('b','son1'),('b','son2'),('b','daughter'),('father','b'),('mother','b'),('daughter','grandson')]
    ])
    write('edges', 'lives_in', [{'from':'person:'+i,'to':'address:fictional','start':'2018-01-01','end':None} for i,_,_ in people])
    registry = EdgeSchemaRegistry.load_default()
    schema = SemanticSchemaRegistry(RuntimeSchemaCatalog.from_data_dir(tmp_path, registry))
    dispatcher = _JsonGraphDispatcher(tmp_path, registry)
    engine = HouseholdFactEngine(dispatcher, schema)
    context = AgentRequestContext('person:a','assistant','Helper','address:fictional',datetime.fromisoformat('2026-09-03T12:00:00-07:00'),'en')
    return engine, context, dispatcher


def ref(*steps):
    return SemanticReference(kind='self', entity_type='person', path=tuple(SemanticRelationStep.model_validate(s) for s in steps))


@pytest.mark.asyncio
@pytest.mark.parametrize('day,expected', [('2026-05-31',43), ('2026-06-01',44), ('2026-06-02',44)])
async def test_age_is_completed_years_not_birth_date(household, day, expected):
    engine,ctx,_=household
    ctx=replace(ctx,current_time=datetime.fromisoformat(day+'T12:00:00-07:00'),locale='zh')
    age=SemanticFactRequest(operation='completed_years',subject=ref(step('spouse','female')),property='birth_date')
    result,*_=await engine.execute(age,ctx)
    assert result.status=='found' and result.value==expected
    assert f'{expected}岁' in FactRenderer().render(age,result,ctx)
    birth=age.model_copy(update={'operation':'select'})
    result,*_=await engine.execute(birth,ctx)
    assert result.value=='1982-06-01'


@pytest.mark.asyncio
async def test_date_filtered_set_preserves_bounds_scope_and_cardinality(household):
    engine,ctx,dispatcher=household
    dispatcher.entities['person:a']['dob']='1988-01-01'
    dispatcher.entities['person:b']['dob']='1988-12-31'
    dispatcher.entities['person:son1']['dob']='1989-01-01'
    dispatcher.entities['person:son2']['dob']='1987-12-31'
    dispatcher.entities['person:outsider']={'id':'person:outsider','name':'Outside','dob':'1988-06-01','gender':'male'}
    members=SemanticReference(kind='current_household',entity_type='address',path=(SemanticRelationStep(relation='member'),))
    condition=SemanticFilter(property='birth_date',operator='date_range',value=('1988-01-01','1989-01-01'))
    query=SemanticFactRequest(operation='select',subject=members,filters=(condition,))
    result,*_=await engine.execute(query,ctx)
    assert result.status=='found'
    assert {item['id'] for item in result.value}=={'person:a','person:b'}
    count,*_=await engine.execute(query.model_copy(update={'operation':'count'}),ctx)
    assert count.value==2
    empty=query.model_copy(update={'filters':(condition.model_copy(update={'value':('1900-01-01','1901-01-01')}),)})
    result,*_=await engine.execute(empty,ctx)
    assert result.status=='found' and result.value==[]
    assert FactRenderer().render(empty,result,replace(ctx,locale='zh'))=='没有找到符合筛选条件的记录。'
    assert FactRenderer().render(empty,result,replace(ctx,locale='en'))=='No records match the filters.'
    dispatcher.entities['person:b'].pop('dob')
    result,*_=await engine.execute(query,ctx)
    assert result.status=='filter_input_missing'


@pytest.mark.asyncio
async def test_age_and_date_filter_eval_expectations_on_invented_household(household):
    from home_cortex.semantic_planner_benchmark import load_probe_dataset, score_structured_result
    engine,ctx,_=household
    dataset=load_probe_dataset(Path(__file__).parents[1]/'benchmarks/semantic_planner_age_filters.yaml')
    ctx=replace(ctx,current_time=dataset.frozen_time)
    example_texts={message['content'] for message in _semantic_planner_examples() if message['role']=='user'}
    for case in dataset.cases:
        assert case.utterance not in example_texts
        result,*_=await engine.execute(case.expected,replace(ctx,caller_entity_id=case.speaker_id))
        assert score_structured_result(result,case), case.case_id


@pytest.mark.asyncio
@pytest.mark.parametrize('value_property', ['birth_date', 'year_of_birth'])
async def test_collection_filter_cannot_silently_ignore_dynamic_operand(household,value_property):
    engine,ctx,_=household
    query=SemanticFactRequest(
        operation='select',
        subject=SemanticReference(kind='current_household',entity_type='address',path=(SemanticRelationStep(relation='member'),)),
        filters=(SemanticFilter(property='birth_date',value_from='anchor',value_property=value_property),),
    )
    assert engine.schema.validation_code(query)=='INVALID_PLAN'
    result,queries,*_=await engine.execute(query,ctx)
    assert result.status=='semantic_plan_unsupported' and queries==0


def step(relation, gender=None):
    return {'relation':relation, **({'filters':[{'property':'gender','value':gender}]} if gender else {})}


@pytest.mark.asyncio
@pytest.mark.parametrize('speaker', ['person:a', 'person:b'])
@pytest.mark.parametrize('relation,expected', [('spouse','2005-05-06'),('residence','2018-01-01')])
async def test_relationship_ownership_across_relations_and_speakers(household,speaker,relation,expected):
    engine,ctx,_=household
    ctx=replace(ctx,caller_entity_id=speaker)
    request=SemanticFactRequest(operation='select',subject=ref(step(relation)),property='start_date',property_source='relationship')
    result,*_=await engine.execute(request,ctx)
    assert result.status=='found' and result.value==expected
    duration=request.model_copy(update={'operation':'duration','mode':'days'})
    result,*_=await engine.execute(duration,ctx)
    assert result.value==(ctx.current_time.date()-datetime.fromisoformat(expected).date()).days
    assert not engine.schema.validates(request.model_copy(update={'property_source':'entity'}))
    if relation=='spouse':
        birthday=request.model_copy(update={'property':'birth_date','property_source':'entity'})
        result,*_=await engine.execute(birthday,ctx)
        assert result.status=='found'
        assert not engine.schema.validates(birthday.model_copy(update={'property_source':'relationship'}))


@pytest.mark.asyncio
@pytest.mark.parametrize('operation,winner', [('argmin','person:a'),('argmax','person:b')])
@pytest.mark.parametrize('reverse', [False,True])
@pytest.mark.parametrize('equal',[False,True])
async def test_pairwise_both_operands_order_and_ties(household,operation,winner,reverse,equal):
    engine,ctx,dispatcher=household
    if equal: dispatcher.entities['person:b']['dob']=dispatcher.entities['person:a']['dob']
    left,right=ref(),ref(step('spouse'))
    if reverse: left,right=right,left
    request=SemanticFactRequest(operation=operation,subject=left,other=right,property='birth_date')
    result,*_=await engine.execute(request,ctx)
    assert result.status=='found'
    assert result.value['equal'] is equal
    assert {result.value[k]['id'] for k in ['selected','other']}=={'person:a','person:b'}
    if not equal: assert result.value['selected']['id']==winner
    else: assert 'same age' in FactRenderer().render(request,result,ctx)
    assert not engine.schema.validates(request.model_copy(update={'property':None}))


@pytest.mark.asyncio
async def test_composition_preserves_intermediate_filters_and_ambiguity(household):
    engine,ctx,_=household
    for gender,expected in [('male','person:father'),('female','person:mother')]:
        request=SemanticFactRequest(operation='resolve_reference',subject=ref(step('spouse','female'),step('parent',gender)))
        result,*_=await engine.execute(request,ctx)
        assert result.evidence.entity_ids==(expected,)
    for speaker in ['person:a','person:b']:
        ctx=replace(ctx,caller_entity_id=speaker)
        son=SemanticFactRequest(operation='annual_occurrence',subject=ref(step('child','male')),property='birth_date',mode='days')
        result,*_=await engine.execute(son,ctx)
        assert result.status=='ambiguous'
        daughter=son.model_copy(update={'subject':ref(step('child','female'))})
        result,*_=await engine.execute(daughter,ctx)
        assert result.status=='found' and result.value==218


@pytest.mark.asyncio
async def test_final_collection_filter_equivalence_and_intermediate_counterexample(household):
    engine,ctx,_=household
    request=SemanticFactRequest(operation='select',subject=ref(step('child')),filters=(SemanticFilter(property='gender',value='male'),))
    alternative=request.model_copy(update={'subject':ref(step('child','male')),'filters':()})
    for candidate in [request,alternative]:
        result,*_=await engine.execute(candidate,ctx)
        assert result.status=='found'
        assert {p['id'] for p in result.value}=={'person:son1','person:son2'}
    # Male child of any child includes daughter's son; a son's child does not.
    before=request.model_copy(update={'subject':ref(step('child'),step('child'))})
    after=alternative.model_copy(update={'subject':ref(step('child','male'),step('child'))})
    first,*_=await engine.execute(before,ctx)
    second,*_=await engine.execute(after,ctx)
    assert [p['id'] for p in first.value]==['person:grandson']
    assert second.value==[]
    assert normalize_semantic_request(before)!=normalize_semantic_request(after)


@pytest.mark.asyncio
async def test_filter_movement_is_not_general_equivalence(household):
    engine,ctx,dispatcher=household
    # Scalar resolution happens before request filtering, but after path filtering.
    scalar=SemanticFactRequest(operation='select',subject=ref(step('child')),property='birth_date',filters=(SemanticFilter(property='gender',value='female'),))
    path=scalar.model_copy(update={'subject':ref(step('child','female')),'filters':()})
    first,*_=await engine.execute(scalar,ctx)
    second,*_=await engine.execute(path,ctx)
    assert first.status=='ambiguous' and second.status=='found'
    # Missing field handling also differs: never canonicalize all filters globally.
    dispatcher.entities['person:daughter'].pop('gender')
    scalar=scalar.model_copy(update={'property':None})
    path=path.model_copy(update={'property':None})
    first,*_=await engine.execute(scalar,ctx)
    second,*_=await engine.execute(path,ctx)
    assert first.status=='filter_input_missing' and second.status=='found'


@pytest.mark.asyncio
async def test_declared_predicates_and_invalid_filters_are_not_repaired(household):
    engine,ctx,_=household
    members=SemanticReference(kind='current_household',entity_type='address',path=(SemanticRelationStep(relation='member'),))
    adult=SemanticFactRequest(operation='count',subject=members,filters=(SemanticFilter(predicate='adult'),))
    result,*_=await engine.execute(adult,ctx)
    assert result.status=='found' and result.value==5  # includes eighteenth birthday today
    minor=adult.model_copy(update={'filters':(SemanticFilter(predicate='minor'),)})
    result,*_=await engine.execute(minor,ctx)
    assert result.value==3
    invalids=[
        adult.model_copy(update={'filters':(*adult.filters,SemanticFilter(predicate='completed_years'))}),
        adult.model_copy(update={'filters':(SemanticFilter(property='adult'),)}),
        adult.model_copy(update={'subject':members.model_copy(update={'path':(SemanticRelationStep(relation='child'),)})}),
        adult.model_copy(update={'subject':members.model_copy(update={'entity_type':'person'})}),
    ]
    class Interpreter:
        async def plan_semantic_fact(self,*args,**kwargs):
            return {'requires_fact':True,'request':self.request.model_dump(mode='json')}
    interpreter=Interpreter()
    for request in invalids:
        interpreter.request=request
        with pytest.raises(SemanticPlannerFailure):
            await SemanticFactPlanner(interpreter,engine.schema).plan([],ctx)


def test_model_contract_and_examples_are_semantic_and_separate(household):
    engine,_,_=household
    contract=engine.schema.planner_output_schema()['$defs']
    assert 'property_source' in contract['SemanticFactRequest']['required']
    assert contract['SemanticFilter']['anyOf'][1]['properties']['predicate']['enum']==['adult','minor']
    payload=engine.schema.planner_capability_payload()
    assert payload['collection_predicates']['adult']['definition_only']=={'property':'birth_date','transform':'completed_years','operator':'gte','value':18}
    assert set(payload['operations']).isdisjoint({'filter','traverse','adult','minor'})
    assert 'dob' not in json.dumps(payload)
    assert 'person:a' not in json.dumps(payload)
    utterances={c.utterance for c in load_semantic_eval_cases()}
    for message in _semantic_planner_examples():
        if message['role']=='user': assert message['content'] not in utterances
        else:
            request=SemanticFactRequest.model_validate(engine.schema.expand_planner_concepts(json.loads(message['content']))['request'])
            assert engine.schema.validates(request)


@pytest.mark.asyncio
async def test_explicit_concept_expansion_is_compositional_and_preserves_raw_output(household):
    engine,ctx,_=household
    raw={'requires_fact':True,'request':{'operation':'resolve_reference','subject':{'kind':'self','entity_type':'person','path':[{'concept':'wife'},{'concept':'father'}]}}}
    class Interpreter:
        async def plan_semantic_fact(self,*args,**kwargs): return raw
    outcome=await SemanticFactPlanner(Interpreter(),engine.schema).plan([],ctx)
    request=outcome.plan.request
    assert request.subject==ref(step('spouse','female'),step('parent','male'))
    assert outcome.diagnostics.output_raw==raw
    assert raw['request']['subject']['path']==[{'concept':'wife'},{'concept':'father'}]
    result,*_=await engine.execute(request,ctx)
    assert result.evidence.entity_ids==('person:father',)
    # Every declared concept expands verbatim; intermediate filters are retained.
    for name,definition in engine.schema.ontology.planner_payload()['reference_concepts'].items():
        payload={'request':{'subject':{'kind':'self','path':[{'concept':name}]}}}
        assert engine.schema.expand_planner_concepts(payload)['request']['subject']['path']==definition['path']


def test_concepts_cannot_override_meaning_and_base_paths_are_not_repaired(household):
    engine,_,_=household
    for use in [{'concept':'invented'}, {'concept':'son','filters':[{'predicate':'invented','property':'gender'}]}, {'concept':'son','relation':'child'}]:
        with pytest.raises(ValueError):
            engine.schema.expand_planner_concepts({'request':{'subject':{'kind':'self','path':[use]}}})
    raw={'request':{'subject':{'kind':'self','path':[{'relation':'child'}]}}}
    assert engine.schema.expand_planner_concepts(raw)==raw


def test_heldout_phrasing_is_not_used_by_interpreter():
    from home_cortex.semantic_planner_benchmark import load_probe_dataset
    from home_cortex.ollama import _PLANNER_INSTRUCTIONS
    root=Path(__file__).parents[1]/'benchmarks'
    resources=_PLANNER_INSTRUCTIONS+json.dumps(_semantic_planner_examples(),ensure_ascii=False)
    known={case.utterance for case in load_semantic_eval_cases()}
    for name in ['semantic_planner_heldout.yaml','semantic_planner_synthetic.yaml']:
        for case in load_probe_dataset(root/name).cases:
            assert case.utterance not in resources
            assert case.utterance not in known


def test_extra_concept_filters_only_narrow_declared_meaning(household):
    engine,_,_=household
    raw={'request':{'subject':{'kind':'self','path':[{'concept':'son','filters':[{'property':'gender','value':'female'}]}]}}}
    path=engine.schema.expand_planner_concepts(raw)['request']['subject']['path']
    assert [f['value'] for f in path[0]['filters']]==['male','female']
    # Repeated expansions and empty extra filters cannot mutate the ontology.
    plain={'request':{'subject':{'kind':'self','path':[{'concept':'son','filters':[]}]}}}
    assert engine.schema.expand_planner_concepts(plain)['request']['subject']['path']==[step('child','male')]


def test_evaluation_alternative_is_limited_to_final_child_list():
    cases=load_semantic_eval_cases()
    case=next(c for c in cases if c.plan_id=='male_children')
    assert len(case.acceptable_alternatives)==1
    assert case.expected.property is None and case.expected.operation=='select'
    assert case.acceptable_alternatives[0].subject==ref(step('child','male'))
    for c in cases:
        if c.plan_id!='male_children':
            assert case.acceptable_alternatives[0] not in c.acceptable_alternatives


@pytest.mark.parametrize('entity,names,correct',[
    ('person:zhigang_ba',['Zhigang Ba','巴志刚'],True),
    ('person:zhigang_ba',['Zhigang'],True),
    ('person:zhigang_ba',['Unrelated name'],False),
    ('person:someone_else',['Zhigang Ba'],False),
])
def test_wife_father_accepted_name_representation_still_requires_correct_person(entity,names,correct):
    from home_cortex.semantic_planner_benchmark import load_probe_dataset, score_structured_result
    from home_cortex.semantic_facts import FactResult,FactEvidence
    case=next(c for c in load_probe_dataset().cases if c.case_id=='wife_father_given_name')
    result=FactResult('found',names,FactEvidence(entity_ids=(entity,)))
    assert score_structured_result(result,case) is correct


@pytest.mark.asyncio
async def test_relationship_filter_movement_changes_edge_property_selection(household):
    engine,ctx,dispatcher=household
    dispatcher.edges['spouse_of'].append({'from':'person:a','to':'person:b','start':'2007-01-01','end':None})
    condition=SemanticFilter(property='start_date',operator='gte',value='2006-01-01',source='relation')
    request=SemanticFactRequest(operation='select',subject=ref(step('spouse')),property='start_date',property_source='relationship',filters=(condition,))
    path=request.model_copy(update={'subject':ref({'relation':'spouse','filters':[condition.model_dump()]}),'filters':()})
    first,*_=await engine.execute(request,ctx)
    second,*_=await engine.execute(path,ctx)
    assert first.status=='ambiguous'
    assert second.status=='found' and second.value=='2007-01-01'


def test_wire_grammar_requires_ownership_and_runtime_types_context(household):
    engine,_,_=household
    defs=engine.schema.planner_output_schema()['$defs']
    assert 'entity_id' not in defs['SemanticReference']['properties']['kind']['enum']
    request=defs['SemanticFactRequest']
    assert 'property_source' in request['required']
    assert request['properties']['property_source']['enum']==['entity','relationship']
    assert engine.schema.validates(SemanticFactRequest(
        operation='select',subject=ref(),property='start_date',
        property_source='relationship')) is False
    assert engine.schema.validates(SemanticFactRequest(
        operation='select',subject=ref(step('spouse')),property='start_date',
        property_source='entity')) is False
    assert engine.schema.validates(SemanticFactRequest(
        operation='select',subject=ref(step('spouse')),property='birth_date',
        property_source='entity')) is True
    with pytest.raises(ValueError):
        SemanticReference(kind='self',value='some name')


def test_ordering_meaning_is_declarative_and_property_specific(household):
    engine,_,_=household
    props=engine.schema.planner_capability_payload()['property_aliases']
    assert 'oldest' in props['birth_date']['ordering']['minimum']
    assert 'youngest' in props['birth_date']['ordering']['maximum']
    assert isinstance(props['display_name'],list)
    assert engine.schema.planner_capability_payload()['relation_signatures']['residence']=={'person':['address']}
