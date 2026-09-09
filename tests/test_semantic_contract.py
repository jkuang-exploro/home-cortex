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
    write('nodes', 'item', [{'id':'item:house','name':'House','item_type':'house'}])
    write('nodes', 'space', [{'id':'space:room','name':'Room','space_type':'room'}])
    write('edges', 'located_in', [{'from':'item:house','to':'address:fictional'}])
    write('edges', 'hosted_by', [{'from':'space:room','to':'item:house'}])
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
    age=SemanticFactRequest(operation='date_difference',mode='years',subject=ref(step('spouse','female')),property='birth_date')
    result,*_=await engine.execute(age,ctx)
    assert result.status=='found' and result.value==expected
    assert result.unit=='years'
    assert f'{expected}岁' in FactRenderer().render(age,result,ctx)
    birth=age.model_copy(update={'operation':'select','mode':None})
    result,*_=await engine.execute(birth,ctx)
    assert result.value=='1982-06-01'


@pytest.mark.asyncio
@pytest.mark.parametrize('relation,unit,expected', [('spouse','years',21),('spouse','months',255),('residence','years',8)])
async def test_one_date_interval_contract_for_relationships_and_units(household,relation,unit,expected):
    from home_cortex.semantic_planner_benchmark import serialize_fact_result, fact_result_from_serialized, SemanticEvalCase, score_structured_result
    engine,ctx,_=household
    query=SemanticFactRequest(operation='date_difference',subject=ref(step(relation)),property='start_date',property_source='relationship',mode=unit)
    result,*_=await engine.execute(query,ctx)
    assert result.status=='found' and result.value==expected and result.unit==unit
    zh=FactRenderer().render(query,result,replace(ctx,locale='zh'))
    assert str(expected) in zh and ('个月' if unit=='months' else '年') in zh
    assert '岁' not in zh and '换算' not in zh
    en=FactRenderer().render(query,result,replace(ctx,locale='en'))
    assert f'{expected} {unit}' in en
    restored=fact_result_from_serialized(serialize_fact_result(result))
    assert restored.unit==unit
    case=SemanticEvalCase('synthetic','person:a','temporal','interval',query,expected_value=expected,expected_unit=unit)
    assert score_structured_result(restored,case)
    assert not score_structured_result(replace(restored,unit='seconds'),case)
    assert not engine.schema.validates(query.model_copy(update={'property_source':'entity'}))


def test_interpreter_exposes_one_interval_operation(household):
    engine,ctx,_=household
    operations=engine.schema.planner_capability_payload()['operations']
    assert 'date_difference' in operations
    assert not set(operations).intersection({'duration','completed_years'})
    assert engine.schema.planner_output_schema()['$defs']['SemanticFactRequest']['properties']['operation']['enum']==operations


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
    for locale, ending in [('zh', '没有找到符合筛选条件的记录。'), ('en', 'No records match the filters.')]:
        text = FactRenderer(detailed=True).render(empty,result,replace(ctx,locale=locale))
        assert text.endswith(ending)
        assert '1900-01-01' in text and '1901-01-01' in text
    dispatcher.entities['person:b'].pop('dob')
    result,*_=await engine.execute(query,ctx)
    assert result.status=='filter_input_missing'


@pytest.mark.asyncio
@pytest.mark.parametrize('filename', ['semantic_planner_age_filters.yaml','semantic_planner_date_intervals.yaml'])
async def test_age_and_date_filter_eval_expectations_on_invented_household(household,filename):
    from home_cortex.semantic_planner_benchmark import load_probe_dataset, score_structured_result
    engine,ctx,_=household
    dataset=load_probe_dataset(Path(__file__).parents[1]/'benchmarks'/filename)
    ctx=replace(ctx,current_time=dataset.frozen_time)
    example_texts={message['content'] for message in _semantic_planner_examples() if message['role']=='user'}
    for case in dataset.cases:
        assert case.utterance not in example_texts
        result,*_=await engine.execute(case.expected,replace(ctx,caller_entity_id=case.speaker_id))
        assert score_structured_result(result,case), case.case_id


@pytest.mark.asyncio
async def test_future_date_is_signed_interval_but_not_valid_age_predicate(household):
    engine,ctx,dispatcher=household
    dispatcher.entities['person:daughter']['dob']='2026-10-01'
    query=SemanticFactRequest(operation='date_difference',subject=ref(step('child','female')),property='birth_date',mode='days')
    result,*_=await engine.execute(query,ctx)
    assert result.value==-28 and result.unit=='days'
    members=SemanticReference(kind='current_household',entity_type='address',path=(SemanticRelationStep(relation='member'),))
    query=SemanticFactRequest(operation='count',subject=members,filters=(SemanticFilter(predicate='minor'),))
    result,*_=await engine.execute(query,ctx)
    assert result.status=='filter_input_missing'


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
async def test_request_filter_runs_before_scalar_cardinality(household):
    engine,ctx,dispatcher=household
    # The IR placements remain distinct, while collection filtering must happen
    # before scalar cardinality is enforced.
    scalar=SemanticFactRequest(operation='select',subject=ref(step('child')),property='birth_date',filters=(SemanticFilter(property='gender',value='female'),))
    path=scalar.model_copy(update={'subject':ref(step('child','female')),'filters':()})
    first,*_=await engine.execute(scalar,ctx)
    second,*_=await engine.execute(path,ctx)
    assert first.status == second.status == 'found'
    assert first.evidence.entity_ids == second.evidence.entity_ids == ('person:daughter',)
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
    assert payload['collection_predicates']['adult']['definition_only']=={'property':'birth_date','transform':'date_difference','mode':'years','require_past':True,'operator':'gte','value':18}
    assert set(payload['operations']).isdisjoint({'filter','traverse','adult','minor'})
    assert 'dob' not in json.dumps(payload)
    assert 'person:a' not in json.dumps(payload)
    utterances={c.utterance for c in load_semantic_eval_cases()}
    identity_kinds=[]
    for message in _semantic_planner_examples():
        if message['role']=='user': assert message['content'] not in utterances
        else:
            payload=json.loads(message['content'])
            if not payload.get('requires_fact') or payload.get('request') is None:
                continue
            request=SemanticFactRequest.model_validate(engine.schema.expand_planner_concepts(payload)['request'])
            assert engine.schema.validates(request)
            if request.operation=='resolve_reference' and not request.subject.path:
                identity_kinds.append(request.subject.kind)
    assert identity_kinds[:2]==['assistant','self']


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
    refs=defs['SemanticReference']['anyOf']
    kinds=[]
    for item in refs:
        kind=item['properties']['kind']
        kinds.extend(kind.get('enum', [kind['const']] if 'const' in kind else []))
    assert 'self' in kinds and 'assistant' in kinds
    assert 'entity_id' not in kinds
    ordinary=next(
        item for item in refs
        if set(item['properties']['kind'].get('enum', ())) >= {'self', 'assistant'}
    )
    assert 'turn_offset' not in ordinary['properties']
    assert 'cardinality' not in ordinary['properties']
    assert ordinary['required']==['kind']
    assert 'second person' in ordinary['description'].casefold()
    assert ordinary['properties']['kind']['enum']==['self','assistant','current_household']
    assert defs['SemanticFactRequest']['properties']['property']['anyOf'][0]=={'type':'null'}
    assert defs['SemanticFactRequest']['properties']['amount']['anyOf'][0]=={'type':'null'}
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
    assert 'same_entity' in engine.schema.planner_capability_payload()['operations']
    assert 'same_entity' in engine.schema.planner_capability_payload()['operation_requirements']


@pytest.mark.asyncio
async def test_same_entity_compares_resolved_ids_including_residence(household):
    engine, ctx, dispatcher = household
    speaker = SemanticReference(kind='self', entity_type='person')
    identity = SemanticFactRequest(
        operation='same_entity', subject=speaker, other=speaker,
    )
    assert engine.schema.validates(identity)
    same, *_ = await engine.execute(identity, ctx)
    assert same.status == 'found' and same.value is True
    assert FactRenderer().render(identity, same, replace(ctx, locale='zh')) == '是。'
    assert FactRenderer().render(identity, same, replace(ctx, locale='en')) == 'Yes.'
    different = SemanticFactRequest(
        operation='same_entity', subject=speaker, other=ref(step('spouse')),
    )
    other, *_ = await engine.execute(different, ctx)
    assert other.status == 'found' and other.value is False
    assert FactRenderer().render(different, other, replace(ctx, locale='zh')) == '不是。'
    assert not engine.schema.validates(identity.model_copy(update={'other': None}))
    assert not engine.schema.validates(identity.model_copy(update={'property': 'birth_date'}))
    assert not engine.schema.validates(identity.model_copy(
        update={'property_source': 'relationship', 'subject': ref(step('residence'))},
    ))

    dispatcher.entities['person:own_mother'] = {
        'id': 'person:own_mother', 'name': 'Own mother',
        'gender': 'female', 'dob': '1957-02-02',
    }
    dispatcher.edges['parent_of'].append({'from': 'person:own_mother', 'to': 'person:a'})
    dispatcher.edges['lives_in'].append({
        'from': 'person:own_mother', 'to': 'address:fictional',
        'start': '2018-01-01', 'end': None,
    })

    def composed(*concepts):
        payload = {
            'request': {
                'operation': 'same_entity',
                'property': None,
                'property_source': 'entity',
                'subject': {
                    'kind': 'self', 'entity_type': 'person',
                    'path': [{'concept': name} for name in concepts],
                },
                'other': {'kind': 'current_household', 'entity_type': 'address', 'value': None},
            }
        }
        request = SemanticFactRequest.model_validate(
            engine.schema.expand_planner_concepts(payload)['request']
        )
        assert engine.schema.validates(request)
        return request

    here = composed('mother', 'residence')
    result, *_ = await engine.execute(here, ctx)
    assert result.status == 'found' and result.value is True
    assert result.evidence.entity_ids == ('address:fictional', 'address:fictional')
    speaker_here, *_ = await engine.execute(composed('residence'), ctx)
    assert speaker_here.status == 'found' and speaker_here.value is True

    dispatcher.entities['address:other'] = {
        'id': 'address:other', 'full_address': 'Somewhere else',
    }
    dispatcher.edges['lives_in'] = [
        edge for edge in dispatcher.edges['lives_in']
        if edge.get('from') != 'person:own_mother'
    ]
    dispatcher.edges['lives_in'].append({
        'from': 'person:own_mother', 'to': 'address:other',
        'start': '2018-01-01', 'end': None,
    })
    away, *_ = await engine.execute(here, ctx)
    assert away.status == 'found' and away.value is False
    assert FactRenderer().render(here, away, replace(ctx, locale='zh')) == '不是。'

    dispatcher.edges['lives_in'] = [
        edge for edge in dispatcher.edges['lives_in']
        if edge.get('from') != 'person:own_mother'
    ]
    missing, *_ = await engine.execute(here, ctx)
    assert missing.status == 'relationship_not_found'


@pytest.mark.asyncio
async def test_atomic_in_law_concept_does_not_duplicate_its_spouse_step(household):
    engine, ctx, dispatcher = household
    dispatcher.entities['person:own_father'] = {
        'id': 'person:own_father', 'name': 'Own father', 'gender': 'male', 'dob': '1950-01-02',
    }
    dispatcher.edges['parent_of'].append({'from': 'person:own_father', 'to': 'person:a'})

    def request(*concepts, operation='resolve_reference', property=None):
        payload = {'request': {'operation': operation, 'property': property,
                   'subject': {'kind': 'self', 'entity_type': 'person',
                               'path': [{'concept': name} for name in concepts]}}}
        return SemanticFactRequest.model_validate(engine.schema.expand_planner_concepts(payload)['request'])

    # The two complete concepts must remain different even in adjacent turns.
    own, *_ = await engine.execute(request('father'), ctx)
    in_law, *_ = await engine.execute(request('father_in_law'), ctx)
    assert own.evidence.entity_ids == ('person:own_father',)
    assert in_law.evidence.entity_ids == ('person:father',)
    birth, *_ = await engine.execute(request('father_in_law', operation='select', property='birth_date'), ctx)
    assert birth.value == '1955-01-01'
    # Explicit outer possession is legitimate; never remove a repeated spouse
    # downstream just because it was wrong for a different natural-language query.
    nested, *_ = await engine.execute(request('spouse', 'father_in_law'), ctx)
    assert nested.evidence.entity_ids == ('person:own_father',)
