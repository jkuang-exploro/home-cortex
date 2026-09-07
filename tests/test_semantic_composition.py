"""Compositional acceptance on invented graph data and a deterministic interpreter."""
import json
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from ollama import ChatResponse
from pydantic import ValidationError

from home_cortex.agent_service import AgentService
from home_cortex.tools import get_tool_definitions
from home_cortex.ollama import OllamaService
from home_cortex.operator_registry import OperatorInput, OperatorExecutionError, execute_operator
from home_cortex.semantic_facts import (
    DiscourseContext, FactRenderer, SemanticFactRequest, SemanticReference,
    SemanticRelationStep, SemanticFilter, SemanticPlannerFailure,
)
from test_semantic_contract import household


def members():
    return SemanticReference(kind='current_household', path=(SemanticRelationStep(relation='member'),))


def named(name='son1'):
    return SemanticReference(kind='named_entity', value=name, entity_type='person')


def discourse(offset=1, cardinality='single', **kwargs):
    return SemanticReference(kind='discourse', entity_type='person', turn_offset=offset,
                             cardinality=cardinality, **kwargs)


def ages(**kwargs):
    return SemanticFactRequest(operation='date_difference', subject=members(),
                               property='birth_date', mode='years', projection='each', **kwargs)


def anniversary(subject=None, **kwargs):
    return SemanticFactRequest(operation='date_add', subject=subject or discourse(),
                               property='birth_date', amount=10, mode='years', **kwargs)


class Client:
    """Only the LLM transport is mocked; normal service/planner/resolver run."""
    def __init__(self, *plans):
        self.plans = list(plans)
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        plan = self.plans.pop(0)
        payload = {'requires_fact': plan is not None,
                   'request': plan.model_dump(mode='json') if plan is not None else None}
        return ChatResponse(message={'role': 'assistant', 'content': json.dumps(payload)})


def agent_for(household, *plans):
    engine, context, dispatcher = household
    client = Client(*plans)
    agent = AgentService(OllamaService('http://unused', 'fake', client=client), dispatcher,
                         system_prompt='test', tools=get_tool_definitions(['calculate']), schema_catalog=engine.schema.catalog,
                         home_entity_id=context.household_id, clock=lambda: context.current_time)
    return agent, client


@pytest.mark.asyncio
async def test_projection_preserves_association_partial_evidence_and_exclusion(household):
    engine, context, dispatcher = household
    dispatcher.entities['person:son2'].pop('dob')
    query = ages(exclude=(SemanticReference(kind='self'),))
    result, *_ = await engine.execute(query, context)
    assert result.status == 'found' and result.shape == 'rows' and result.value is None
    rows = {row.entity['id']: row for row in result.rows}
    assert len(rows) == 7 and 'person:a' not in rows
    assert rows['person:son1'].value == 18 and rows['person:son1'].unit == 'years'
    assert rows['person:son2'].status == 'computation_input_missing'
    assert rows['person:son2'].unit == 'years'
    assert rows['person:son2'].missing_requirements == ('birth_date',)
    for entity_id, row in rows.items():
        assert row.evidence.entity_ids == (entity_id,)
        assert len(row.evidence.relationships) == 1
        assert row.evidence.relationships[0].source_id == entity_id
    for locale in ('en', 'zh'):
        text = FactRenderer().render(query, result, replace(context, locale=locale))
        assert 'son1' in text and 'son2' in text and '18' in text
    singular, *_ = await engine.execute(query.model_copy(update={'projection':'scalar', 'exclude':()}), context)
    assert singular.status == 'ambiguous'


@pytest.mark.asyncio
async def test_empty_rows_and_filter_failure_are_distinct(household):
    engine, context, dispatcher = household
    query = ages(filters=(SemanticFilter(property='birth_date', operator='lt', value='1900-01-01'),))
    result, *_ = await engine.execute(query, context)
    assert result.status == 'found' and result.shape == 'rows' and result.rows == ()
    unresolved, *_ = await engine.execute(query.model_copy(update={'exclude':(named('absent'),)}), context)
    assert unresolved.status == 'entity_not_found'
    dispatcher.entities['person:son2'].pop('dob')
    result, *_ = await engine.execute(query, context)
    assert result.status == 'filter_input_missing'


@pytest.mark.asyncio
async def test_relationship_rows_keep_multiple_edges_and_missing_dates(household):
    engine, context, dispatcher = household
    dispatcher.edges['lives_in'].append({'from':'person:son1','to':'address:fictional','start':'2020-08-01'})
    dispatcher.edges['lives_in'][0].pop('start')
    query = ages().model_copy(update={'property':'start_date','property_source':'relationship'})
    result, *_ = await engine.execute(query, context)
    rows = [row for row in result.rows if row.entity['id'] == 'person:son1']
    assert len(rows) == 2 and {row.value for row in rows} == {8, 6}
    assert {row.evidence.relationships[0].start for row in rows} == {'2018-01-01','2020-08-01'}
    assert result.rows[0].status == 'relation_property_unavailable'
    assert all(row.unit == 'years' for row in rows)
    added, *_ = await engine.execute(query.model_copy(update={'operation':'date_add','amount':3}), context)
    assert {row.value for row in added.rows if row.entity['id']=='person:son1'} == {'2021-01-01','2023-08-01'}


@pytest.mark.asyncio
async def test_named_ambiguity_not_hidden_by_projection_or_exclusion(household):
    engine, context, dispatcher = household
    dispatcher.entities['person:b']['name'] = 'son1'
    query = ages().model_copy(update={'subject':named(), 'projection':'scalar'})
    assert (await engine.execute(query,context))[0].status == 'ambiguous'
    assert (await engine.execute(ages(exclude=(named(),)), context))[0].status == 'ambiguous'
    assert (await engine.execute(ages(exclude=(SemanticReference(kind='unresolved', entity_type='person'),)),context))[0].status == 'ambiguous'


@pytest.mark.parametrize('value,amount,mode,expected', [
    ('2000-02-29',1,'years','2001-03-01'),
    ('2000-02-29',4,'years','2004-02-29'),
    ('2023-01-31',1,'months','2023-03-01'),
    ('2024-01-31',1,'months','2024-03-01'),
    ('2024-03-31',-1,'months','2024-03-01'),
    ('2024-01-31',2,'months','2024-03-31'),
    ('2024-02-29',1,'days','2024-03-01'),
    ('2016-05-12',10,'years','2026-05-12'),
    ('2024-01-01',-1,'days','2023-12-31'),
    ('2000-02-29',0,'years','2000-02-29'),
])
def test_calendar_offsets(value, amount, mode, expected):
    result = execute_operator('date_add', OperatorInput([{'v':value}], field='v', amount=amount,
                              mode=mode, now=datetime(2030,1,1,tzinfo=ZoneInfo('America/Los_Angeles'))))
    assert result == expected


@pytest.mark.parametrize('stored,amount,mode,expected', [
    ('2024-01-02T01:30:00Z',1,'months','2024-02-01T17:30:00-08:00'),
    ('2024-01-02T01:30:00',1,'months','2024-02-01T17:30:00-08:00'),
    ('2024-03-09T12:00:00-08:00',1,'days','2024-03-10T12:00:00-07:00'),
])
def test_offsets_use_household_wall_time(stored,amount,mode,expected):
    assert execute_operator('date_add', OperatorInput([{'v':stored}],field='v',amount=amount,mode=mode,
                            now=datetime(2030,1,1,tzinfo=ZoneInfo('America/Los_Angeles')))) == expected


@pytest.mark.parametrize('stored,amount,mode', [
    ('2024-03-09T02:30:00-08:00',1,'days'),
    ('2024-11-02T01:30:00-07:00',1,'days'),
    ('9999-12-31',1,'days'),
    ('0001-01-01',-1,'years'),
])
def test_unrepresentable_calendar_offset_fails(stored, amount, mode):
    with pytest.raises(OperatorExecutionError):
        execute_operator('date_add', OperatorInput([{'v':stored}],field='v',amount=amount,mode=mode,
                            now=datetime(2030,1,1,tzinfo=ZoneInfo('America/Los_Angeles'))))


@pytest.mark.parametrize('amount', [True, '10', 1.5, 120001])
def test_offset_amount_is_strict_integer(amount):
    with pytest.raises(ValidationError):
        anniversary().model_validate({**anniversary().model_dump(),'amount':amount})


@pytest.mark.asyncio
async def test_past_anniversary_and_relationship_owner(household):
    engine, context, _ = household
    query = anniversary(named())
    result, *_ = await engine.execute(query, context)
    assert result.value == '2018-09-03' and result.unit is None
    next_date, *_ = await engine.execute(query.model_copy(update={'operation':'annual_occurrence','amount':None,'mode':None}),context)
    assert next_date.value == '2026-09-03'
    marriage = query.model_copy(update={'subject':SemanticReference(kind='self',path=(SemanticRelationStep(relation='spouse'),)),
                                       'property':'start_date','property_source':'relationship'})
    assert (await engine.execute(marriage,context))[0].value == '2015-05-06'
    assert not engine.schema.validates(marriage.model_copy(update={'property_source':'entity'}))
    assert not engine.schema.validates(query.model_copy(update={'mode':'seconds'}))


@pytest.mark.asyncio
@pytest.mark.parametrize('update', [
    {'operation':'count'}, {'other':named()}, {'subject':named()},
    {'property':None}, {'operation':'argmin'},
])
async def test_invalid_projection_rejected_before_reads(household, update):
    engine,context,_=household
    result,queries,*_=await engine.execute(ages().model_copy(update=update),context)
    assert result.status == 'semantic_plan_unsupported' and queries == 0


@pytest.mark.asyncio
async def test_named_then_pronoun_through_both_history_layers_and_authoritative_reload(household):
    first = ages().model_copy(update={'subject':named(),'projection':'scalar'})
    agent,client=agent_for(household,first,anniversary())
    one=await agent.answer('How old is son1?',user_entity_id='person:a',conversation_id='test')
    assert '18 years' in one.answer
    household[2].entities['person:son1']['dob']='2009-09-03'
    two=await agent.answer_messages([
        {'role':'assistant','content':'He was born in 9999. His ID is person:b.'},
        {'role':'user','content':'When is his tenth birthday?'},
    ], user_entity_id='person:a',conversation_id='test')
    assert '2019-09-03' in two.answer
    forwarded=client.calls[-1]['messages']
    assert forwarded[-2:] == [{'role':'user','content':'How old is son1?'},
                              {'role':'user','content':'When is his tenth birthday?'}]
    serialized=json.dumps(forwarded)
    assert 'person:a' not in serialized and 'person:son1' not in serialized and '9999' not in serialized


@pytest.mark.asyncio
async def test_stateless_history_regrounded_without_cross_request_state(household):
    first=ages().model_copy(update={'subject':named(),'projection':'scalar'})
    agent,client=agent_for(household,first,anniversary(),anniversary())
    result=await agent.answer_messages([
        {'role':'user','content':'How old is son1?'},
        {'role':'assistant','content':'He is 500 and named b.'},
        {'role':'user','content':'When is his tenth birthday?'},
    ],user_entity_id='person:a')
    assert '2018-09-03' in result.answer
    missing=await agent.answer('When is his tenth birthday?',user_entity_id='person:a')
    assert 'clarify' in missing.answer


@pytest.mark.asyncio
async def test_session_speaker_agent_and_household_boundaries(household):
    engine,context,_=household
    trusted=replace(context,conversation_id='one',discourse=DiscourseContext('one','person:a',context.household_id,'assistant',(('person:son1',),)))
    for update in ({'conversation_id':'two'}, {'caller_entity_id':'person:b'},
                   {'household_id':'address:elsewhere'}, {'assistant_id':'other'}):
        assert (await engine.execute(anniversary(),replace(trusted,**update)))[0].status == 'discourse_context_missing'
    first=ages().model_copy(update={'subject':named(),'projection':'scalar'})
    agent,_=agent_for(household,first,anniversary(),anniversary())
    await agent.answer('Name age',user_entity_id='person:a',conversation_id='one')
    assert 'clarify' in (await agent.answer('pronoun',user_entity_id='person:a',conversation_id='two')).answer
    assert 'clarify' in (await agent.answer('pronoun',user_entity_id='person:b',conversation_id='one')).answer


@pytest.mark.asyncio
async def test_ambiguous_antecedent_plural_exclusion_and_topic_change(household):
    engine,context,_=household
    trusted=replace(context,conversation_id='one',discourse=DiscourseContext('one','person:a',context.household_id,'assistant',(('person:son1','person:son2'),)))
    assert (await engine.execute(anniversary(),trusted))[0].status == 'ambiguous'
    query=ages(exclude=(discourse(cardinality='collection'),))
    result,*_=await engine.execute(query,trusted)
    assert len(result.rows)==6
    plural=anniversary(discourse(cardinality='collection'),projection='each')
    assert len((await engine.execute(plural,trusted))[0].rows)==2
    first=ages().model_copy(update={'subject':named(),'projection':'scalar'})
    # Non-fact turn is recorded by the semantic coordinator; no prose generation needed.
    agent,_=agent_for(household,first,None,anniversary(),anniversary(discourse(3)))
    ctx=replace(context,conversation_id='one')
    for text in ('person age','new topic'):
        await agent.semantic_conversations.try_answer([{'role':'user','content':text}],context=ctx)
    answer=await agent.semantic_conversations.try_answer([{'role':'user','content':'his date'}],context=ctx)
    assert answer.result.status=='discourse_context_missing'
    explicit=await agent.semantic_conversations.try_answer([{'role':'user','content':'the original person'}],context=ctx)
    assert explicit.result.value=='2018-09-03'


@pytest.mark.asyncio
async def test_discourse_path_composes_relationship_anniversary(household):
    first=SemanticFactRequest(operation='resolve_reference',subject=named('a'))
    followup=SemanticFactRequest(operation='date_add',subject=discourse(path=(SemanticRelationStep(relation='spouse'),)),
                                property='start_date',property_source='relationship',amount=7,mode='years')
    agent,_=agent_for(household,first,followup)
    await agent.answer('Who is a?',user_entity_id='person:a',conversation_id='one')
    answer=await agent.answer('His seventh wedding anniversary?',user_entity_id='person:a',conversation_id='one')
    assert '2012-05-06' in answer.answer


@pytest.mark.asyncio
async def test_model_cannot_insert_ids_in_exclusion(household):
    query=ages(exclude=(SemanticReference(kind='entity_id',value='person:a'),))
    agent,_=agent_for(household,query,query)
    with pytest.raises(SemanticPlannerFailure) as failure:
        await agent.semantic_facts.planner.plan([{'role':'user','content':'members'}],household[1])
    assert failure.value.diagnostics.validation_result=='MODEL_ORIGINATED_ENTITY_ID'


@pytest.mark.asyncio
async def test_projection_property_selection_and_serialization(household):
    from home_cortex.semantic_planner_benchmark import serialize_fact_result, fact_result_from_serialized, normalize_semantic_request
    engine,context,dispatcher=household
    dispatcher.entities['person:son1'].pop('dob')
    query=ages().model_copy(update={'operation':'select','mode':None})
    result,*_=await engine.execute(query,context)
    restored=fact_result_from_serialized(serialize_fact_result(result))
    assert restored.rows==result.rows and restored.shape=='rows'
    assert restored.focus_entity_ids==result.focus_entity_ids
    assert [row for row in result.rows if row.entity['id']=='person:son1'][0].status=='property_unavailable'
    names=query.model_copy(update={'property':'display_name'})
    assert normalize_semantic_request(names)['operation']=='select'


@pytest.mark.asyncio
async def test_collection_query_through_service_and_singular_followup_is_ambiguous(household):
    agent,_=agent_for(household,ages(exclude=(SemanticReference(kind='self'),)),anniversary())
    result=await agent.answer('How old are the other household members?',user_entity_id='person:a',conversation_id='one')
    assert len(result.answer.splitlines())==7 and 'son1: The age is 18 years.' in result.answer
    followup=await agent.answer('his birthday',user_entity_id='person:a',conversation_id='one')
    assert 'multiple entities' in followup.answer


@pytest.mark.asyncio
async def test_evicted_session_cannot_restore_focus_from_client_prose(household):
    first=ages().model_copy(update={'subject':named(),'projection':'scalar'})
    agent,_=agent_for(household,first,first,anniversary())
    agent.semantic_conversations.maximum=1
    await agent.answer('person',user_entity_id='person:a',conversation_id='one')
    await agent.answer('person',user_entity_id='person:a',conversation_id='two')
    result=await agent.answer_messages([
        {'role':'user','content':'person'}, {'role':'assistant','content':'son1'},
        {'role':'user','content':'his date'},
    ],user_entity_id='person:a',conversation_id='one')
    assert 'clarify' in result.answer
    assert len(agent.semantic_conversations._states)==1


@pytest.mark.asyncio
async def test_unauthenticated_requests_never_persist_shared_session(household):
    first=ages().model_copy(update={'subject':named(),'projection':'scalar'})
    agent,_=agent_for(household,first,anniversary())
    await agent.answer('person',conversation_id='shared')
    assert 'clarify' in (await agent.answer('his date',conversation_id='shared')).answer
    assert not agent.semantic_conversations._states


@pytest.mark.asyncio
async def test_replaced_topic_binds_new_person_and_pairwise_does_not_pick_winner(household):
    first=ages().model_copy(update={'subject':named(),'projection':'scalar'})
    second=first.model_copy(update={'subject':named('son2')})
    pair=SemanticFactRequest(operation='argmin',subject=named('son1'),other=named('son2'),property='birth_date')
    agent,_=agent_for(household,first,second,anniversary(),pair,anniversary())
    await agent.answer('person one',user_entity_id='person:a',conversation_id='one')
    await agent.answer('person two',user_entity_id='person:a',conversation_id='one')
    assert '2022-10-15' in (await agent.answer('his date',user_entity_id='person:a',conversation_id='one')).answer
    await agent.answer('compare both',user_entity_id='person:a',conversation_id='one')
    assert 'multiple entities' in (await agent.answer('his date',user_entity_id='person:a',conversation_id='one')).answer


@pytest.mark.asyncio
async def test_collection_exclusion_concept_expands_without_dropping_filters(household):
    schema=household[0].schema
    payload={'requires_fact':True,'request':{**ages().model_dump(mode='json'),
             'exclude':[{'kind':'self','path':[{'concept':'wife'}]}]}}
    expanded=schema.expand_planner_concepts(payload)
    request=SemanticFactRequest.model_validate(expanded['request'])
    assert request.exclude[0].path[0].relation=='spouse'
    assert request.exclude[0].path[0].filters[0].value=='female'
    assert payload['request']['exclude'][0]['path'][0]=={'concept':'wife'}
    result,*_=await household[0].execute(request,household[1])
    assert 'person:b' not in {row.entity['id'] for row in result.rows}


@pytest.mark.asyncio
async def test_session_concurrent_turns_serialize(household):
    import asyncio
    first=ages().model_copy(update={'subject':named(),'projection':'scalar'})
    agent,client=agent_for(household,first,anniversary())
    original=client.chat
    entered=asyncio.Event()
    release=asyncio.Event()
    async def slow_chat(**kwargs):
        if not entered.is_set():
            entered.set()
            await release.wait()
        return await original(**kwargs)
    client.chat=slow_chat
    one=asyncio.create_task(agent.answer('person',user_entity_id='person:a',conversation_id='one'))
    await entered.wait()
    two=asyncio.create_task(agent.answer('his date',user_entity_id='person:a',conversation_id='one'))
    await asyncio.sleep(0)
    assert not client.calls
    release.set()
    results=await asyncio.gather(one,two)
    assert '2018-09-03' in results[1].answer


@pytest.mark.asyncio
async def test_deleted_antecedent_is_reloaded_and_fails(household):
    first=ages().model_copy(update={'subject':named(),'projection':'scalar'})
    agent,_=agent_for(household,first,anniversary())
    await agent.answer('person',user_entity_id='person:a',conversation_id='one')
    del household[2].entities['person:son1']
    assert 'could not find' in (await agent.answer('his date',user_entity_id='person:a',conversation_id='one')).answer


@pytest.mark.parametrize('route,stream', [('/v1/chat/completions',False),('/v1/chat/completions',True),('/agent/steward/chat',False)])
def test_http_named_then_pronoun_and_session_ownership(household, api_client, route, stream):
    from home_cortex.api import app, VIRTUAL_MODEL
    client,_=api_client
    first=ages().model_copy(update={'subject':named(),'projection':'scalar'})
    agent,_=agent_for(household,first,anniversary())
    app.state.agents={'steward':agent}
    app.state.retrieval.records=[{'id':'person:a','name':'Synthetic A'}, {'id':'person:b','name':'Synthetic B'}]
    app.state.settings=SimpleNamespace(cortex_api_key=None,cortex_identity_map={'id:a':'person:a','id:b':'person:b'})
    headers={'X-OpenWebUI-User-Id':'a'}
    created=client.post('/agent/steward/conversations',json={'language':'en'},headers=headers)
    assert created.status_code==201
    conversation_id=created.json()['id']
    def body(text):
        if route.endswith('completions'):
            return {'model':VIRTUAL_MODEL,'messages':[{'role':'user','content':text}],
                    'stream':stream,'conversation_id':conversation_id}
        return {'message':text,'conversation_id':conversation_id}
    first_response=client.post(route,json=body('How old is son1?'),headers=headers)
    assert first_response.status_code==200 and '18 years' in first_response.text
    response=client.post(route,json=body('When is his tenth birthday?'),headers=headers)
    assert response.status_code==200 and '2018-09-03' in response.text
    assert 'How may I help you' not in response.text
    denied=client.post(route,json=body('his date'),headers={'X-OpenWebUI-User-Id':'b'})
    assert denied.status_code==404 and denied.json()['error']['code']=='conversation_not_found'


# Reuse the API fixture's isolated application lifecycle, replacing its fake agent above.
from test_api import api_client


@pytest.mark.asyncio
async def test_row_scoring_requires_associations_values_units_and_status(household):
    from home_cortex.semantic_planner_benchmark import SemanticEvalCase, score_structured_result
    engine,context,dispatcher=household
    dispatcher.entities['person:son2'].pop('dob')
    query=ages()
    result,*_=await engine.execute(query,context)
    expected=tuple({'entity_id':row.entity['id'],'value':row.value,'unit':row.unit,'status':row.status}
                   for row in result.rows)
    case=SemanticEvalCase('invented','person:a','collection','projection',query,
                           expected_status='found',expected_rows=expected)
    assert score_structured_result(result,case) is True
    assert score_structured_result(result,replace(case,expected_rows=None)) is None
    for field,bad in [('entity_id','person:absent'),('value',9999),('unit','seconds'),('status','property_unavailable')]:
        wrong=replace(case,expected_rows=({**expected[0],field:bad},*expected[1:]))
        assert score_structured_result(result,wrong) is False


def test_date_add_boundary_agrees_with_completed_period(household):
    for source, mode in [('2023-01-31','months'),('2024-01-31','months'),('2000-02-29','years')]:
        target=execute_operator('date_add',OperatorInput([{'v':source}],field='v',amount=1,mode=mode,
                                now=household[1].current_time))
        instant=datetime.fromisoformat(target+'T00:00:00-08:00')
        assert execute_operator('date_difference',OperatorInput([{'v':source}],field='v',mode=mode,now=instant))==1


@pytest.mark.asyncio
async def test_relationship_projection_filters_the_same_edge(household):
    engine,context,dispatcher=household
    dispatcher.edges['lives_in'].append({'from':'person:son1','to':'address:fictional','start':'2020-08-01'})
    query=ages(filters=(SemanticFilter(property='start_date',source='relation',operator='gte',value='2020-01-01'),)).model_copy(
        update={'property':'start_date','property_source':'relationship'})
    result,*_=await engine.execute(query,context)
    assert len(result.rows)==1 and result.rows[0].entity['id']=='person:son1'
    assert result.rows[0].value==6
    assert result.rows[0].evidence.relationships[0].start=='2020-08-01'


@pytest.mark.asyncio
async def test_discourse_collection_composes_entity_filters_and_predicates(household):
    engine,context,_=household
    context=replace(context,conversation_id='one',discourse=DiscourseContext('one','person:a',context.household_id,'assistant',(('person:son1','person:son2'),)))
    query=ages(filters=(SemanticFilter(predicate='minor'),)).model_copy(update={'subject':discourse(cardinality='collection')})
    result,*_=await engine.execute(query,context)
    assert len(result.rows)==1 and result.rows[0].entity['id']=='person:son2'
    assert result.rows[0].value==13
