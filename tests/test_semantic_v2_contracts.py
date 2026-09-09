"""V2 contract invariants; synthetic graphs and existing full-plan composition golds."""
import copy
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import yaml

from home_cortex.composition_eval import (
    household_engine, load_standalone_cases, load_all_composition_datasets,
    request_context, gold_matches, HOUSEHOLD_IDS,
)
from home_cortex.semantic_contracts import PropertyContract, SemanticType
from home_cortex.semantic_ontology import SemanticOntology
from home_cortex.semantic_facts import (
    DiscourseContext, HouseholdFactEngine, SemanticSchemaRegistry,
    SemanticFactRequest, SemanticReference, SemanticRelationStep, SemanticFilter,
    SemanticFactService, SemanticFactPlanner,
)
from home_cortex.schema_catalog import EntityTypeSchema
from test_semantic_contract import household

V2 = Path(__file__).parents[1] / 'schemas/semantic/ontology-v2.yaml'


def candidate(engine):
    return HouseholdFactEngine(engine.dispatcher, SemanticSchemaRegistry(engine.schema.catalog, SemanticOntology.from_file(V2)))


def members():
    return SemanticReference(kind='current_household', path=(SemanticRelationStep(relation='member'),))


def count(*filters):
    return SemanticFactRequest(operation='count', subject=members(), filters=filters)


@pytest.mark.asyncio
async def test_all_composition_gold_and_sequence_results_match_v1():
    engines = {}
    for name in ('alpha', 'beta', 'gamma'):
        old, _ = household_engine(name)
        engines[name] = old, candidate(old)
    for case in load_standalone_cases():
        old, new = engines[case.household]
        for request in (case.expected, *case.acceptable_alternatives):
            assert new.schema.validates(request), (case.case_id, new.schema.contract_error(request))
            before = copy.deepcopy(request)
            context = request_context(speaker_id=case.speaker_id, household=case.household)
            baseline, *_ = await old.execute(request, context)
            actual, *_ = await new.execute(request, context)
            assert actual == baseline, case.case_id
            assert request == before
        actual, *_ = await new.execute(case.expected, context)
        assert gold_matches(actual, case), case.case_id
    for dataset in load_all_composition_datasets():
        for sequence in dataset.sequences:
            old, new = engines[sequence.household]
            turns = []
            for plan_id in sequence.plan_ids:
                request = dataset.plans[plan_id]
                context = request_context(speaker_id=sequence.speaker_id, household=sequence.household,
                    conversation_id=sequence.sequence_id, discourse=DiscourseContext(
                        sequence.sequence_id, sequence.speaker_id, HOUSEHOLD_IDS[sequence.household],
                        'assistant:composition', tuple(turns[-8:]),
                    ))
                baseline, *_ = await old.execute(request, context)
                actual, *_ = await new.execute(request, context)
                assert actual == baseline, (sequence.sequence_id, plan_id)
                turns.append(actual.focus_entity_ids if actual.status == 'found' else ())
            assert gold_matches(actual, sequence.last)


@pytest.mark.parametrize('value,code', [(17, 'INVALID_LITERAL_TYPE'), ('女性', 'VALUE_OUT_OF_DOMAIN'),
                                       ('unlisted', 'VALUE_OUT_OF_DOMAIN'), (None, 'INVALID_LITERAL_TYPE')])
def test_domain_errors_are_not_repaired(household, value, code):
    schema = candidate(household[0]).schema
    request = count(SemanticFilter(property='gender', value=value))
    before = request.model_dump()
    assert schema.contract_error(request) == code
    assert not schema.validates(request)
    assert request.model_dump() == before


@pytest.mark.parametrize('kind,value,accepted', [
    ('integer', True, False), ('integer', 1.0, False), ('number', True, False),
    ('number', float('nan'), False), ('number', float('inf'), False), ('number', 1, True),
    ('date', '2024-02-29', True), ('date', '2023-02-29', False), ('date', '20240101', False),
    ('datetime', '2026-01-01T00:00:00Z', True), ('datetime', '2026-01-01T00:00:00', False),
])
def test_types_are_strict(kind, value, accepted):
    assert SemanticType.parse({'kind': kind}).accepts(value) is accepted


@pytest.mark.parametrize('operator,value', [('in', ()), ('in', ('female', 3)), ('gt', 'female')])
def test_operator_domain_and_membership(household, operator, value):
    schema = candidate(household[0]).schema
    assert not schema.validates(count(SemanticFilter(property='gender', operator=operator, value=value)))


@pytest.mark.parametrize('value', [('2026-01-02', '2026-01-01'), ('2026-01-01', '2026-01-01'), ('2026-02-30', '2026-03-01')])
def test_date_ranges_reject_invalid_boundaries(household, value):
    schema = candidate(household[0]).schema
    assert not schema.validates(count(SemanticFilter(property='birth_date', operator='date_range', value=value)))


@pytest.mark.asyncio
@pytest.mark.parametrize('value,status', [(17, 'filter_unsupported'), ('not-in-domain', 'filter_unsupported'), (None, 'filter_input_missing')])
async def test_bad_stored_data_is_not_a_zero_count(household, value, status):
    engine = candidate(household[0])
    engine.dispatcher.entities['person:a']['gender'] = value
    request = count(SemanticFilter(property='gender', value='female'))
    result, *_ = await engine.execute(request, household[1])
    assert result.status == status and result.value is None


@pytest.mark.asyncio
async def test_bad_projection_and_anchor_data_fail_explicitly(household):
    engine = candidate(household[0])
    engine.dispatcher.entities['person:a']['dob'] = 'invalid'
    query = SemanticFactRequest(operation='select', subject=SemanticReference(kind='self'), property='birth_date')
    result, *_ = await engine.execute(query, household[1])
    assert result.status == 'property_unavailable'
    query = query.model_copy(update={'operation': 'date_difference', 'mode': 'years'})
    result, *_ = await engine.execute(query, household[1])
    assert result.status == 'computation_input_missing'
    payload = {'request': {'operation': 'count', 'subject': {'kind': 'self', 'path': [{'concept': 'older_brother'}]}}}
    query = SemanticFactRequest.model_validate(engine.schema.expand_planner_concepts(payload)['request'])
    # Our fixture has no parents for self; use a child as the anchor to reach siblings.
    context = replace(household[1], caller_entity_id='person:son1')
    engine.dispatcher.entities['person:son1']['dob'] = 'invalid'
    result, *_ = await engine.execute(query, context)
    assert result.status == 'filter_unsupported'


def test_schema_generation_and_runtime_share_closed_domain(household):
    schema = candidate(household[0]).schema
    payload = schema.planner_capability_payload()
    assert set(payload['property_contracts']['gender']['values']) == {'male', 'female'}
    generated = schema.planner_output_schema()
    branches = generated['$defs']['SemanticCollectionFilter']['anyOf']
    gender = [branch for branch in branches if branch['properties'].get('property', {}).get('enum') == ['gender']]
    assert gender
    assert not any('gt' in branch['properties']['operator']['enum'] for branch in gender)
    equality = next(branch for branch in gender if 'eq' in branch['properties']['operator']['enum'])
    assert equality['properties']['value']['enum'] == ['male', 'female']
    assert all('value_from' not in branch['properties'] for branch in branches)
    assert all('dob' not in json.dumps(view) and 'lives_in' not in json.dumps(view) for view in (generated, payload))
    payload['property_contracts'].clear()
    generated['$defs'].clear()
    assert schema.planner_capability_payload()['property_contracts']
    assert schema.planner_output_schema()['$defs']
    with pytest.raises(FrozenInstanceError):
        schema.contracts.fingerprint = 'changed'


def test_contradiction_does_not_delete_conditions_or_nested_hops(household):
    schema = candidate(household[0]).schema
    request = count(SemanticFilter(predicate='adult'), SemanticFilter(predicate='minor'))
    assert schema.contract_error(request) == 'CONTRADICTORY_PREDICATES'
    assert len(request.filters) == 2
    payload = {'request': {'operation': 'count', 'subject': {'kind': 'self', 'path': [{'concept': 'wife'}, {'concept': 'father_in_law'}]}}}
    before = copy.deepcopy(payload)
    expanded = schema.expand_planner_concepts(payload)
    request = SemanticFactRequest.model_validate(expanded['request'])
    assert schema.validates(request)
    assert [step.relation for step in request.subject.path] == ['spouse', 'spouse', 'parent']
    assert [item.value for step in request.subject.path for item in step.filters] == ['female', 'male']
    assert before == payload


@pytest.mark.asyncio
async def test_stateless_persistence_and_isolation_remain_unchanged(household):
    engine = candidate(household[0])
    context = replace(household[1], conversation_id='one')
    request = SemanticFactRequest(operation='select', subject=SemanticReference(kind='discourse', entity_type='person', turn_offset=1), property='birth_date')
    trusted = replace(context, discourse=DiscourseContext('one', context.caller_entity_id, context.household_id, context.assistant_id, (('person:b',),)))
    service = SemanticFactService(engine, SemanticFactPlanner(None, engine.schema))
    answer = await service.answer_request(request, context=trusted)
    assert answer.result.status == 'found' and answer.timings.llm_call_count == 0
    for update in ({'discourse': None}, {'conversation_id': 'two'}, {'caller_entity_id': 'person:b'},
                   {'household_id': 'address:elsewhere'}, {'assistant_id': 'another'}):
        result, *_ = await engine.execute(request, replace(trusted, **update))
        assert result.status == 'discourse_context_missing'
    result, *_ = await engine.execute(count(SemanticFilter(property='gender', value='female')), household[1])
    assert result.status == 'found' and result.value == 3


def load_changed(tmp_path, change):
    raw = yaml.safe_load(V2.read_text())
    change(raw)
    path = tmp_path / 'v2.yaml'
    path.write_text(yaml.safe_dump(raw, allow_unicode=True))
    return SemanticOntology.from_file(path)


@pytest.mark.parametrize('change', [
    lambda raw: raw['properties']['gender'].update(type={'kind': 'imaginary'}),
    lambda raw: raw['properties']['gender'].update(filter_operators=['gt']),
    lambda raw: raw['properties']['gender']['values']['female'].update(aliases=['male']),
    lambda raw: raw['properties']['gender'].pop('applies_to'),
    lambda raw: raw['properties']['gender']['applies_to'].update(relationship=['invented']),
    lambda raw: raw['collection_predicates']['minor'].pop('disjoint_with'),
    lambda raw: raw['reference_concepts']['wife']['path'][0]['filters'][0].update(value='unlisted'),
])
def test_invalid_declarations_fail_at_load(tmp_path, change):
    with pytest.raises(ValueError):
        load_changed(tmp_path, change)


def test_new_numeric_property_needs_only_declarations_and_binding(household, tmp_path):
    def change(raw):
        raw['properties']['score'] = {'fields': ['score'], 'aliases': [], 'type': {'kind': 'integer'},
            'applies_to': {'entity': ['person'], 'relationship': []}, 'filter_operators': ['eq', 'gte', 'in', 'exists']}
    ontology = load_changed(tmp_path, change)
    catalog = household[0].schema.catalog
    person = catalog.entities['person']
    entities = dict(catalog.entities)
    entities['person'] = EntityTypeSchema('person', (*person.properties, 'score'), {**person.property_types, 'score': 'integer'})
    schema = SemanticSchemaRegistry(replace(catalog, entities=entities), ontology)
    request = count(SemanticFilter(property='score', operator='gte', value=2))
    assert schema.validates(request)
    assert not schema.validates(request.model_copy(update={'filters': (SemanticFilter(property='score', value=True),)}))
    # The same declaration cannot silently attach to a relationship or another type.
    assert schema._semantic_kind(frozenset({'person', 'address'}), 'score') is None
    assert not schema.validates(request.model_copy(update={'filters': (SemanticFilter(property='score', source='relation', value=2),)}))


def test_absent_bindings_disable_complete_concepts_and_bad_kinds_fail(household):
    catalog = household[0].schema.catalog
    entities = dict(catalog.entities)
    person = entities['person']
    entities['person'] = EntityTypeSchema('person', tuple(p for p in person.properties if p != 'gender'),
                                         {p: k for p, k in person.property_types.items() if p != 'gender'})
    schema = SemanticSchemaRegistry(replace(catalog, entities=entities), SemanticOntology.from_file(V2))
    assert 'wife' not in schema.planner_capability_payload()['reference_concepts']
    assert 'spouse' in schema.planner_capability_payload()['reference_concepts']
    with pytest.raises(ValueError, match='unknown reference concept'):
        schema.expand_planner_concepts({'request': {'subject': {'kind': 'self', 'path': [{'concept': 'wife'}]}}})
    entities['person'] = EntityTypeSchema('person', person.properties, {**person.property_types, 'gender': 'integer'})
    with pytest.raises(ValueError, match='Incompatible catalog type'):
        SemanticSchemaRegistry(replace(catalog, entities=entities), SemanticOntology.from_file(V2))


def test_fingerprint_is_stable_across_processes():
    program = "from pathlib import Path; from home_cortex.composition_eval import household_engine; from home_cortex.semantic_facts import SemanticSchemaRegistry; from home_cortex.semantic_ontology import SemanticOntology; print(SemanticSchemaRegistry(household_engine('alpha')[0].schema.catalog, SemanticOntology.from_file(Path('schemas/semantic/ontology-v2.yaml'))).contracts.fingerprint)"
    outputs = [subprocess.check_output([sys.executable, '-c', program], env={**os.environ, 'PYTHONHASHSEED': seed}, text=True) for seed in ('1', '77')]
    assert outputs[0] == outputs[1]


def test_closed_domain_has_no_implicit_ordering(household):
    schema = candidate(household[0]).schema
    for operation in ('argmin', 'argmax', 'min', 'max', 'latest', 'earliest'):
        assert not schema.validates(SemanticFactRequest(operation=operation, subject=members(), property='gender'))


@pytest.mark.asyncio
async def test_duplicate_conditions_and_different_predicate_sites_are_preserved(household):
    engine = candidate(household[0])
    adult = SemanticFilter(predicate='adult')
    request = count(adult, adult, SemanticFilter(property='gender', value='female'))
    before = request.model_dump()
    result, *_ = await engine.execute(request, household[1])
    assert result.status == 'found' and result.value == 2
    assert request.model_dump() == before
    # An intermediate female relative and final male relative are independent sites.
    payload = {'request': {'operation': 'resolve_reference', 'subject': {'kind': 'self', 'path': [{'concept': 'wife'}, {'concept': 'father'}]}}}
    request = SemanticFactRequest.model_validate(engine.schema.expand_planner_concepts(payload)['request'])
    result, *_ = await engine.execute(request, household[1])
    assert result.status == 'found' and result.value['id'] == 'person:father'


@pytest.mark.asyncio
async def test_relationship_values_and_partial_rows_keep_ownership(household):
    engine = candidate(household[0])
    engine.dispatcher.edges['lives_in'][0]['start'] = 'not-a-date'
    relation_filter = SemanticFilter(source='relation', property='start_date', operator='gte', value='2000-01-01')
    result, *_ = await engine.execute(count(relation_filter), household[1])
    assert result.status == 'filter_unsupported'
    request = SemanticFactRequest(operation='select', subject=members(), property='start_date', property_source='relationship', projection='each')
    result, *_ = await engine.execute(request, household[1])
    assert result.status == 'found' and result.shape == 'rows'
    assert any(row.status == 'relation_property_unavailable' for row in result.rows)
    assert any(row.status == 'found' for row in result.rows)


def test_v1_remains_permissive_where_v2_explicitly_tightens(household):
    old = household[0].schema
    new = candidate(household[0]).schema
    request = count(SemanticFilter(property='gender', value=17))
    assert old.validates(request) and not new.validates(request)
    ontology = new.ontology
    assert ontology.properties['full_address'].contract.accepts({'street': 'Example'})
    names = ontology.properties['display_name'].contract
    assert names.accepts('Example') and names.accepts(['Example', '示例'])
    assert not names.accepts(['Example', 7])


@pytest.mark.asyncio
async def test_valid_v2_followup_uses_one_interpreter_call(household):
    from test_semantic_facts import _Interpreter
    engine = candidate(household[0])
    context = replace(household[1], conversation_id='one')
    context = replace(context, discourse=DiscourseContext('one', context.caller_entity_id,
        context.household_id, context.assistant_id, (('person:b',),)))
    request = SemanticFactRequest(operation='select', subject=SemanticReference(
        kind='discourse', entity_type='person', turn_offset=1), property='birth_date')
    interpreter = _Interpreter(request)
    service = SemanticFactService(engine, SemanticFactPlanner(interpreter, engine.schema))
    answer = await service.try_answer([{'role': 'user', 'content': 'Her date of birth?'}], context=context)
    assert answer.result.status == 'found' and answer.result.value == '1982-06-01'
    assert interpreter.calls == 1 and answer.timings.llm_call_count == 1
