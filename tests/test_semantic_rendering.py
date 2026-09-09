"""Renderer fidelity against executed plans on invented household data."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from home_cortex.schema_catalog import RuntimeSchemaCatalog
from home_cortex.semantic_display import SemanticDisplay
from home_cortex.semantic_facts import (
    FactRenderer, FactResult, HouseholdFactEngine, SemanticFactPlanner,
    SemanticFactRequest, SemanticFactService, SemanticFilter,
    SemanticReference, SemanticRelationStep, SemanticSchemaRegistry,
)
from home_cortex.semantic_ontology import SemanticOntology
from test_semantic_contract import household, ref, step

ONTOLOGY = Path(__file__).parents[1] / 'schemas/semantic/ontology.yaml'


def members():
    return SemanticReference(kind='current_household', path=(SemanticRelationStep(relation='member'),))


def female():
    return SemanticFilter(property='gender', value='female')


@pytest.mark.asyncio
@pytest.mark.parametrize('locale', ['en', 'zh-CN'])
@pytest.mark.parametrize('predicates,expected', [((), 3), (('adult',), 2), (('minor',), 1)])
async def test_count_retains_every_condition_without_changing_result(household, locale, predicates, expected):
    engine, context, _ = household
    request = SemanticFactRequest(operation='count', subject=members(), filters=(
        *(SemanticFilter(predicate=p) for p in predicates), female(),
    ))
    assert engine.schema.validates(request)
    result, *_ = await engine.execute(request, context)
    before = copy.deepcopy((request, result))
    assert result.status == 'found' and result.value == expected
    text = FactRenderer(engine.schema.ontology, detailed=True).render(request, result, replace(context, locale=locale))
    assert ('性别 = 女性' if locale.startswith('zh') else 'gender = female') in text
    assert ('当前家庭' if locale.startswith('zh') else 'current household') in text
    assert str(expected) in text
    for predicate, label in [('adult', '成年人'), ('minor', '未成年人')]:
        if predicate in predicates:
            assert (label if locale.startswith('zh') else predicate) in text
    if not predicates:
        assert '成年' not in text and 'adult' not in text and 'minor' not in text
    assert (request, result) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('predicates,expected', [
    ((), '家里有3位女性。'),
    (('adult',), '家里有2位成年女性。'),
    (('minor',), '家里有1位未成年女性。'),
])
async def test_normal_count_composes_ontology_modifiers(household, predicates, expected):
    engine, context, _ = household
    request = SemanticFactRequest(operation='count', subject=members(), filters=(
        *(SemanticFilter(predicate=p) for p in predicates), female(),
    ))
    result, *_ = await engine.execute(request, context)
    text = FactRenderer(engine.schema.ontology).render(
        request, result, replace(context, locale='zh-CN')
    )
    assert text == expected
    assert '查询范围' not in text


@pytest.mark.asyncio
async def test_normal_household_member_count_uses_plain_language(household):
    engine, context, _ = household
    request = SemanticFactRequest(operation='count', subject=members())
    result, *_ = await engine.execute(request, context)
    assert FactRenderer().render(
        request, result, replace(context, locale='zh')
    ) == '家里有8个人。'


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['count', 'select'])
async def test_zero_retains_combined_conditions_and_bounds(household, operation):
    engine, context, _ = household
    request = SemanticFactRequest(operation=operation, subject=members(), filters=(
        SemanticFilter(predicate='adult'), female(),
        SemanticFilter(property='birth_date', operator='date_range', value=('1800-01-01', '1801-01-01')),
    ))
    result, *_ = await engine.execute(request, context)
    assert result.status == 'found' and result.value == (0 if operation == 'count' else [])
    for locale in ('zh', 'en'):
        text = FactRenderer(detailed=True).render(request, result, replace(context, locale=locale))
        assert '1800-01-01' in text and '1801-01-01' in text
        assert ('成年人' if locale == 'zh' else 'adult') in text
        assert ('性别 = 女性' if locale == 'zh' else 'gender = female') in text
        assert ('不含终点' if locale == 'zh' else 'end exclusive') in text


@pytest.mark.asyncio
async def test_nested_traversal_keeps_intermediate_filter_separate(household):
    engine, context, _ = household
    request = SemanticFactRequest(operation='resolve_reference', subject=ref(
        step('spouse', 'female'), step('parent', 'male'),
    ))
    result, *_ = await engine.execute(request, context)
    assert result.status == 'found' and result.value['id'] == 'person:father'
    text = FactRenderer(detailed=True).render(request, result, context)
    assert '1. spouse {entity.gender = female} → 2. parent {entity.gender = male}' in text
    assert 'result filters' not in text
    assert 'person:father' not in text


@pytest.mark.asyncio
async def test_relation_conditions_distinguish_any_edge_from_same_edge(household):
    engine, context, dispatcher = household
    dispatcher.edges['lives_in'].append({'from': 'person:son1', 'to': 'address:fictional', 'start': '2020-01-01'})
    filters = tuple(SemanticFilter(source='relation', property='start_date', value=value)
                    for value in ('2018-01-01', '2020-01-01'))
    count = SemanticFactRequest(operation='count', subject=members(), filters=filters)
    counted, *_ = await engine.execute(count, context)
    assert counted.status == 'found' and counted.value == 1
    text = FactRenderer(detailed=True).render(count, counted, context)
    assert text.count('at least one associated relationship.start date') == 2
    each = count.model_copy(update={'operation': 'select', 'property': 'start_date',
                                    'property_source': 'relationship', 'projection': 'each'})
    projected, *_ = await engine.execute(each, context)
    assert projected.status == 'found' and projected.shape == 'rows' and projected.rows == ()
    text = FactRenderer(detailed=True).render(each, projected, context)
    assert 'same record' in text and 'at least one' not in text
    assert '2018-01-01' in text and '2020-01-01' in text and 'No matching records' in text


@pytest.mark.asyncio
async def test_exclusion_retains_its_own_path_filters(household):
    engine, context, _ = household
    request = SemanticFactRequest(operation='count', subject=members(), filters=(female(),),
                                  exclude=(ref(step('spouse', 'female')),))
    result, *_ = await engine.execute(request, context)
    assert result.status == 'found' and result.value == 2
    text = FactRenderer(detailed=True).render(request, result, context)
    assert 'result filters: entity.gender = female; exclude entities: (you → 1. spouse {entity.gender = female})' in text


@pytest.mark.asyncio
async def test_partial_projection_describes_conditions_once(household):
    engine, context, dispatcher = household
    dispatcher.entities['person:daughter'].pop('dob')
    request = SemanticFactRequest(operation='date_difference', subject=members(), property='birth_date',
                                  mode='years', projection='each', filters=(female(),))
    result, *_ = await engine.execute(request, context)
    assert result.status == 'found' and len(result.rows) == 3
    text = FactRenderer(detailed=True).render(request, result, context)
    assert text.count('gender = female') == 1
    assert 'daughter:' in text and 'unavailable' in text
    assert 'b: The age is 44 years.' in text


@pytest.mark.asyncio
async def test_new_property_and_translation_are_declarative_and_service_uses_them(household, tmp_path):
    engine, context, dispatcher = household
    raw = yaml.safe_load(ONTOLOGY.read_text())
    raw['properties']['preferred_language'] = {
        'fields': ['language'], 'aliases': [], 'label': {'en': 'preferred language', 'zh': '首选语言'},
        'value_labels': {'ja': {'en': 'Japanese', 'zh': '日语'}},
    }
    path = tmp_path / 'custom-ontology.yaml'
    path.write_text(yaml.safe_dump(raw, allow_unicode=True))
    ontology = SemanticOntology.from_file(path)
    people_path = tmp_path / 'nodes/person.json'
    people = json.loads(people_path.read_text())
    for person in people:
        person['language'] = 'ja' if person['id'] == 'person:b' else 'en'
        dispatcher.entities[person['id']]['language'] = person['language']
    people_path.write_text(json.dumps(people))
    schema = SemanticSchemaRegistry(RuntimeSchemaCatalog.from_data_dir(tmp_path, engine.schema.edge_registry), ontology)
    engine = HouseholdFactEngine(dispatcher, schema)
    service = SemanticFactService(engine, SemanticFactPlanner(None, schema))
    request = SemanticFactRequest(operation='count', subject=members(), filters=(
        female(), SemanticFilter(property='preferred_language', value='ja'),
    ))
    answer = await service.answer_request(request, context=replace(context, locale='zh-CN'))
    assert answer.result.status == 'found' and answer.result.value == 1
    assert answer.text == '家里有1位女性，筛选条件还包括：实体.首选语言 = 日语。'
    assert answer.timings.llm_call_count == 0


def test_missing_metadata_falls_back_without_reinterpreting_values():
    ontology = SemanticOntology.load_default()
    display = SemanticDisplay(ontology, 'fr-CA')
    assert display.condition(female(), 'you', collection=True) == 'entity.gender = female'
    unknown = SemanticFilter(property='novel', operator='ne', value='unexpected')
    assert display.condition(unknown, 'you', collection=True) == 'entity.novel ≠ "unexpected"'
    unknown_predicate = SemanticFilter(predicate='novel')
    assert display.condition(unknown_predicate, 'you', collection=True) == 'novel'
    assert display.condition(SemanticFilter(property='gender', value='女性'), 'you', collection=True) == 'entity.gender = "女性"'


@pytest.mark.parametrize('operator,value,expected', [
    ('eq', 'female', '= female'), ('ne', 'female', '≠ female'),
    ('gt', 3, '> 3'), ('gte', 3, '≥ 3'), ('lt', 3, '< 3'), ('lte', 3, '≤ 3'),
    ('in', ('female', 'male'), '∈ [female, male]'),
    ('exists', False, 'exists(false)'), ('exists', None, 'exists(null)'),
])
def test_filter_operator_grammar(operator, value, expected):
    display = SemanticDisplay(SemanticOntology.load_default(), 'en')
    assert expected in display.condition(SemanticFilter(property='gender', operator=operator, value=value), 'you', collection=True)


@pytest.mark.asyncio
async def test_anchor_comparison_keeps_original_root(household):
    engine, context, _ = household
    request = SemanticFactRequest(operation='count', subject=ref(
        step('spouse'), step('child'),
        {'relation': 'parent', 'filters': [{'property': 'birth_date', 'operator': 'lt', 'value_from': 'anchor'}]},
    ))
    assert engine.schema.validates(request)
    result, *_ = await engine.execute(request, context)
    text = FactRenderer(detailed=True).render(request, result, context)
    assert '3. parent {entity.birth date < anchor(you).birth date}' in text


@pytest.mark.asyncio
async def test_pairwise_filter_keeps_other_operand_and_no_age_for_other_property(household):
    engine, context, _ = household
    request = SemanticFactRequest(operation='argmin', subject=ref(step('spouse', 'female')),
                                  other=ref(step('child', 'female')), property='birth_date')
    result, *_ = await engine.execute(request, context)
    assert result.status == 'found'
    text = FactRenderer(detailed=True).render(request, result, context)
    assert 'compare with: (you → 1. child {entity.gender = female})' in text
    assert 'older' in text
    # Rendering an already-executed ordered comparison must not invent an age claim.
    generic = request.model_copy(update={'property': 'score'})
    text = FactRenderer(detailed=True).render(generic, result, context)
    assert 'older' not in text and 'age' not in text and 'result entity.score' in text


def test_negative_gender_condition_does_not_acquire_wife_label(household):
    _, context, _ = household
    request = SemanticFactRequest(operation='resolve_reference', subject=ref(
        {'relation': 'spouse', 'filters': [{'property': 'gender', 'operator': 'ne', 'value': 'female'}]},
    ))
    text = FactRenderer(detailed=True).render(request, FactResult('found', {'name': 'Example'}), replace(context, locale='zh'))
    assert '性别 ≠ 女性' in text and '妻子' not in text

    normal = FactRenderer().render(
        request,
        FactResult('found', {'name': 'Example'}),
        replace(context, locale='zh'),
    )
    assert normal == '您配偶是Example。（条件：实体.性别 ≠ 女性）'


def test_plain_text_labels_and_literals_cannot_create_markup():
    display = SemanticDisplay(SemanticOntology.load_default(), 'en')
    text = display.condition(SemanticFilter(property='novel', value='<script>\n[link](url)'), 'you', collection=True)
    assert '<script>' not in text and '\\[link\\]' in text and '\n' not in text


@pytest.mark.asyncio
async def test_concept_expansion_preserves_repeated_hops_and_all_filters(household):
    engine, context, _ = household
    payload = {'requires_fact': True, 'request': {'operation': 'count', 'subject': {
        'kind': 'self', 'path': [{'concept': 'wife'}, {'concept': 'father_in_law'}],
    }}}
    request = SemanticFactRequest.model_validate(engine.schema.expand_planner_concepts(payload)['request'])
    assert engine.schema.validates(request)
    result, *_ = await engine.execute(request, context)
    text = FactRenderer(detailed=True).render(request, result, context)
    assert '1. spouse {entity.gender = female} → 2. spouse → 3. parent {entity.gender = male}' in text
    assert payload['request']['subject']['path'] == [{'concept': 'wife'}, {'concept': 'father_in_law'}]


@pytest.mark.asyncio
async def test_traversal_edge_and_entity_conditions_have_separate_owners(household):
    engine, context, _ = household
    request = SemanticFactRequest(operation='count', subject=SemanticReference(
        kind='current_household', path=(SemanticRelationStep(relation='member', filters=(
            SemanticFilter(source='relation', property='start_date', value='2018-01-01'), female(),
        )),),
    ))
    result, *_ = await engine.execute(request, context)
    assert result.status == 'found' and result.value == 3
    text = FactRenderer(detailed=True).render(request, result, context)
    assert '1. household member {this relationship.start date = "2018-01-01" AND entity.gender = female}' in text
    assert 'at least one' not in text
