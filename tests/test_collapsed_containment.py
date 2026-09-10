"""Containment views over invented graph facts; no interpreter heuristics."""
import copy
import json
from datetime import datetime

import pytest

from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.fact_benchmark import _JsonGraphDispatcher
from home_cortex.schema_catalog import RuntimeSchemaCatalog
from home_cortex.semantic_facts import (
    AgentRequestContext, HouseholdFactEngine, SemanticFactRequest,
    SemanticSchemaRegistry,
)


def graph(tmp_path, collapse=True):
    nodes = {
        'item': [{'id': 'item:cabinet', 'name': ['Cabinet']},
                 *[{'id': f'item:{name}', 'name': [name]} for name in ('cup', 'plate', 'bowl')]],
        'space': [{'id': f'space:{name}', 'name': [name]} for name in ('upper', 'lower', 'drawer')],
    }
    if collapse is not None:
        nodes['item'][0]['collapse'] = collapse
    edges = {
        'hosted_by': [{'from': 'space:upper', 'to': 'item:cabinet'},
                      {'from': 'space:lower', 'to': 'item:cabinet'},
                      {'from': 'space:drawer', 'to': 'space:lower'}],
        'located_in': [{'from': f'item:{item}', 'to': f'space:{space}'}
                       for item, space in [('cup', 'upper'), ('plate', 'lower'), ('bowl', 'drawer')]],
    }
    for folder, tables in [('nodes', nodes), ('edges', edges)]:
        (tmp_path / folder).mkdir()
        for table, records in tables.items():
            (tmp_path / folder / f'{table}.json').write_text(json.dumps(records))
    registry = EdgeSchemaRegistry.load_default(tmp_path)
    dispatcher = _JsonGraphDispatcher(tmp_path, registry)
    schema = SemanticSchemaRegistry(RuntimeSchemaCatalog.from_data_dir(tmp_path, registry))
    return HouseholdFactEngine(dispatcher, schema), dispatcher


async def contents(engine, name='Cabinet', entity_type='item', operation='select'):
    request = SemanticFactRequest.model_validate({
        'subject': {'kind': 'named_entity', 'value': name, 'entity_type': entity_type,
                    'path': [{'relation': 'contents'}]}, 'operation': operation,
    })
    context = AgentRequestContext(caller_entity_id=None, household_id=None,
        assistant_id='steward', assistant_display_name='Steward',
        current_time=datetime(2026, 9, 1), locale='en')
    return (await engine.execute(request, context))[0]


@pytest.mark.asyncio
async def test_recursive_contents_preserve_authoritative_locations(tmp_path):
    engine, dispatcher = graph(tmp_path)
    before = copy.deepcopy((dispatcher.entities, dispatcher.edges))
    result = await contents(engine)
    assert result.status == 'found'
    assert {item['id'] for item in result.value} == {'item:cup', 'item:plate', 'item:bowl'}
    assert {(edge.source_id, edge.target_id) for edge in result.evidence.relationships
            if edge.relation == 'contents'} == {
        ('item:cup', 'space:upper'), ('item:plate', 'space:lower'), ('item:bowl', 'space:drawer')}
    assert (await contents(engine, operation='count')).value == 3
    assert (dispatcher.entities, dispatcher.edges) == before
    assert {call[0] for call in dispatcher.calls} <= {'get_entity', 'get_relationships', 'resolve_entity_alias'}


@pytest.mark.asyncio
@pytest.mark.parametrize('collapse', [False, None])
async def test_opt_in_and_direct_subspace(tmp_path, collapse):
    engine, _ = graph(tmp_path, collapse)
    assert (await contents(engine)).value == []
    result = await contents(engine, 'upper', 'space')
    assert [item['id'] for item in result.value] == ['item:cup']


@pytest.mark.asyncio
async def test_direct_subspace_with_collapsed_parent(tmp_path):
    engine, _ = graph(tmp_path)
    assert [item['id'] for item in (await contents(engine, 'lower', 'space')).value] == ['item:plate']


@pytest.mark.asyncio
async def test_hosting_cycle_fails_closed(tmp_path):
    engine, dispatcher = graph(tmp_path)
    dispatcher.edges['hosted_by'].append({'from': 'space:lower', 'to': 'space:drawer'})
    result = await contents(engine)
    assert result.status == 'computation_impossible'
    assert result.missing_requirements == ('hosting_cycle',)


@pytest.mark.asyncio
async def test_hosted_collection_limit_is_not_silently_truncated(tmp_path):
    engine, _ = graph(tmp_path)
    engine.resolver.max_records = 1
    assert (await contents(engine)).status == 'collection_incomplete'


@pytest.mark.asyncio
@pytest.mark.parametrize('space_count', [1, 3])
async def test_one_or_multiple_direct_hosted_spaces(tmp_path, space_count):
    engine, dispatcher = graph(tmp_path)
    spaces = ['upper', 'lower', 'drawer'][:space_count]
    dispatcher.edges['hosted_by'] = [
        {'from': f'space:{space}', 'to': 'item:cabinet'} for space in spaces]
    result = await contents(engine)
    assert result.status == 'found'
    assert len(result.value) == space_count


@pytest.mark.asyncio
async def test_collapsed_result_composes_with_location_traversal(tmp_path):
    engine, _ = graph(tmp_path)
    request = SemanticFactRequest.model_validate({
        'subject': {'kind': 'named_entity', 'value': 'Cabinet', 'entity_type': 'item',
                    'path': [{'relation': 'contents'}, {'relation': 'location'}]},
        'operation': 'select',
    })
    context = AgentRequestContext(caller_entity_id=None, household_id=None,
        assistant_id='steward', assistant_display_name='Steward',
        current_time=datetime(2026, 9, 1), locale='en')
    result = (await engine.execute(request, context))[0]
    assert result.status == 'found'
    assert {entity['id'] for entity in result.value} == {
        'space:upper', 'space:lower', 'space:drawer'}


@pytest.mark.asyncio
async def test_household_containment_regression_sequence(tmp_path):
    from home_cortex.semantic_facts import FactRenderer

    engine, dispatcher = graph(tmp_path)
    dispatcher.entities['item:cabinet']['name'] = ['冰箱']
    dispatcher.entities['space:upper']['name'] = ['冰箱门架']
    dispatcher.entities['item:cup']['name'] = ['奶酪']
    dispatcher.entities['item:house'] = {'id': 'item:house', 'name': ['家']}
    for index in range(7):
        room_id = f'space:room{index}'
        dispatcher.entities[room_id] = {'id': room_id, 'name': ['厨房' if index == 0 else f'房间{index}']}
        dispatcher.edges['hosted_by'].append({'from': room_id, 'to': 'item:house'})
    dispatcher.edges['located_in'].append({'from': 'item:cabinet', 'to': 'space:room0'})
    context = AgentRequestContext(caller_entity_id=None, household_id=None,
        assistant_id='steward', assistant_display_name='Steward',
        current_time=datetime(2026, 9, 1), locale='zh')
    cases = [
        ('家', 'item', 'hosted_space', 'count', 7),
        ('厨房', 'space', 'contents', 'select', {'item:cabinet'}),
        ('冰箱', 'item', 'contents', 'select', {'item:cup', 'item:plate', 'item:bowl'}),
        ('冰箱门架', 'space', 'contents', 'select', {'item:cup'}),
    ]
    rendered = []
    for name, entity_type, relation, operation, expected in cases:
        request = SemanticFactRequest.model_validate({
            'subject': {'kind': 'named_entity', 'value': name, 'entity_type': entity_type,
                        'path': [{'relation': relation}]}, 'operation': operation,
        })
        result = (await engine.execute(request, context))[0]
        assert result.status == 'found'
        actual = result.value if operation == 'count' else {item['id'] for item in result.value}
        assert actual == expected
        rendered.append(FactRenderer().render(request, result, context))
    assert '7' in rendered[0]
    assert '冰箱' in rendered[1]
    assert '奶酪' in rendered[2]
    assert '奶酪' in rendered[3]


@pytest.mark.asyncio
async def test_ingestion_supports_optional_boolean_and_nested_hosting(tmp_path):
    from surrealdb import AsyncSurreal
    from home_cortex.ingestion import ingest_directory

    graph(tmp_path)
    for relation in ('lives_in', 'parent_of', 'spouse_of'):
        (tmp_path / 'edges' / f'{relation}.json').write_text('[]')
    client = AsyncSurreal('mem://')
    await client.connect()
    await client.use('test', 'collapsed')
    try:
        await ingest_directory(client, tmp_path)
        rows = await client.query('SELECT * FROM item WHERE collapse = true;')
        assert len(rows) == 1
        path = tmp_path / 'nodes' / 'item.json'
        records = json.loads(path.read_text())
        records[0]['collapse'] = 'true'
        path.write_text(json.dumps(records))
        with pytest.raises(ValueError, match='collapse as a boolean'):
            await ingest_directory(client, tmp_path)
    finally:
        await client.close()
