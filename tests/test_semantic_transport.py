"""Synthetic codec checks; no runtime household records or scored utterances."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from home_cortex.semantic_facts import SemanticPlan, SemanticSchemaRegistry
from home_cortex.semantic_transport import (
    SemanticTransport, canonical_json, pack_capabilities, unpack_capabilities,
)
from test_semantic_contract import household
from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.schema_catalog import RuntimeSchemaCatalog


def registry():
    return SemanticSchemaRegistry(RuntimeSchemaCatalog.from_data_dir(
        Path(__file__).parent / 'static_test_data', EdgeSchemaRegistry.load_default()))


def test_all_reusable_demonstrations_roundtrip(household):
    from home_cortex.ollama import _semantic_planner_examples
    schema = household[0].schema
    codec = SemanticTransport(schema.planner_output_schema())
    for message in _semantic_planner_examples():
        if message['role'] != 'assistant':
            continue
        payload = json.loads(message['content'])
        # Demonstrations historically omit eq despite the collection schema
        # requiring it. Supply that explicit canonical default for strict output.
        request = payload.get('request') or {}
        for condition in request.get('filters', []):
            if 'property' in condition:
                condition.setdefault('operator', 'eq')
        text = codec.encode(payload)
        assert codec.decode(text) == payload
        assert codec.decode_plan(text, schema) == SemanticPlan.model_validate(schema.expand_planner_concepts(payload))


@pytest.mark.parametrize('semantic_request', [
    {'operation': 'select', 'subject': {'kind': 'unresolved'}},
    {'operation': 'same_entity', 'subject': {'kind': 'self'}, 'other': {'kind': 'assistant'}},
    {'operation': 'select', 'subject': {'kind': 'named_entity', 'value': 'a|b,\\"雪', 'entity_type': 'item'}, 'projection': 'each', 'exclude': [{'kind': 'self'}]},
    {'operation': 'date_add', 'subject': {'kind': 'discourse', 'entity_type': 'person', 'turn_offset': 3, 'cardinality': 'collection'}, 'amount': -8, 'mode': 'months'},
    {'operation': 'date_difference', 'subject': {'kind': 'self', 'path': [{'relation': 'spouse', 'filters': [{'property': 'birth_date', 'value_from': 'anchor', 'value_property': 'birth_date', 'operator': 'lt'}]}]}, 'property': 'start_date', 'property_source': 'relationship', 'mode': 'years'},
    {'operation': 'count', 'subject': {'kind': 'current_household'}, 'filters': [{'property': 'birth_date', 'transform': 'date_difference', 'mode': 'years', 'operator': 'gte', 'value': 40}, {'predicate': 'adult'}]},
    {'operation': 'select', 'subject': {'kind': 'self'}, 'filters': [{'property': 'gender', 'operator': 'in', 'value': ['male', 'female']}, {'property': 'birth_date', 'operator': 'exists', 'value': False}]},
    {'operation': 'unit_conversion', 'subject': {'kind': 'self'}, 'property': 'height', 'from_unit': 'cm', 'to_unit': 'm'},
])
def test_typed_ir_lossless(semantic_request):
    codec = SemanticTransport(SemanticPlan.model_json_schema())
    plan = SemanticPlan.model_validate({'requires_fact': True, 'request': semantic_request})
    assert codec.decode_plan(codec.encode(plan)) == plan


def test_nonfact_and_explicit_defaults_are_equivalent():
    codec = SemanticTransport(SemanticPlan.model_json_schema())
    plan = SemanticPlan(requires_fact=False)
    assert codec.decode_plan(codec.encode(plan)) == plan
    assert codec.encode(plan) == codec.encode(SemanticPlan(requires_fact=False, request=None))


@pytest.mark.parametrize('mutation', ['version', 'unknown', 'enum', 'duplicate', 'trailing', 'nan', 'fence', 'type', 'expanded', 'extra_slot', 'bool_version', 'overflow'])
def test_strict_rejections(mutation):
    codec = SemanticTransport(SemanticPlan.model_json_schema())
    valid = codec.encode({'requires_fact': False, 'request': None})
    wire = json.loads(valid)
    if mutation == 'version': wire[0] = 999
    if mutation == 'bool_version': wire[0] = True
    if mutation == 'unknown': wire[1][-1]['zz'] = None
    if mutation == 'enum': wire[1][-1][codec.aliases['request']] = ['invented', ['self']]
    if mutation == 'type': wire[1][0] = 'false'
    if mutation == 'extra_slot': wire[1].append(None)
    text = json.dumps(wire)
    if mutation == 'duplicate': text = '[1,[false,{"s":null,"s":null}]]'
    if mutation == 'trailing': text += '{}'
    if mutation == 'nan': text = text.replace('null', 'NaN')
    if mutation == 'overflow': text = text.replace('null', '1e999')
    if mutation == 'fence': text = '```json\n' + text + '\n```'
    if mutation == 'expanded': text = '{"requires_fact":false,"request":null}'
    with pytest.raises(ValueError): codec.decode(text)


def test_unknown_runtime_symbol_and_property_owner_rejected():
    schema = registry()
    codec = SemanticTransport(schema.planner_output_schema())
    payload = {'requires_fact': True, 'request': {'operation': 'select', 'subject': {'kind': 'self'}, 'property': 'invented', 'property_source': 'entity'}}
    with pytest.raises(ValueError): codec.decode(codec.encode(payload, validate=False))
    payload['request']['property'] = None
    payload['request']['property_source'] = 'guessed'
    with pytest.raises(ValueError): codec.decode(codec.encode(payload, validate=False))


def test_capability_tables_preserve_order_and_reserved_literals():
    value = {'nested': {str(i): {'a_long_column': i, 'another_column': [3, 1, 2]} for i in range(10)}, '$table': ['literal'], 'empty': {}, 'null': None}
    packed = pack_capabilities(value)
    assert unpack_capabilities(packed) == value
    assert len(canonical_json(packed)) < len(canonical_json(value))


def test_cross_process_determinism():
    script = '''
from home_cortex.semantic_transport import *
from home_cortex.semantic_facts import SemanticPlan
c=SemanticTransport(SemanticPlan.model_json_schema())
v={k: {'ordered': [3,1,2], 'set': {'b','a'}} for k in {'beta','alpha'}}
print(canonical_json(c.schema))
print(canonical_json(pack_capabilities(v)))
print(c.encode(SemanticPlan(requires_fact=False)))
'''
    outputs = [subprocess.check_output([sys.executable, '-c', script], env={**os.environ, 'PYTHONHASHSEED': seed, 'PYTHONPATH': 'src'}) for seed in ('0', '1', '42', 'random')]
    assert len(set(outputs)) == 1


@pytest.mark.asyncio
async def test_retry_diagnostics_preserve_each_transport_attempt(household):
    from ollama import ChatResponse
    from home_cortex.ollama import OllamaService
    from home_cortex.semantic_facts import SemanticFactPlanner
    engine, context, _ = household
    codec = SemanticTransport(engine.schema.planner_output_schema())
    payload = {'requires_fact': True, 'request': {
        'operation': 'count', 'subject': {'kind': 'current_household', 'path': [{'concept': 'member'}]},
        'property': None, 'property_source': 'entity', 'filters': [{'predicate': 'minor'}],
    }}
    class Client:
        def __init__(self):
            self.responses = ['PRIVATE_INVALID_OUTPUT', codec.encode(payload)]
            self.calls = []
        async def chat(self, **kwargs):
            self.calls.append(kwargs)
            return ChatResponse(message={'role': 'assistant', 'content': self.responses.pop(0)}, prompt_eval_count=111, eval_count=22)
    client = Client()
    # Explicit offline adapter: the serving client must not require this codec.
    from home_cortex.semantic_transport import decode_response
    class OfflineCompactInterpreter:
        last_planner_runtime = {}
        async def plan_semantic_fact(self, messages, capabilities, output_schema, **kwargs):
            response = await client.chat(messages=messages, format=codec.schema)
            self.last_planner_runtime = {'prompt_eval_count': response.prompt_eval_count,
                                         'eval_count': response.eval_count}
            return decode_response(codec, response.message.content, self.last_planner_runtime)
    service = OfflineCompactInterpreter()
    outcome = await SemanticFactPlanner(service, engine.schema).plan(
        [{'role': 'user', 'content': 'Count minors in this home.'}], context)
    assert outcome.plan.request.operation == 'count'
    assert outcome.diagnostics.attempt_count == 2
    attempts = outcome.diagnostics.transport['attempts']
    assert [a['transport']['transport_parse_success'] for a in attempts] == [False, True]
    assert [a['prompt_eval_count'] for a in attempts] == [111, 111]
    assert 'PRIVATE_INVALID_OUTPUT' not in canonical_json(outcome.diagnostics.transport)
    assert 'PRIVATE_INVALID_OUTPUT' not in canonical_json(client.calls[1]['messages'])
    assert client.calls[0]['format'] == codec.schema


def test_optional_tail_and_slot_order_cannot_be_confused(household):
    codec = SemanticTransport(household[0].schema.planner_output_schema())
    payload = {'requires_fact': True, 'request': {
        'operation': 'same_entity', 'subject': {'kind': 'self'}, 'other': {'kind': 'assistant'},
        'property': None, 'property_source': 'entity',
    }}
    text = codec.encode(payload)
    assert codec.decode(text) == payload
    wire = json.loads(text)
    # Request required slots are operation, property, property_source, subject.
    request_wire = wire[1][1][codec.aliases['request']]
    request_wire[1], request_wire[2] = request_wire[2], request_wire[1]
    with pytest.raises(ValueError): codec.decode(canonical_json(wire))


def test_transport_dictionary_version_snapshot():
    from home_cortex.semantic_transport import fingerprint
    codec = SemanticTransport(SemanticPlan.model_json_schema())
    # Changing field allocation requires an intentional codec version change.
    assert (codec.version, fingerprint(codec.aliases)[:8]) == (1, '5ae1d749')


@pytest.mark.parametrize(('filter_wire', 'expected_validation'), [
    ('["eq","display_name",null,{}]', 'INVALID_PLAN'),
    ('["eq","display_name","林青",{"u":"relation"}]', 'UNKNOWN_PROPERTY'),
])
def test_production_identity_counterexamples_are_not_repaired(household, filter_wire, expected_validation):
    # Captured from qwen3.5:9b / Ollama 0.32.15 with the synthetic contract.
    # Both outputs parse; the model invented filters absent from the utterance.
    schema = household[0].schema
    codec = SemanticTransport(schema.planner_output_schema())
    wire = '[1,[true,{"s":["resolve_reference",null,"entity",["assistant",{"d":"person","z":null}],{"f":[' + filter_wire + ']}]}]]'
    plan = codec.decode_plan(wire, schema)
    assert plan.requires_fact is True
    assert plan.request.subject.kind == 'assistant'
    assert len(plan.request.filters) == 1
    assert schema.validation_code(plan.request) == expected_validation
    # Control is a different plan, not a repair performed by the codec/runtime.
    control = SemanticPlan.model_validate({'requires_fact': True, 'request': {
        'operation': 'resolve_reference', 'subject': {'kind': 'assistant'},
        'property': None, 'property_source': 'entity',
    }})
    assert schema.validation_code(control.request) == 'VALID'
