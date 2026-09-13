"""Prompt accounting and experiment isolation, independent of model behavior."""
import pytest

from home_cortex import ollama as prompts
from scripts.profiling.planner_prompt_audit import prompt_components
from scripts.benchmarks.planner_prompt_experiment import reduced_examples


@pytest.mark.parametrize('messages', [
    [],
    [{'role': 'user', 'content': '我是谁'}],
    [{'role': 'user', 'content': 'First question'},
     {'role': 'assistant', 'content': 'This answer must never enter the planner'},
     {'role': 'user', 'content': '她呢？'},
     {'role': 'system', 'content': 'Retry validation failed'}],
])
def test_components_partition_actual_message_content(messages):
    parts, built = prompt_components(messages, {'properties': {}}, household_now='2026-09-12')
    assert sum(len(value.encode()) for value in parts.values()) == sum(len(m['content'].encode()) for m in built)
    assert 'This answer must never enter the planner' not in ''.join(parts.values())
    assert parts['user_input'] == next((m['content'] for m in reversed(messages) if m['role'] == 'user'), '')
    if len(messages) > 1:
        assert 'First question' in parts['history']
        assert prompts._PLANNER_HISTORY_BOUNDARY in parts['history']
        assert 'Retry validation failed' in parts['supplemental_notes_including_retry']


def test_candidate_restores_original_examples_even_after_failure():
    original = prompts._semantic_planner_examples()
    with pytest.raises(RuntimeError):
        with reduced_examples({6, 34}):
            assert len(prompts._semantic_planner_examples()) == len(original) - 4
            assert prompts._semantic_planner_examples()[-12:] == original[-12:]
            raise RuntimeError('benchmark interrupted')
    assert prompts._semantic_planner_examples() == original


def test_reject_out_of_range_candidate():
    with pytest.raises(ValueError):
        with reduced_examples({999}):
            pass


def test_performance_budget_is_distinct_from_context_safety():
    from scripts.benchmarks.planner_prompt_experiment import budget_checks
    rows = [{'planner_attempt_count': 1, 'model_calls': [
        {'prompt_tokens': 8600, 'output_tokens': 50, 'done_reason': 'stop'},
    ]}]
    checks = budget_checks(rows, 8500)
    assert checks['normal_exceeded'] == 1
    assert checks['context_exceeded'] == 0
    rows[0]['model_calls'][0]['prompt_tokens'] = prompts.OLLAMA_NUM_CTX - prompts.PLANNER_NUM_PREDICT + 1
    assert budget_checks(rows, 8500)['context_exceeded'] == 1
    assert not budget_checks([], 8500)['all_counts_available']


def test_normal_fixture_prompt_byte_budget():
    # Offline creep alarm, not a claim that bytes equal native tokens. The live
    # benchmark separately enforces 8,500 measured tokens on the deployed model.
    from scripts.benchmarks.semantic_planner_benchmark import build_json_fact_service
    from scripts import PROJECT_ROOT
    service, context = build_json_fact_service(PROJECT_ROOT / 'benchmarks/fixtures/semantic-contract', PROJECT_ROOT / 'schemas/edge', None)
    parts, _ = prompt_components([{'role': 'user', 'content': '家里有几个人'}], service.engine.schema.planner_capability_payload(), household_now=context.current_time.isoformat())
    assert sum(len(v.encode()) for v in parts.values()) <= 32000


def test_fixed_acceptance_cases_expand_through_canonical_ontology():
    from scripts.benchmarks.planner_prompt_experiment import fixed_cases
    from scripts.benchmarks.semantic_planner_benchmark import build_json_fact_service
    from scripts import PROJECT_ROOT
    service, _ = build_json_fact_service(PROJECT_ROOT / 'tests/static_test_data', PROJECT_ROOT / 'schemas/edge', None)
    cases = fixed_cases(PROJECT_ROOT / 'benchmarks/planner_prompt_compression.yaml', service.engine.schema)
    assert len(cases) == 12
    assert cases[-1].expected.subject.kind == 'assistant'
    assert cases[-1].expected.property == 'birth_date'
    assert cases[8].expected.subject.path  # Complete father-in-law expansion.


def test_comparison_detects_case_regression_despite_equal_aggregate():
    from scripts.profiling.compare_prompt_experiments import compare
    import copy
    before = {'cases': 2, 'repeat': 1, 'evaluation_cases_sha256': 'fixed',
              'provenance': {key: 'same' for key in ('data_tree_sha256', 'schema_tree_sha256', 'model_digest', 'ollama_version', 'request_settings')},
              'metrics': {'prompt_tokens': {'p50': 8000}},
              'failures': [{'case_id': 'a', 'sample_index': 0, 'plan_match': False, 'answer_correct': False}]}
    after = copy.deepcopy(before)
    after['failures'][0]['case_id'] = 'b'
    report = compare(before, after)
    assert report['semantic_regression']
    assert report['plan_match']['regressions'] == [('b', 0)]
    after['provenance']['model_digest'] = 'different'
    with pytest.raises(ValueError, match='model_digest'):
        compare(before, after)
