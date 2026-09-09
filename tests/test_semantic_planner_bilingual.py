"""Language-invariant planner contract: Chinese, English, and mixed utterances."""
from __future__ import annotations

import json
import re
from pathlib import Path

from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.ollama import _PLANNER_INSTRUCTIONS, _semantic_planner_examples
from home_cortex.schema_catalog import RuntimeSchemaCatalog
from home_cortex.semantic_facts import SemanticFactRequest, SemanticSchemaRegistry
from home_cortex.semantic_planner_benchmark import (
    load_bilingual_dataset,
    load_probe_dataset,
    load_semantic_eval_cases,
    normalize_semantic_request,
)


ROOT = Path(__file__).parents[1]
CJK = re.compile(r"[\u4e00-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")


def _schema() -> SemanticSchemaRegistry:
    return SemanticSchemaRegistry(
        RuntimeSchemaCatalog.from_data_dir(
            ROOT / "tests" / "static_test_data",
            EdgeSchemaRegistry.load_default(),
        )
    )


def _example_users() -> list[str]:
    return [
        message["content"]
        for message in _semantic_planner_examples()
        if message["role"] == "user"
    ]


def _example_payloads() -> list[tuple[str, dict]]:
    messages = _semantic_planner_examples()
    rows = []
    for index, message in enumerate(messages):
        if message["role"] != "user":
            continue
        rows.append((message["content"], json.loads(messages[index + 1]["content"])))
    return rows


def _classify(text: str) -> str:
    has_cjk = bool(CJK.search(text))
    has_latin = bool(LATIN.search(text))
    if has_cjk and has_latin:
        return "mixed"
    if has_cjk:
        return "zh"
    if has_latin:
        return "en"
    return "other"


def _strip_named_values(payload: dict) -> dict:
    def walk(value):
        if isinstance(value, dict):
            cleaned = {
                key: walk(item)
                for key, item in value.items()
                if not (key == "value" and value.get("kind") == "named_entity")
            }
            return cleaned
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    return walk(payload)


def _canonical(schema: SemanticSchemaRegistry, request: dict, *, ignore_named_value: bool = False):
    expanded = schema.expand_planner_concepts({"requires_fact": True, "request": request})
    normalized = normalize_semantic_request(
        SemanticFactRequest.model_validate(expanded["request"])
    )
    return _strip_named_values(normalized) if ignore_named_value else normalized


def test_planner_instructions_are_english_primary_with_chinese_lexicon():
    latin = len(LATIN.findall(_PLANNER_INSTRUCTIONS))
    cjk = len(CJK.findall(_PLANNER_INSTRUCTIONS))
    assert latin > cjk * 3
    for snippet in (
        "Surface language must not change the IR",
        "Do not translate the utterance as an intermediate step",
        "kind=self",
        "kind=assistant",
        "current_household",
        "father_in_law",
        "named_entity.value",
        "projection=each",
        "date_add",
        "annual_occurrence",
        "date_difference",
        "argmin",
        "same_entity",
        "property_source",
        "我 / 我的",
        "你 / 您",
        "以上",
        "以下",
        "满 N 岁",
        "未满 N 岁",
        "岳父",
        "公公",
        "丈夫",
        "妻子",
        "儿子",
        "女儿",
        "林青 stays 林青",
        "son ≠ child",
        "wife ≠ spouse",
        "father_in_law ≠ parent",
    ):
        assert snippet in _PLANNER_INSTRUCTIONS


def test_few_shot_set_is_bilingual_without_eval_wording():
    users = _example_users()
    counts = {"zh": 0, "en": 0, "mixed": 0, "other": 0}
    for utterance in users:
        counts[_classify(utterance)] += 1
    total = len(users)
    assert total <= 42
    assert counts["zh"] >= 8
    assert counts["en"] >= 8
    assert counts["mixed"] >= 5
    eval_utterances = {case.utterance for case in load_semantic_eval_cases()}
    assert users
    assert eval_utterances.isdisjoint(users)
    for name in (
        "semantic_planner_heldout.yaml",
        "semantic_planner_synthetic.yaml",
        "semantic_planner_age_filters.yaml",
        "semantic_planner_date_intervals.yaml",
    ):
        held = {case.utterance for case in load_probe_dataset(ROOT / "benchmarks" / name).cases}
        assert held.isdisjoint(users)


def test_examples_cover_required_concepts_and_preserve_literals():
    payloads = _example_payloads()
    users = [utterance for utterance, _ in payloads]
    assert "请介绍一下你自己。" in users
    assert "Who is the authenticated speaker?" in users
    assert "Who belongs to my household?" in users
    assert "Where is 收纳盒?" in users
    assert "我 son 几岁了？" in users
    assert "Who is 我岳父?" in users
    assert "我 wife 的 birthday 是哪天？" in users
    by_user = dict(payloads)
    son = by_user["我 son 几岁了？"]["request"]
    assert son["operation"] == "date_difference"
    assert [step["concept"] for step in son["subject"]["path"]] == ["son"]
    assert son["property"] == "birth_date"
    assert son["mode"] == "years"
    father = by_user["Who is 我岳父?"]["request"]
    assert father["operation"] == "resolve_reference"
    assert [step["concept"] for step in father["subject"]["path"]] == ["father_in_law"]
    wife = by_user["我 wife 的 birthday 是哪天？"]["request"]
    assert wife["operation"] == "select"
    assert [step["concept"] for step in wife["subject"]["path"]] == ["wife"]
    assert wife["property"] == "birth_date"
    named = by_user["How many days until 林青's next birthday?"]["request"]
    assert named["subject"]["value"] == "林青"
    charger = by_user["Where is 爸爸's charger?"]["request"]
    assert charger["subject"]["value"] == "爸爸's charger"
    assert charger["subject"]["entity_type"] == "item"
    chat = by_user["Just chatting, no household question."]
    assert chat == {"requires_fact": False, "request": None}


def test_cross_language_discourse_examples_use_turn_offset():
    payloads = _example_payloads()
    by_user = dict(payloads)
    first = by_user["周岚现在多大？"]["request"]
    assert first["subject"]["kind"] == "named_entity"
    assert first["subject"]["value"] == "周岚"
    follow = by_user["When was his 20th birthday?"]["request"]
    assert follow["operation"] == "date_add"
    assert follow["subject"] == {
        "kind": "discourse",
        "entity_type": "person",
        "turn_offset": 1,
        "cardinality": "single",
    }
    assert follow["amount"] == 20
    named = by_user["How old is Zhou Lan?"]["request"]
    assert named["subject"]["value"] == "Zhou Lan"
    second = by_user["他的二十岁生日是哪天？"]["request"]
    assert second["subject"]["kind"] == "discourse"
    assert second["subject"]["turn_offset"] == 1


def test_required_parity_pairs_share_normalized_ir():
    schema = _schema()
    dataset = load_bilingual_dataset()
    required = {
        "self_identity",
        "assistant_identity",
        "household_count",
        "household_list",
        "father_in_law",
        "son_age",
        "oldest_member",
        "age_under_30",
        "storage_box",
        "linqing_age",
    }
    seen = set()
    for pair in dataset["pairs"]:
        seen.add(pair["id"])
        ignore = bool(pair.get("ignore_named_value"))
        expected = _canonical(schema, pair["expected"], ignore_named_value=ignore)
        assert pair["zh"] and pair["en"]
        if pair["id"] == "self_identity":
            assert expected["operation"] == "resolve_reference"
            assert expected["subject"]["kind"] == "self"
        if pair["id"] == "assistant_identity":
            assert expected["subject"]["kind"] == "assistant"
        if pair["id"] == "household_count":
            assert expected["operation"] == "count"
            assert expected["subject"]["kind"] == "current_household"
            assert [step["relation"] for step in expected["subject"]["path"]] == ["member"]
        if pair["id"] == "son_age":
            assert expected["operation"] == "date_difference"
            assert expected["property"] == "birth_date"
            relations = [step.get("relation") for step in expected["subject"]["path"]]
            assert relations == ["child"]
            assert any(
                item.get("value") == "male"
                for item in expected["subject"]["path"][0].get("filters") or ()
            )
        if pair["id"] == "age_under_30":
            condition = expected["filters"][0]
            assert condition["property"] == "birth_date"
            assert condition["transform"] == "date_difference"
            assert condition["mode"] == "years"
            assert condition["operator"] == "lt"
            assert condition["value"] == 30
        if pair["id"] == "linqing_age":
            assert expected["subject"]["value"] == "林青"
        # Chinese and English share one expected plan by construction.
        assert _canonical(schema, pair["expected"], ignore_named_value=ignore) == expected
    assert required <= seen


def test_mixed_utterances_use_the_same_canonical_concepts():
    schema = _schema()
    dataset = load_bilingual_dataset()
    plans = {pair["id"]: pair["expected"] for pair in dataset["pairs"]}
    by_id = {item["id"]: item for item in dataset["mixed"]}
    for key, pair_id in (
        ("son_age_mixed", "son_age"),
        ("father_in_law_mixed", "father_in_law"),
    ):
        mixed = by_id[key]
        expected = mixed.get("expected") or plans[mixed["equivalent_to"]]
        assert mixed["equivalent_to"] == pair_id
        canonical = _canonical(schema, expected)
        monolingual = _canonical(schema, plans[pair_id])
        assert canonical == monolingual
    wife = _canonical(schema, by_id["wife_birthday_mixed"]["expected"])
    assert [step["relation"] for step in wife["subject"]["path"]] == ["spouse"]
    assert any(
        item.get("value") == "female"
        for step in wife["subject"]["path"]
        for item in step.get("filters") or ()
    )


def test_discourse_contract_does_not_guess_names():
    schema = _schema()
    dataset = load_bilingual_dataset()
    for sequence in dataset["discourse"]:
        first, second = sequence["turns"]
        lead = _canonical(schema, first["expected"])
        follow = _canonical(schema, second["expected"])
        assert lead["subject"]["kind"] == "named_entity"
        assert follow["subject"]["kind"] == "discourse"
        assert follow["subject"]["turn_offset"] == 1
        assert follow["subject"]["cardinality"] == "single"
        assert follow["subject"].get("value") is None
        assert follow["operation"] == "date_add"
        assert follow["amount"] == 20


def test_negative_bilingual_contract_forbids_speaker_and_path_regressions():
    schema = _schema()
    dataset = load_bilingual_dataset()
    by_id = {item["id"]: item for item in dataset["negatives"]}
    who = _canonical(schema, {"operation": "resolve_reference", "subject": {"kind": "self", "entity_type": "person"}})
    assert who["subject"]["kind"] != by_id["who_am_i_not_assistant"]["forbidden_subject_kind"]
    household = _canonical(schema, by_id["household_not_self_member"]["expected"])
    assert household["subject"]["kind"] == "current_household"
    assert [step["relation"] for step in household["subject"]["path"]] == ["member"]
    son = _canonical(schema, by_id["son_not_child"]["expected"])
    assert [step["relation"] for step in son["subject"]["path"]] == ["child"]
    assert any(
        item.get("value") == "male"
        for step in son["subject"]["path"]
        for item in step.get("filters") or ()
    )
    wife = _canonical(schema, by_id["wife_not_spouse"]["expected"])
    assert any(
        item.get("value") == "female"
        for step in wife["subject"]["path"]
        for item in step.get("filters") or ()
    )
    father = _canonical(schema, by_id["father_in_law_not_parent"]["expected"])
    relations = [step["relation"] for step in father["subject"]["path"]]
    assert relations == ["spouse", "parent"]
