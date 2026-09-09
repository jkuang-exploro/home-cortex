#!/usr/bin/env python3
"""Emit Ticket 4 households and YAML from the approved coverage matrix."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path

import yaml

from home_cortex.composition_eval import (
    COMPOSITION_ROOT,
    HOUSEHOLD_IDS,
    HOUSEHOLD_ROOT,
    composition_fingerprint_payload,
    dump_json,
    household_engine,
    request_context,
)
from home_cortex.semantic_facts import DiscourseContext, SemanticFactRequest
from home_cortex.semantic_planner_benchmark import FROZEN_EVAL_TIME, primary_entity_ids

ROOT = Path(__file__).resolve().parents[2]


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):  # noqa: ANN001
        return True


def dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(
        payload,
        Dumper=NoAliasDumper,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=88,
    )
    path.write_text(text, encoding="utf-8")


def person(pid: str, name: str, gender: str, dob: str | None) -> dict:
    row = {"id": pid, "name": name, "gender": gender}
    if dob is not None:
        row["dob"] = dob
    return row


def write_household(
    name: str,
    people: list[dict],
    residents: list[str],
    spouses: list[dict],
    parents: list[tuple[str, str]],
) -> None:
    directory = HOUSEHOLD_ROOT / name
    (directory / "nodes").mkdir(parents=True, exist_ok=True)
    (directory / "edges").mkdir(parents=True, exist_ok=True)
    address_id = HOUSEHOLD_IDS[name]
    (directory / "nodes" / "person.json").write_text(
        json.dumps(people, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "nodes" / "address.json").write_text(
        json.dumps(
            [{"id": address_id, "full_address": f"Invented {name} home"}],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "edges" / "lives_in.json").write_text(
        json.dumps(
            [
                {"from": pid, "to": address_id, "start": "2018-01-01", "end": None}
                for pid in residents
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "edges" / "spouse_of.json").write_text(
        json.dumps(spouses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "edges" / "parent_of.json").write_text(
        json.dumps([{"from": parent, "to": child} for parent, child in parents], indent=2)
        + "\n",
        encoding="utf-8",
    )


def write_households() -> None:
    write_household(
        "alpha",
        [
            person("person:alpha_self", "安乔", "male", "1980-01-15"),
            person("person:alpha_wife", "安梅", "female", "1982-04-20"),
            person("person:alpha_son_adult", "安晨", "male", "2006-03-10"),
            person("person:alpha_son_minor", "安野", "male", "2014-07-22"),
            person("person:alpha_daughter", "安秋", "female", "2012-11-05"),
            person("person:alpha_father", "安石", "male", "1952-08-08"),
            person("person:alpha_mother", "安兰", "female", "1954-09-09"),
            person("person:alpha_fil", "梅山", "male", "1950-02-14"),
            person("person:alpha_mil", "梅竹", "female", "1953-06-18"),
            person("person:alpha_niece", "安禾", "female", "2015-05-30"),
            person("person:alpha_sister", "安芳", "female", "1985-08-08"),
        ],
        [
            "person:alpha_self",
            "person:alpha_wife",
            "person:alpha_son_adult",
            "person:alpha_son_minor",
            "person:alpha_daughter",
            "person:alpha_father",
            "person:alpha_mother",
            "person:alpha_fil",
            "person:alpha_mil",
            "person:alpha_niece",
        ],
        [
            {"from": "person:alpha_self", "to": "person:alpha_wife", "start": "2004-06-18", "end": None},
            {"from": "person:alpha_father", "to": "person:alpha_mother", "start": "1978-01-01", "end": None},
            {"from": "person:alpha_fil", "to": "person:alpha_mil", "start": "1975-05-05", "end": None},
        ],
        [
            ("person:alpha_self", "person:alpha_son_adult"),
            ("person:alpha_wife", "person:alpha_son_adult"),
            ("person:alpha_self", "person:alpha_son_minor"),
            ("person:alpha_wife", "person:alpha_son_minor"),
            ("person:alpha_self", "person:alpha_daughter"),
            ("person:alpha_wife", "person:alpha_daughter"),
            ("person:alpha_father", "person:alpha_self"),
            ("person:alpha_mother", "person:alpha_self"),
            ("person:alpha_fil", "person:alpha_wife"),
            ("person:alpha_mil", "person:alpha_wife"),
            ("person:alpha_sister", "person:alpha_niece"),
        ],
    )
    write_household(
        "beta",
        [
            person("person:beta_self", "林夏", "female", "1983-02-11"),
            person("person:beta_husband", "林海", "male", "1981-07-19"),
            person("person:beta_dau_adult", "林晴", "female", "2005-12-01"),
            person("person:beta_dau_minor", "林露", "female", "2013-08-16"),
            person("person:beta_son_minor", "林川", "male", "2016-02-02"),
            person("person:beta_father", "夏松", "male", "1955-03-03"),
            person("person:beta_mother", "夏荷", "female", "1958-11-11"),
            person("person:beta_fil", "海峰", "male", "1949-12-25"),
            person("person:beta_mil", "海萍", "female", "1951-04-04"),
            person("person:beta_nephew", "林岩", "male", "2018-06-06"),
            person("person:beta_brother", "夏柏", "male", "1986-01-01"),
        ],
        [
            "person:beta_self",
            "person:beta_husband",
            "person:beta_dau_adult",
            "person:beta_dau_minor",
            "person:beta_son_minor",
            "person:beta_father",
            "person:beta_mother",
            "person:beta_fil",
            "person:beta_mil",
            "person:beta_nephew",
        ],
        [
            {"from": "person:beta_self", "to": "person:beta_husband", "start": "2003-11-02", "end": None},
            {"from": "person:beta_father", "to": "person:beta_mother", "start": "1980-02-02", "end": None},
            {"from": "person:beta_fil", "to": "person:beta_mil", "start": "1974-08-08", "end": None},
        ],
        [
            ("person:beta_self", "person:beta_dau_adult"),
            ("person:beta_husband", "person:beta_dau_adult"),
            ("person:beta_self", "person:beta_dau_minor"),
            ("person:beta_husband", "person:beta_dau_minor"),
            ("person:beta_self", "person:beta_son_minor"),
            ("person:beta_husband", "person:beta_son_minor"),
            ("person:beta_father", "person:beta_self"),
            ("person:beta_mother", "person:beta_self"),
            ("person:beta_fil", "person:beta_husband"),
            ("person:beta_mil", "person:beta_husband"),
            ("person:beta_brother", "person:beta_nephew"),
        ],
    )
    write_household(
        "gamma",
        [
            person("person:gamma_self", "顾北", "male", "1979-10-10"),
            person("person:gamma_dau_a", "顾棠", "female", "2011-03-03"),
            person("person:gamma_dau_b", "顾薇", "female", "2014-09-09"),
            person("person:gamma_just_adult", "顾成", "male", "2008-09-03"),
            person("person:gamma_still_minor", "顾未", "male", "2008-09-04"),
            person("person:gamma_roommate", "顾南", "female", "1985-05-05"),
            person("person:gamma_roommate_child", "顾苗", "female", "2019-01-20"),
            person("person:gamma_missing", "顾隐", "male", None),
        ],
        [
            "person:gamma_self",
            "person:gamma_dau_a",
            "person:gamma_dau_b",
            "person:gamma_just_adult",
            "person:gamma_still_minor",
            "person:gamma_roommate",
            "person:gamma_roommate_child",
        ],
        [],
        [
            ("person:gamma_self", "person:gamma_dau_a"),
            ("person:gamma_self", "person:gamma_dau_b"),
            ("person:gamma_self", "person:gamma_just_adult"),
            ("person:gamma_self", "person:gamma_still_minor"),
            ("person:gamma_roommate", "person:gamma_roommate_child"),
        ],
    )


def ref(kind: str, *steps: dict, entity_type: str | None = None, **extra: object) -> dict:
    payload = {
        "kind": kind,
        "entity_type": entity_type
        or ("address" if kind == "current_household" else "person"),
    }
    if steps:
        payload["path"] = list(steps)
    payload.update(extra)
    return payload


def step(relation: str, gender: str | None = None) -> dict:
    payload: dict = {"relation": relation}
    if gender is not None:
        payload["filters"] = [{"property": "gender", "value": gender}]
    return payload


def members() -> dict:
    return ref("current_household", step("member"))


def count_members(*filters: dict) -> dict:
    payload = {"operation": "count", "subject": members(), "property_source": "entity"}
    if filters:
        payload["filters"] = list(filters)
    return payload


def select_members(*filters: dict) -> dict:
    payload = {"operation": "select", "subject": members(), "property_source": "entity"}
    if filters:
        payload["filters"] = list(filters)
    return payload


def gender(value: str) -> dict:
    return {"property": "gender", "value": value}


def predicate(name: str) -> dict:
    return {"predicate": name}


PLANS = {
    "household_list": select_members(),
    "household_count": count_members(),
    "household_female_count": count_members(gender("female")),
    "household_male_count": count_members(gender("male")),
    "household_adult_count": count_members(predicate("adult")),
    "household_minor_count": count_members(predicate("minor")),
    "household_adult_female_count": count_members(predicate("adult"), gender("female")),
    "household_minor_female_count": count_members(predicate("minor"), gender("female")),
    "household_female_list": select_members(gender("female")),
    "household_minor_list": select_members(predicate("minor")),
    "household_oldest": {
        "operation": "argmin",
        "subject": members(),
        "property": "birth_date",
        "property_source": "entity",
    },
    "self_children_list": {
        "operation": "select",
        "subject": ref("self", step("child")),
        "property_source": "entity",
    },
    "self_daughters_list": {
        "operation": "select",
        "subject": ref("self", step("child", "female")),
        "property_source": "entity",
    },
    "self_sons_count": {
        "operation": "count",
        "subject": ref("self", step("child", "male")),
        "property_source": "entity",
    },
    "self_adult_daughters_count": {
        "operation": "count",
        "subject": ref("self", step("child", "female")),
        "property_source": "entity",
        "filters": [predicate("adult")],
    },
    "daughter_identity": {
        "operation": "resolve_reference",
        "subject": ref("self", step("child", "female")),
        "property_source": "entity",
    },
    "father_in_law": {
        "operation": "resolve_reference",
        "subject": ref("self", step("spouse"), step("parent", "male")),
        "property_source": "entity",
    },
    "wife_father": {
        "operation": "resolve_reference",
        "subject": ref("self", step("spouse", "female"), step("parent", "male")),
        "property_source": "entity",
    },
    "spouse_father_in_law": {
        "operation": "resolve_reference",
        "subject": ref(
            "self", step("spouse"), step("spouse"), step("parent", "male")
        ),
        "property_source": "entity",
    },
    "father": {
        "operation": "resolve_reference",
        "subject": ref("self", step("parent", "male")),
        "property_source": "entity",
    },
    "husband_father": {
        "operation": "resolve_reference",
        "subject": ref("self", step("spouse", "male"), step("parent", "male")),
        "property_source": "entity",
    },
    "wife_birth_select": {
        "operation": "select",
        "subject": ref("self", step("spouse", "female")),
        "property": "birth_date",
        "property_source": "entity",
    },
    "wife_birthday_days": {
        "operation": "annual_occurrence",
        "subject": ref("self", step("spouse", "female")),
        "property": "birth_date",
        "property_source": "entity",
        "mode": "days",
    },
    "wife_age_years": {
        "operation": "date_difference",
        "subject": ref("self", step("spouse", "female")),
        "property": "birth_date",
        "property_source": "entity",
        "mode": "years",
    },
    "husband_birth_select": {
        "operation": "select",
        "subject": ref("self", step("spouse", "male")),
        "property": "birth_date",
        "property_source": "entity",
    },
    "husband_birthday_days": {
        "operation": "annual_occurrence",
        "subject": ref("self", step("spouse", "male")),
        "property": "birth_date",
        "property_source": "entity",
        "mode": "days",
    },
    "named_missing_birth": {
        "operation": "select",
        "subject": {
            "kind": "named_entity",
            "entity_type": "person",
            "value": "顾隐",
        },
        "property": "birth_date",
        "property_source": "entity",
    },
    "year_1999_count": {
        "operation": "count",
        "subject": members(),
        "property_source": "entity",
        "filters": [
            {
                "property": "birth_date",
                "operator": "date_range",
                "value": ["1999-01-01", "2000-01-01"],
            }
        ],
    },
    "unresolved_they": {
        "operation": "resolve_reference",
        "subject": {"kind": "unresolved", "entity_type": "person"},
        "property_source": "entity",
    },
    "discourse_they": {
        "operation": "resolve_reference",
        "subject": {
            "kind": "discourse",
            "entity_type": "person",
            "turn_offset": 1,
            "cardinality": "collection",
        },
        "property_source": "entity",
    },
    "discourse_oldest": {
        "operation": "argmin",
        "subject": {
            "kind": "discourse",
            "entity_type": "person",
            "turn_offset": 1,
            "cardinality": "collection",
        },
        "property": "birth_date",
        "property_source": "entity",
    },
    "discourse_birthday_days": {
        "operation": "annual_occurrence",
        "subject": {
            "kind": "discourse",
            "entity_type": "person",
            "turn_offset": 1,
            "cardinality": "single",
        },
        "property": "birth_date",
        "property_source": "entity",
        "mode": "days",
    },
}

FORBIDDEN = {
    "self_member": {
        "operation": "select",
        "subject": ref("self", step("member")),
        "property_source": "entity",
    },
    "father_then_spouse": {
        "operation": "resolve_reference",
        "subject": ref("self", step("parent", "male"), step("spouse")),
        "property_source": "entity",
    },
    "adult_and_minor": count_members(predicate("adult"), predicate("minor")),
}


def case(
    case_id: str,
    cell: str,
    utterance: str,
    plan: str,
    *,
    speaker: str | None = None,
    notes: str,
    forbidden: list[str] | None = None,
    acceptable_plans: list[str] | None = None,
    acceptable_statuses: list[str] | None = None,
    expected_status: str | None = None,
) -> dict:
    row = {
        "id": case_id,
        "cell": cell,
        "utterance": utterance,
        "plan": plan,
        "notes": notes,
    }
    if speaker:
        row["speaker_id"] = speaker
    if forbidden:
        row["forbidden_plans"] = forbidden
    if acceptable_plans:
        row["acceptable_plans"] = acceptable_plans
    if acceptable_statuses:
        row["acceptable_statuses"] = acceptable_statuses
    if expected_status:
        row["expected_status"] = expected_status
    return row


DEV_CASES = {
    "alpha": [
        case("dev-alpha-S1-1", "S1", "请列出家里的人", "household_list", notes="Household members, not self→member.", forbidden=["self_member"]),
        case("dev-alpha-S2-1", "S2", "家里有几个人", "household_count", notes="Household count 10 with member ids."),
        case("dev-alpha-S3-1", "S3", "我家里都有谁", "household_list", notes="Still household members. Forbidden self→member.", forbidden=["self_member"]),
        case("dev-alpha-F1-1", "F1", "家里有几个女的", "household_female_count", notes="Female ≠ adult ≠ minor. Count collides with male=5; assert ids.", forbidden=["household_adult_count", "household_minor_count", "household_male_count"]),
        case("dev-alpha-F2-1", "F2", "家里有几个成年人", "household_adult_count", notes="Adult predicate, no gender."),
        case("dev-alpha-F3-1", "F3", "家里有几个未成年人", "household_minor_count", notes="Household minors, not my children."),
        case("dev-alpha-F4-1", "F4", "家里有几个成年女性", "household_adult_female_count", notes="adult AND gender=female."),
        case("dev-alpha-F5-1", "F5", "家里有几个未成年女性", "household_minor_female_count", notes="minor AND gender=female; not my daughters."),
        case("dev-alpha-F6-1", "F6", "请列出家里的女性", "household_female_list", notes="Female member list."),
        case("dev-alpha-F7-1", "F7", "家里有几个男的", "household_male_count", notes="Male count collides with female=5; assert ids.", forbidden=["household_adult_count", "household_female_count"]),
        case("dev-alpha-C1-1", "C1", "请列出我的孩子", "self_children_list", notes="Includes adult son; excludes niece."),
        case("dev-alpha-C2-1", "C2", "请列出家里的未成年人", "household_minor_list", notes="Minors include niece, exclude adult son."),
        case("dev-alpha-C3-1", "C3", "请列出我的女儿", "self_daughters_list", notes="One daughter; household minor females are two."),
        case("dev-alpha-C4-1", "C4", "我有几个儿子", "self_sons_count", notes="Two sons vs one household minor male."),
        case("dev-alpha-K1-1", "K1", "我的岳父是哪位", "father_in_law", notes="梅山. Not father, not spouse then FIL."),
        case("dev-alpha-K2-1", "K2", "我妻子的父亲是哪位", "wife_father", notes="wife then father only. Not father_in_law."),
        case("dev-alpha-K3-1", "K3", "我配偶的岳父是哪位", "spouse_father_in_law", notes="Speaker's father 安石, not FIL."),
        case("dev-alpha-K4-1", "K4", "我的父亲是哪位", "father", notes="安石. Distinct from FIL."),
        case("dev-alpha-O1-1", "O1", "我妻子的出生日期是哪天", "wife_birth_select", notes="select(birth_date), not annual_occurrence."),
        case("dev-alpha-O2-1", "O2", "我妻子下个生日还有多少天", "wife_birthday_days", notes="annual_occurrence days, not stored date."),
        case("dev-alpha-O3-1", "O3", "我妻子现在多少岁", "wife_age_years", notes="date_difference years."),
        case("dev-alpha-O4-1", "O4", "家里谁最年长", "household_oldest", notes="Oldest household member."),
        case("dev-alpha-E1-1", "E1", "我有几个成年女儿", "self_adult_daughters_count", notes="Empty set, count 0 found."),
        case("dev-alpha-E7-1", "E7", "他们是谁", "unresolved_they", notes="Clarification, not a guessed person.", acceptable_plans=["discourse_they"], acceptable_statuses=["discourse_context_missing"]),
    ],
    "beta": [
        case("dev-beta-S1-1", "S1", "请列出家里的人", "household_list", notes="Beta member ids differ from alpha."),
        case("dev-beta-F1-1", "F1", "家里有几个女的", "household_female_count", notes="Count 5, ids ≠ alpha females."),
        case("dev-beta-F4-1", "F4", "家里有几个成年女性", "household_adult_female_count", notes="Adult females=4 vs alpha 3."),
        case("dev-beta-F5-1", "F5", "家里有几个未成年女性", "household_minor_female_count", notes="Minor females=1 vs alpha 2."),
        case("dev-beta-C1-1", "C1", "请列出我的孩子", "self_children_list", notes="林晴, 林露, 林川."),
        case("dev-beta-C3-1", "C3", "请列出我的女儿", "self_daughters_list", notes="Two daughters vs alpha one."),
        case("dev-beta-K1-1", "K1", "我的岳父是哪位", "father_in_law", notes="海峰, not 梅山."),
        case("dev-beta-K3-1", "K3", "我配偶的岳父是哪位", "spouse_father_in_law", notes="Speaker's father 夏松."),
        case("dev-beta-K5-1", "K5", "我丈夫的父亲是哪位", "husband_father", notes="husband then father only; 海峰."),
    ],
    "gamma": [
        case("dev-gamma-S2-1", "S2", "家里有几个人", "household_count", notes="Seven residents."),
        case("dev-gamma-F3-1", "F3", "家里有几个未成年人", "household_minor_count", notes="Four minors including 顾未."),
        case("dev-gamma-K6-1", "K6", "我的岳父是哪位", "father_in_law", notes="No spouse, empty in-laws."),
        case("dev-gamma-E2-1", "E2", "顾隐哪天出生", "named_missing_birth", notes="Missing dob is property_unavailable."),
        case("dev-gamma-E4-1", "E4", "家里有几个成年人", "household_adult_count", notes="Includes 顾成 (18 today), excludes 顾未."),
        case("dev-gamma-E5-1", "E5", "家里未成年的有几个", "household_minor_count", notes="Includes 顾未 (17), excludes 顾成."),
        case("dev-gamma-E6-1", "E6", "家里1999年出生的有几个人", "year_1999_count", notes="Zero-match year, found 0."),
        case("dev-gamma-E8-1", "E8", "我女儿是谁", "daughter_identity", speaker="person:gamma_self", notes="Two daughters, ambiguous."),
        case("dev-gamma-E8-2", "E8", "我女儿是谁", "daughter_identity", speaker="person:gamma_roommate", notes="Roommate's only daughter 顾苗."),
    ],
}

FRZ_CASES = {
    "alpha": [
        case("frz-alpha-S1-1", "S1", "Who are the people in this household?", "household_list", notes="Held-out English household list.", forbidden=["self_member"]),
        case("frz-alpha-S2-1", "S2", "本户现在一共几位成员？", "household_count", notes="Held-out 本户 count."),
        case("frz-alpha-S3-1", "S3", "咱家这户人都有哪些？", "household_list", notes="Held-out 咱家 still household members.", forbidden=["self_member"]),
        case("frz-alpha-F2-1", "F2", "本户已满十八岁的成员有几位？", "household_adult_count", notes="Adult predicate paraphrase."),
        case("frz-alpha-F5-1", "F5", "本户未成年的女孩子有几个？", "household_minor_female_count", notes="minor AND female."),
        case("frz-alpha-F6-1", "F6", "List the women who live here.", "household_female_list", notes="English female list."),
        case("frz-alpha-C2-1", "C2", "Who in this household is still a minor?", "household_minor_list", notes="Household minors, not my children."),
        case("frz-alpha-C4-1", "C4", "我名下的儿子一共几个？", "self_sons_count", notes="My sons count."),
        case("frz-alpha-K2-1", "K2", "我老婆她爸是哪一位？", "wife_father", notes="wife then father only."),
        case("frz-alpha-K3-1", "K3", "我爱人的岳父是谁？", "spouse_father_in_law", notes="spouse then FIL → 安石."),
        case("frz-alpha-K4-1", "K4", "我自己的爸爸是哪一位？", "father", notes="Speaker's father."),
        case("frz-alpha-O1-1", "O1", "What is my wife's date of birth on record?", "wife_birth_select", notes="Stored date, English."),
        case("frz-alpha-O1-2", "O1", "我妻子出生那天是哪一号？", "wife_birth_select", notes="Stored date, not countdown."),
        case("frz-alpha-O2-1", "O2", "我太太距离下次过生日还有几天？", "wife_birthday_days", notes="Countdown, held-out 太太."),
        case("frz-alpha-O3-1", "O3", "我太太今年满几周岁了？", "wife_age_years", notes="Age years."),
        case("frz-alpha-O4-1", "O4", "本户出生最早的是哪一位？", "household_oldest", notes="Oldest member."),
        case("frz-alpha-E1-1", "E1", "How many of my daughters are adults?", "self_adult_daughters_count", notes="Empty adult daughters."),
        case("frz-alpha-E7-1", "E7", "Who are they?", "unresolved_they", notes="Clarification.", acceptable_plans=["discourse_they"], acceptable_statuses=["discourse_context_missing"]),
        case("frz-alpha-F4-2", "F4", "成年的女性成员有几位？", "household_adult_female_count", notes="Adult female vs female."),
    ],
    "beta": [
        case("frz-beta-F1-1", "F1", "How many females live in this household?", "household_female_count", notes="English female count, beta ids."),
        case("frz-beta-F4-1", "F4", "How many adult women are in this household?", "household_adult_female_count", notes="Adult women=4."),
        case("frz-beta-F7-1", "F7", "本户男性成员有几位？", "household_male_count", notes="Beta males, ids required."),
        case("frz-beta-C1-1", "C1", "把我自己的子女都列出来", "self_children_list", notes="My children on beta."),
        case("frz-beta-C2-1", "C2", "本户的小孩子都有谁", "household_minor_list", notes="Household minors on beta."),
        case("frz-beta-C3-1", "C3", "我自己的千金都有谁", "self_daughters_list", notes="My daughters."),
        case("frz-beta-K1-1", "K1", "请指出我的公公是哪一位？", "father_in_law", notes="公公 = FIL on female speaker."),
        case("frz-beta-K2-1", "K2", "Which person is my wife's father?", "wife_father", notes="Female speaker, no wife → relationship_not_found. Not repaired to FIL."),
        case("frz-beta-K5-1", "K5", "我先生的父亲叫什么身份？", "husband_father", notes="husband then father resolve_reference."),
    ],
    "gamma": [
        case("frz-gamma-F3-1", "F3", "咱家还没成年的有几位？", "household_minor_count", notes="Gamma minors=4."),
        case("frz-gamma-K6-1", "K6", "Name the person who is my father in law.", "father_in_law", notes="Empty in-laws, English held-out."),
        case("frz-gamma-E2-1", "E2", "顾隐的出生日期能查到吗？", "named_missing_birth", notes="Missing dob."),
        case("frz-gamma-E4-1", "E4", "本户已成年的一共几人？", "household_adult_count", notes="Includes 顾成."),
        case("frz-gamma-E5-1", "E5", "How many household members are still minors?", "household_minor_count", notes="Includes 顾未."),
        case("frz-gamma-E6-1", "E6", "本户在一九九九年出生的有几位？", "year_1999_count", notes="Zero-match year."),
        case("frz-gamma-E8-1", "E8", "我闺女是哪位？", "daughter_identity", speaker="person:gamma_self", notes="Ambiguous daughters."),
        case("frz-gamma-E8-2", "E8", "我闺女是哪位？", "daughter_identity", speaker="person:gamma_roommate", notes="顾苗 only."),
    ],
}

DEV_SEQUENCES = [
    {
        "id": "dev-seq-G1",
        "household": "alpha",
        "speaker_id": "person:alpha_self",
        "cell": "G1",
        "notes": "After adult count, gender count must drop adult.",
        "turns": [
            {"utterance": "家里有几个成年人", "plan": "household_adult_count"},
            {
                "utterance": "家里有几个男的",
                "plan": "household_male_count",
                "forbidden_plans": ["household_adult_count", "adult_and_minor"],
            },
        ],
    },
    {
        "id": "dev-seq-G2",
        "household": "alpha",
        "speaker_id": "person:alpha_self",
        "cell": "G2",
        "notes": "FIL after father must not emit father then spouse.",
        "turns": [
            {"utterance": "我的父亲是哪位", "plan": "father"},
            {
                "utterance": "我的岳父是哪位",
                "plan": "father_in_law",
                "forbidden_plans": ["father_then_spouse", "spouse_father_in_law"],
            },
        ],
    },
    {
        "id": "dev-seq-G3",
        "household": "alpha",
        "speaker_id": "person:alpha_self",
        "cell": "G3",
        "notes": "Wife countdown after FIL; no leftover in-law hops.",
        "turns": [
            {"utterance": "我的岳父是哪位", "plan": "father_in_law"},
            {
                "utterance": "我妻子下个生日还有多少天",
                "plan": "wife_birthday_days",
                "forbidden_plans": ["wife_birth_select"],
            },
        ],
    },
    {
        "id": "dev-seq-G4",
        "household": "alpha",
        "speaker_id": "person:alpha_self",
        "cell": "G4",
        "notes": "Same last utterance as standalone O1 after unrelated list.",
        "turns": [
            {"utterance": "请列出家里的人", "plan": "household_list"},
            {"utterance": "我妻子的出生日期是哪天", "plan": "wife_birth_select"},
        ],
    },
    {
        "id": "dev-seq-G5",
        "household": "alpha",
        "speaker_id": "person:alpha_self",
        "cell": "G5",
        "notes": "After my children, 我家里都有谁 is still household members.",
        "turns": [
            {"utterance": "请列出我的孩子", "plan": "self_children_list"},
            {
                "utterance": "我家里都有谁",
                "plan": "household_list",
                "forbidden_plans": ["self_member", "self_children_list"],
            },
        ],
    },
    {
        "id": "dev-seq-G6",
        "household": "alpha",
        "speaker_id": "person:alpha_self",
        "cell": "G6",
        "notes": "Adult female after female count must add adult and keep gender.",
        "turns": [
            {"utterance": "家里有几个女的", "plan": "household_female_count"},
            {
                "utterance": "家里有几个成年女性",
                "plan": "household_adult_female_count",
                "forbidden_plans": ["household_female_count", "household_adult_count"],
            },
        ],
    },
    {
        "id": "dev-seq-G7",
        "household": "alpha",
        "speaker_id": "person:alpha_self",
        "cell": "G7",
        "notes": "他们 who is oldest: discourse collection only.",
        "turns": [
            {"utterance": "请列出家里的人", "plan": "household_list"},
            {
                "utterance": "他们谁最年长",
                "plan": "discourse_oldest",
                "forbidden_plans": ["household_oldest"],
            },
        ],
    },
    {
        "id": "dev-seq-G8",
        "household": "alpha",
        "speaker_id": "person:alpha_self",
        "cell": "G8",
        "notes": "她 countdown: discourse only, not restart-as-wife.",
        "turns": [
            {"utterance": "我妻子的出生日期是哪天", "plan": "wife_birth_select"},
            {
                "utterance": "她下个生日还有几天",
                "plan": "discourse_birthday_days",
                "forbidden_plans": ["wife_birthday_days", "wife_birth_select"],
            },
        ],
    },
]

FRZ_SEQUENCES = [
    {
        "id": "frz-seq-G1",
        "household": "beta",
        "speaker_id": "person:beta_self",
        "cell": "G1",
        "notes": "Gender after adult on beta so leaked adult count cannot reuse alpha 7.",
        "turns": [
            {"utterance": "本户已满十八岁的成员有几位？", "plan": "household_adult_count"},
            {
                "utterance": "本户男性成员有几位？",
                "plan": "household_male_count",
                "forbidden_plans": ["household_adult_count"],
            },
        ],
    },
    {
        "id": "frz-seq-G2",
        "household": "beta",
        "speaker_id": "person:beta_self",
        "cell": "G2",
        "notes": "公公 after own father.",
        "turns": [
            {"utterance": "我自己的爸爸是哪一位？", "plan": "father"},
            {
                "utterance": "请指出我的公公是哪一位？",
                "plan": "father_in_law",
                "forbidden_plans": ["father_then_spouse"],
            },
        ],
    },
    {
        "id": "frz-seq-G3",
        "household": "beta",
        "speaker_id": "person:beta_self",
        "cell": "G3",
        "notes": "Husband countdown after in-law; leftover hops fail.",
        "turns": [
            {"utterance": "请指出我的公公是哪一位？", "plan": "father_in_law"},
            {"utterance": "我先生下次生日还有几天？", "plan": "husband_birthday_days"},
        ],
    },
    {
        "id": "frz-seq-G4",
        "household": "beta",
        "speaker_id": "person:beta_self",
        "cell": "G4",
        "notes": "Standalone-equivalent last turn after household list.",
        "turns": [
            {"utterance": "Who are the people in this household?", "plan": "household_list"},
            {"utterance": "What is my husband's date of birth on record?", "plan": "husband_birth_select"},
        ],
    },
    {
        "id": "frz-seq-G5",
        "household": "beta",
        "speaker_id": "person:beta_self",
        "cell": "G5",
        "notes": "After my children, household list wording must not keep child.",
        "turns": [
            {"utterance": "把我自己的子女都列出来", "plan": "self_children_list"},
            {
                "utterance": "咱家这户人都有哪些？",
                "plan": "household_list",
                "forbidden_plans": ["self_member"],
            },
        ],
    },
    {
        "id": "frz-seq-G6",
        "household": "beta",
        "speaker_id": "person:beta_self",
        "cell": "G6",
        "notes": "Adult women after females.",
        "turns": [
            {"utterance": "How many females live in this household?", "plan": "household_female_count"},
            {"utterance": "How many adult women are in this household?", "plan": "household_adult_female_count"},
        ],
    },
    {
        "id": "frz-seq-G7",
        "household": "beta",
        "speaker_id": "person:beta_self",
        "cell": "G7",
        "notes": "Which of them was born first: discourse only.",
        "turns": [
            {"utterance": "Who are the people in this household?", "plan": "household_list"},
            {
                "utterance": "Which of them was born first?",
                "plan": "discourse_oldest",
                "forbidden_plans": ["household_oldest"],
            },
        ],
    },
    {
        "id": "frz-seq-G8",
        "household": "gamma",
        "speaker_id": "person:gamma_roommate",
        "cell": "G8",
        "notes": "她 after unique 闺女; discourse, not wife.",
        "turns": [
            {"utterance": "我闺女是哪位？", "plan": "daughter_identity"},
            {
                "utterance": "她下个生日还有几天？",
                "plan": "discourse_birthday_days",
                "forbidden_plans": ["wife_birthday_days"],
            },
        ],
    },
]

SPEAKERS = {
    "alpha": "person:alpha_self",
    "beta": "person:beta_self",
    "gamma": "person:gamma_self",
}

COUNT_ID_PLANS = {
    "household_count",
    "household_female_count",
    "household_male_count",
    "household_adult_count",
    "household_minor_count",
    "household_adult_female_count",
    "household_minor_female_count",
    "self_sons_count",
    "self_adult_daughters_count",
    "year_1999_count",
}


def used_plan_ids(cases: list[dict], sequences: list[dict] | None = None) -> tuple[set[str], set[str]]:
    used: set[str] = {row["plan"] for row in cases}
    forbidden: set[str] = set()
    for row in cases:
        used.update(row.get("acceptable_plans") or ())
        forbidden.update(row.get("forbidden_plans") or ())
    for sequence in sequences or ():
        for turn in sequence["turns"]:
            used.add(turn["plan"])
            used.update(turn.get("acceptable_plans") or ())
            forbidden.update(turn.get("forbidden_plans") or ())
    return used, forbidden


def plans_for(ids: set[str]) -> dict:
    payload = {key: deepcopy(PLANS[key]) for key in sorted(ids) if key in PLANS}
    return payload


def forbidden_for(ids: set[str]) -> dict:
    payload = {}
    for key in sorted(ids):
        if key in FORBIDDEN:
            payload[key] = deepcopy(FORBIDDEN[key])
        elif key in PLANS:
            payload[key] = deepcopy(PLANS[key])
    return payload


async def fill_case(row: dict, household: str, default_speaker: str) -> dict:
    engine, _ = household_engine(household)
    speaker = row.get("speaker_id") or default_speaker
    request = SemanticFactRequest.model_validate(PLANS[row["plan"]])
    context = request_context(speaker_id=speaker, household=household)
    result, *_ = await engine.execute(request, context)
    filled = dict(row)
    filled["expected_status"] = result.status
    if result.unit:
        filled["expected_unit"] = result.unit
    ids = list(primary_entity_ids(result))
    if ids:
        filled["expected_entity_ids"] = ids
    elif row["plan"] in COUNT_ID_PLANS and result.status == "found":
        filled["expected_entity_ids"] = []
    if result.status == "found" and not isinstance(result.value, (list, dict)):
        filled["expected_value"] = result.value
    if "acceptable_statuses" in filled and result.status == "ambiguous":
        pass
    return filled


async def fill_sequence(sequence: dict) -> dict:
    household = sequence["household"]
    engine, _ = household_engine(household)
    speaker = sequence["speaker_id"]
    conversation_id = sequence["id"]
    turns: list[tuple[str, ...]] = []
    filled_turns = []
    for index, turn in enumerate(sequence["turns"]):
        request = SemanticFactRequest.model_validate(PLANS[turn["plan"]])
        discourse = None
        if request.subject.kind == "discourse":
            discourse = DiscourseContext(
                conversation_id,
                speaker,
                HOUSEHOLD_IDS[household],
                "assistant:composition",
                tuple(turns[-8:]),
            )
        context = request_context(
            speaker_id=speaker,
            household=household,
            conversation_id=conversation_id,
            discourse=discourse,
        )
        result, *_ = await engine.execute(request, context)
        focus = result.focus_entity_ids if result.status == "found" else ()
        turns.append(tuple(focus))
        filled = dict(turn)
        if index == len(sequence["turns"]) - 1:
            filled["expected_status"] = result.status
            if result.unit:
                filled["expected_unit"] = result.unit
            ids = list(primary_entity_ids(result))
            if ids:
                filled["expected_entity_ids"] = ids
            if result.status == "found" and not isinstance(result.value, (list, dict)):
                filled["expected_value"] = result.value
        filled_turns.append(filled)
    out = dict(sequence)
    out["frozen_time"] = FROZEN_EVAL_TIME
    out["turns"] = filled_turns
    return out


async def emit() -> None:
    write_households()
    for split, household_cases, sequences in (
        ("development", DEV_CASES, DEV_SEQUENCES),
        ("frozen", FRZ_CASES, FRZ_SEQUENCES),
    ):
        for household, rows in household_cases.items():
            filled = [
                await fill_case(row, household, SPEAKERS[household]) for row in rows
            ]
            used, forbidden_ids = used_plan_ids(filled)
            payload = {
                "version": 1,
                "household": household,
                "plans": plans_for(used),
                "forbidden_plans": forbidden_for(forbidden_ids),
                "probe": {
                    "frozen_time": FROZEN_EVAL_TIME,
                    "default_speaker_id": SPEAKERS[household],
                    "household_id": HOUSEHOLD_IDS[household],
                    "household": household,
                    "cases": filled,
                },
            }
            dump_yaml(COMPOSITION_ROOT / split / f"{household}.yaml", payload)
        filled_sequences = [await fill_sequence(item) for item in sequences]
        used, forbidden_ids = used_plan_ids([], filled_sequences)
        dump_yaml(
            COMPOSITION_ROOT / split / "sequences.yaml",
            {
                "version": 1,
                "plans": plans_for(used),
                "forbidden_plans": forbidden_for(forbidden_ids),
                "sequences": filled_sequences,
            },
        )

    standalone = []
    for split, household_cases in (("development", DEV_CASES), ("frozen", FRZ_CASES)):
        for household, rows in household_cases.items():
            for row in rows:
                standalone.append(
                    {
                        "id": row["id"],
                        "split": split,
                        "household": household,
                        "cell": row["cell"],
                        "utterance": row["utterance"],
                        "speaker_id": row.get("speaker_id") or SPEAKERS[household],
                        "plan": row["plan"],
                        "expression_family": "canonical" if split == "development" else "held-out",
                    }
                )
    dump_yaml(
        COMPOSITION_ROOT / "frozen" / "MANIFEST.yaml",
        {
            "version": 1,
            "frozen_time": FROZEN_EVAL_TIME,
            "size": {
                "development_standalone": sum(len(v) for v in DEV_CASES.values()),
                "frozen_standalone": sum(len(v) for v in FRZ_CASES.values()),
                "development_sequences": len(DEV_SEQUENCES),
                "frozen_sequences": len(FRZ_SEQUENCES),
            },
            "cases": [item for item in standalone if item["split"] == "frozen"],
            "sequences": [
                {
                    "id": item["id"],
                    "household": item["household"],
                    "cell": item["cell"],
                    "utterances": [turn["utterance"] for turn in item["turns"]],
                }
                for item in FRZ_SEQUENCES
            ],
        },
    )
    dump_json(COMPOSITION_ROOT / "fingerprints.json", composition_fingerprint_payload())
    print(
        "emitted",
        sum(len(v) for v in DEV_CASES.values()),
        "dev",
        sum(len(v) for v in FRZ_CASES.values()),
        "frozen",
        len(DEV_SEQUENCES),
        len(FRZ_SEQUENCES),
        "sequences",
    )


if __name__ == "__main__":
    asyncio.run(emit())
