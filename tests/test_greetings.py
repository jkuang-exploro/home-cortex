from types import MappingProxyType
from typing import Any

import pytest

from home_cortex.agents import AgentDefinition, ModelConfiguration, get_agent
from home_cortex.greetings import GreetingResolver, GreetingService

HOUSEHOLD = {
    "id": "location:fort_cerritos",
    "name": ["Fort Cerritos", "喜瑞匡家"],
}


def _person(
    record_id: str,
    *,
    address_en: str | None,
    address_zh: str | None,
) -> dict[str, Any]:
    address_as = {
        language: value
        for language, value in (("en", address_en), ("zh", address_zh))
        if value is not None
    }
    return {
        "id": record_id,
        "name": [record_id.rpartition(":")[2].replace("_", " ").title(), "测试人"],
        "address_as": address_as,
    }


def _relationship(
    person_id: str,
    role: str | None,
    household_id: str = "location:fort_cerritos",
) -> dict[str, Any]:
    relationship: dict[str, Any] = {
        "id": f"resides_in:{person_id.rpartition(':')[2]}",
        "in": person_id,
        "out": household_id,
        "relation": "resides_in",
        "related_entity": HOUSEHOLD,
    }
    if role is not None:
        relationship["household_role"] = role
    return relationship


def _resolve(
    person: dict[str, Any] | None,
    role: str | None,
    language: str,
):
    relationships = (
        [_relationship(str(person["id"]), role)] if person is not None else []
    )
    return GreetingResolver(get_agent("steward").settings).resolve(
        person=person,
        household=HOUSEHOLD,
        relationships=relationships,
        household_id="location:fort_cerritos",
        language=language,
    )


def test_owner_greeting_uses_address_as_not_name_or_internal_id() -> None:
    person = _person(
        "person:jian_kuang",
        address_en="Mr. Kuang",
        address_zh="先生",
    )

    greeting = _resolve(person, "owner", "zh")

    assert greeting.text == "先生，您回来了。老管家在此，今日有什么需要吩咐？"
    assert "匡健" not in greeting.text
    assert "person:" not in greeting.text
    assert greeting.reception_category == "owner"


def test_different_owners_can_use_configured_person_overrides() -> None:
    first = _person(
        "person:jian_kuang",
        address_en="Mr. Kuang",
        address_zh="先生",
    )
    second = _person(
        "person:pu_ba",
        address_en="Mrs. Ba",
        address_zh="太太",
    )

    first_greeting = _resolve(first, "owner", "zh")
    second_greeting = _resolve(second, "owner", "zh")

    assert first_greeting.text != second_greeting.text
    assert second_greeting.text == "太太，您回来了。今日有什么需要老管家处理？"


@pytest.mark.parametrize(
    ("role", "address", "expected"),
    [
        (
            "minor_dependent",
            "公子",
            "公子，您回来了。今天过得怎么样？有什么需要可以告诉我。",
        ),
        (
            "adult_dependent",
            "老先生",
            "老先生，您回来了。有什么需要我为您准备的吗？",
        ),
    ],
)
def test_household_dependents_receive_role_appropriate_greetings(
    role: str,
    address: str,
    expected: str,
) -> None:
    person = _person("person:family_member", address_en=None, address_zh=address)

    assert _resolve(person, role, "zh").text == expected


def test_guest_receives_guest_language_without_owner_authority() -> None:
    guest = _person(
        "person:zhang_guest",
        address_en="Mr. Zhang",
        address_zh="张先生",
    )

    greeting = _resolve(guest, "guest", "zh")

    assert greeting.text == "张先生，您好，欢迎来到喜瑞匡家。有什么需要请告诉我。"
    assert "吩咐" not in greeting.text
    assert "您回来了" not in greeting.text


def test_unknown_person_and_unknown_relationship_are_never_owners() -> None:
    person = _person(
        "person:unrelated",
        address_en="Dr. Visitor",
        address_zh="访客",
    )

    unidentified = _resolve(None, None, "zh")
    unrelated = _resolve(person, None, "zh")

    assert unidentified.text == unrelated.text == "您好，有什么可以帮您？"
    assert unidentified.reception_category == "unknown"
    assert unrelated.reception_category == "unknown"
    assert "吩咐" not in unrelated.text


def test_known_role_without_a_human_reference_uses_neutral_text() -> None:
    person = {"id": "person:unnamed"}

    greeting = _resolve(person, "owner", "zh")

    assert greeting.reception_category == "owner"
    assert greeting.text == "您好，有什么可以帮您？"
    assert "person:" not in greeting.text


def test_conflicting_relationship_roles_fail_closed_to_unknown() -> None:
    person = _person(
        "person:conflicting",
        address_en="Visitor",
        address_zh="访客",
    )
    relationships = [
        _relationship(str(person["id"]), "owner"),
        _relationship(str(person["id"]), "guest"),
    ]

    greeting = GreetingResolver(get_agent("steward").settings).resolve(
        person=person,
        household=HOUSEHOLD,
        relationships=relationships,
        household_id="location:fort_cerritos",
        language="zh",
    )

    assert greeting.reception_category == "unknown"
    assert greeting.text == "您好，有什么可以帮您？"


def test_chinese_and_english_policies_resolve_independently() -> None:
    person = _person(
        "person:jian_kuang",
        address_en="Mr. Kuang",
        address_zh="先生",
    )

    english = _resolve(person, "owner", "en")
    chinese = _resolve(person, "owner", "zh")

    assert english.text.startswith("Mr. Kuang, welcome home.")
    assert "the butler" in english.text.casefold()
    assert "老管家" not in english.text
    assert chinese.text.startswith("先生，您回来了。")
    assert "老管家" in chinese.text
    assert "the butler" not in chinese.text


def test_missing_localized_address_uses_existing_address_fallback() -> None:
    person = _person(
        "person:dylan_kuang",
        address_en=None,
        address_zh="公子",
    )

    greeting = _resolve(person, "minor_dependent", "en-US")

    assert greeting.language == "en"
    assert greeting.text.startswith("公子, welcome home.")


def test_greeting_does_not_fabricate_household_status() -> None:
    person = _person(
        "person:jian_kuang",
        address_en="Mr. Kuang",
        address_zh="先生",
    )

    greeting = _resolve(person, "owner", "zh").text

    assert "一切如常" not in greeting
    assert "厨房" not in greeting
    assert "已经回来" not in greeting


class FakeRetrieval:
    def __init__(self, person_id: str, role: str) -> None:
        self.person_id = person_id
        self.role = role
        self.calls: list[tuple[str, str]] = []

    async def get_entity(self, record_id: str) -> dict[str, Any] | None:
        self.calls.append(("entity", record_id))
        if record_id == HOUSEHOLD["id"]:
            return HOUSEHOLD
        return None

    async def get_relationships(
        self,
        entity_id: str,
        relation: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(("relationships", entity_id))
        return [_relationship(self.person_id, self.role)]


@pytest.mark.asyncio
async def test_greeting_service_uses_graph_state_without_an_llm_call() -> None:
    person = _person(
        "person:jian_kuang",
        address_en="Mr. Kuang",
        address_zh="先生",
    )
    retrieval = FakeRetrieval(str(person["id"]), "owner")

    greeting = await GreetingService(retrieval).resolve(
        get_agent("steward"),
        person,
        "zh",
    )

    assert greeting.reception_category == "owner"
    assert retrieval.calls == [
        ("entity", "location:fort_cerritos"),
        ("relationships", "person:jian_kuang"),
    ]


def test_switching_inference_models_does_not_change_greeting_policy() -> None:
    steward = get_agent("steward")
    person = _person(
        "person:jian_kuang",
        address_en="Mr. Kuang",
        address_zh="先生",
    )
    first = GreetingResolver(steward.settings).resolve(
        person=person,
        household=HOUSEHOLD,
        relationships=[_relationship(str(person["id"]), "owner")],
        household_id="location:fort_cerritos",
        language="zh",
    )
    model_switched_definition = AgentDefinition(
        id=steward.id,
        display_name=steward.display_name,
        description=steward.description,
        prompt=steward.prompt,
        model=ModelConfiguration(provider="ollama", name="small-tool-model"),
        allowed_tools=steward.allowed_tools,
        tool_definitions=steward.tool_definitions,
        settings=steward.settings,
    )
    second = GreetingResolver(model_switched_definition.settings).resolve(
        person=person,
        household=HOUSEHOLD,
        relationships=[_relationship(str(person["id"]), "owner")],
        household_id="location:fort_cerritos",
        language="zh",
    )

    assert first == second


def test_another_agent_can_supply_an_independent_reception_policy() -> None:
    person = _person(
        "person:jian_kuang",
        address_en="Mr. Kuang",
        address_zh="先生",
    )
    accountant = AgentDefinition(
        id="accountant",
        display_name="账房",
        description="Finance agent",
        prompt="Finance prompt",
        model=ModelConfiguration(provider="ollama", name="another-model"),
        allowed_tools=(),
        tool_definitions=(),
        settings=MappingProxyType(
            {
                "home_entity_id": "location:fort_cerritos",
                "reception": {
                    "default_language": "en",
                    "greetings": {
                        "en": {
                            "owner": "{address_as}, the accounts are ready.",
                            "unknown": "Hello from accounting.",
                        }
                    },
                },
            }
        ),
    )

    greeting = GreetingResolver(accountant.settings).resolve(
        person=person,
        household=HOUSEHOLD,
        relationships=[_relationship(str(person["id"]), "owner")],
        household_id="location:fort_cerritos",
        language="en",
    )

    assert greeting.text == "Mr. Kuang, the accounts are ready."
    assert "butler" not in greeting.text
