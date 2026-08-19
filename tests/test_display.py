from copy import deepcopy

from home_cortex.display import (
    DisplayNameResolver,
    DisplayTextStream,
    internal_ids_requested,
    resolve_display_name,
    resolve_person_reference,
)

FORT_CERRITOS = {
    "id": "location:fort_cerritos",
    "name": ["Fort Cerritos", "喜瑞匡家"],
}


def test_resolves_legacy_name_list_for_chinese_and_english() -> None:
    assert resolve_display_name(FORT_CERRITOS, "zh") == "喜瑞匡家"
    assert resolve_display_name(FORT_CERRITOS, "en") == "Fort Cerritos"


def test_resolves_localized_name_mapping() -> None:
    entity = {
        "id": "vehicle:model_y",
        "name": {"zh": "特斯拉 Model Y", "en": "Tesla Model Y"},
    }

    assert resolve_display_name(entity, "zh-CN") == "特斯拉 Model Y"
    assert resolve_display_name(entity, "en-US") == "Tesla Model Y"


def test_missing_requested_language_uses_default_display_name() -> None:
    entity = {
        "id": "space:fort_cerritos:garage",
        "default_display_name": "Garage",
        "name": {"en": "Garage"},
    }

    assert resolve_display_name(entity, "zh") == "Garage"


def test_missing_display_name_safely_falls_back_to_internal_id() -> None:
    assert (
        resolve_display_name({"id": "space:fort_cerritos:garage"}, "zh")
        == "space:fort_cerritos:garage"
    )


def test_render_replaces_known_ids_but_preserves_unknown_ids() -> None:
    resolver = DisplayNameResolver([FORT_CERRITOS])

    rendered = resolver.render(
        "Home location:fort_cerritos; unknown vehicle:missing.",
        "en",
    )

    assert rendered == "Home Fort Cerritos; unknown vehicle:missing."


def test_normal_render_hides_all_known_internal_id_shapes() -> None:
    entities = [
        FORT_CERRITOS,
        {"id": "person:jian_kuang", "name": {"en": "Jian Kuang"}},
        {"id": "vehicle:model_y", "name": {"en": "Tesla Model Y"}},
        {
            "id": "space:fort_cerritos:garage",
            "name": {"en": "Garage"},
        },
    ]
    resolver = DisplayNameResolver(entities)

    rendered = resolver.render(
        "person:jian_kuang uses vehicle:model_y at "
        "location:fort_cerritos in space:fort_cerritos:garage.",
        "en",
    )

    assert rendered == (
        "Jian Kuang uses Tesla Model Y at Fort Cerritos in Garage."
    )


def test_explicit_internal_id_request_disables_presentation_replacement() -> None:
    resolver = DisplayNameResolver([FORT_CERRITOS])

    assert internal_ids_requested(
        [{"role": "user", "content": "What is its internal ID?"}]
    )
    assert resolver.render(
        "location:fort_cerritos",
        "en",
        expose_internal_ids=True,
    ) == "location:fort_cerritos"


def test_stream_renderer_handles_an_id_split_across_chunks() -> None:
    stream = DisplayTextStream(DisplayNameResolver([FORT_CERRITOS]), "zh")

    chunks = [
        stream.feed("这里是 location:"),
        stream.feed("fort_cerritos。"),
        stream.finish(),
    ]

    assert "".join(chunks) == "这里是 喜瑞匡家。"


def test_resolver_is_reusable_and_does_not_mutate_graph_results() -> None:
    tool_result = {
        "result": [
            {
                "id": "lives_in:jian_home",
                "out": "location:fort_cerritos",
                "related_entity": FORT_CERRITOS,
            }
        ]
    }
    original = deepcopy(tool_result)
    resolver = DisplayNameResolver([tool_result])

    assert resolver.resolve("location:fort_cerritos", "zh") == "喜瑞匡家"
    assert tool_result == original


def test_person_reference_modes_keep_identity_name_and_address_distinct() -> None:
    person = {
        "id": "person:jian_kuang",
        "name": {"zh": "匡健", "en": "Jian Kuang"},
        "address_as": {"zh": "先生", "en": "Mr. Kuang"},
    }

    assert resolve_person_reference(person, "zh", mode="address") == "先生"
    assert resolve_person_reference(person, "zh", mode="name") == "匡健"
    assert resolve_person_reference(person, "zh", mode="id") == (
        "person:jian_kuang"
    )
    assert resolve_person_reference(person, "en", mode="address") == "Mr. Kuang"


def test_configured_household_forms_of_address_are_not_inferred() -> None:
    people = [
        ("person:pu_ba", "太太"),
        ("person:dylan_kuang", "公子"),
        ("person:evelyn_kuang", "小姐"),
        ("person:zhigang_ba", "老先生"),
    ]

    for record_id, address_as in people:
        person = {
            "id": record_id,
            "name": {"zh": "测试姓名"},
            "address_as": {"zh": address_as},
        }
        assert resolve_person_reference(person, "zh") == address_as


def test_missing_localized_address_uses_another_configured_address() -> None:
    person = {
        "id": "person:dylan_kuang",
        "name": {"en": "Dylan Kuang"},
        "address_as": {"zh": "公子"},
    }

    assert resolve_person_reference(person, "en") == "公子"


def test_missing_address_falls_back_to_localized_name_without_inference() -> None:
    person = {
        "id": "person:alex_example",
        "name": {"zh": "艾力克斯", "en": "Alex Example"},
        "gender": "male",
        "age": 12,
        "household_role": "son",
    }

    assert resolve_person_reference(person, "zh") == "艾力克斯"
    assert resolve_person_reference(person, "en") == "Alex Example"


def test_normal_render_prefers_person_address_but_id_mode_keeps_id() -> None:
    person = {
        "id": "person:pu_ba",
        "name": {"zh": "巴璞"},
        "address_as": {"zh": "太太"},
    }
    resolver = DisplayNameResolver([person])

    assert resolver.render(
        "person:pu_ba 已经回家。",
        "zh",
    ) == "太太 已经回家。"
    assert resolver.resolve("person:pu_ba", "zh", mode="id") == "person:pu_ba"
