from home_cortex.identity import resolve_user_entity_id


def test_resolves_openwebui_user_id_before_email() -> None:
    identity_map = {
        "id:user-123": "person:jian_kuang",
        "email:jian@example.com": "person:other_person",
    }

    entity_id = resolve_user_entity_id(
        {
            "X-OpenWebUI-User-Id": "user-123",
            "X-OpenWebUI-User-Email": "jian@example.com",
        },
        identity_map,
    )

    assert entity_id == "person:jian_kuang"


def test_resolves_email_case_insensitively() -> None:
    entity_id = resolve_user_entity_id(
        {"X-OpenWebUI-User-Email": "Jian@Example.com"},
        {"email:jian@example.com": "person:jian_kuang"},
    )

    assert entity_id == "person:jian_kuang"


def test_unknown_openwebui_user_is_not_resolved() -> None:
    assert (
        resolve_user_entity_id(
            {"X-OpenWebUI-User-Id": "unknown"},
            {"id:user-123": "person:jian_kuang"},
        )
        is None
    )


def test_client_supplied_person_id_headers_are_ignored() -> None:
    assert (
        resolve_user_entity_id(
            {
                "X-Identity": "person:jian_kuang",
                "X-Person-Id": "person:jian_kuang",
                "X-OpenWebUI-User-Id": "person:jian_kuang",
            },
            {"id:user-123": "person:jian_kuang"},
        )
        is None
    )
