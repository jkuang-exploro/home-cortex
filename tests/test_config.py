import pytest
from pydantic import ValidationError

from home_cortex.config import Settings


def test_identity_map_normalizes_email() -> None:
    settings = Settings(
        _env_file=None,
        ollama_model="test-model",
        cortex_api_key="test-key",
        cortex_identity_map={
            "email:Jian@Example.com": "person:jian_kuang",
        },
    )

    assert settings.cortex_identity_map == {
        "email:jian@example.com": "person:jian_kuang"
    }


def test_identity_map_requires_api_key() -> None:
    with pytest.raises(ValidationError, match="CORTEX_API_KEY is required"):
        Settings(
            _env_file=None,
            ollama_model="test-model",
            cortex_identity_map={
                "id:webui-user-123": "person:jian_kuang",
            },
        )


def test_calendar_bindings_require_person_ids_and_unique_calendar_ids() -> None:
    settings = Settings(
        _env_file=None,
        ollama_model="test-model",
        calendar_bindings=[
            {
                "id": "jian_primary",
                "person_id": "person:jian_kuang",
                "provider_calendar_id": "primary",
                "readers": ["person:pu_ba"],
            }
        ],
    )

    assert settings.calendar_timezone == "America/Los_Angeles"
    assert settings.calendar_bindings[0].id == "jian_primary"
    assert settings.calendar_bindings[0].readers == ("person:pu_ba",)


def test_partial_google_calendar_oauth_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Google Calendar OAuth"):
        Settings(
            _env_file=None,
            ollama_model="test-model",
            google_calendar_client_id="client-id",
        )


def test_calendar_bindings_parse_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CALENDAR_BINDINGS",
        (
            '[{"id":"jian_primary","person_id":"person:jian_kuang",'
            '"provider_calendar_id":"primary"}]'
        ),
    )

    settings = Settings(_env_file=None, ollama_model="test-model")

    assert settings.calendar_bindings[0].person_id == "person:jian_kuang"
    assert settings.google_calendar_client_secret is None


def test_openrouter_provider_requires_key_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    with pytest.raises(ValidationError, match="OPENROUTER_API_KEY"):
        Settings(_env_file=None, llm_provider="openrouter", openrouter_model="openai/gpt-4.1")
    with pytest.raises(ValidationError, match="OPENROUTER_MODEL"):
        Settings(
            _env_file=None,
            llm_provider="openrouter",
            openrouter_api_key="sk-or-test",
        )
    settings = Settings(
        _env_file=None,
        llm_provider="openrouter",
        openrouter_api_key="sk-or-test",
        openrouter_model="anthropic/claude-sonnet-4",
    )
    assert settings.llm_provider == "openrouter"
    assert settings.ollama_model is None
    assert settings.openrouter_model == "anthropic/claude-sonnet-4"


def test_ollama_provider_still_requires_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with pytest.raises(ValidationError, match="OLLAMA_MODEL"):
        Settings(_env_file=None)


def test_unknown_calendar_timezone_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Unknown calendar timezone"):
        Settings(
            _env_file=None,
            ollama_model="test-model",
            calendar_timezone="Not/A_Zone",
        )


@pytest.mark.parametrize(
    "identity_map",
    [
        {"Jian Kuang": "person:jian_kuang"},
        {"id:webui-user-123": "address:fort_cerritos"},
    ],
)
def test_identity_map_rejects_unsafe_keys_and_non_person_values(
    identity_map: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            ollama_model="test-model",
            cortex_api_key="test-key",
            cortex_identity_map=identity_map,
        )
