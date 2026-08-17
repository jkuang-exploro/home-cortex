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


@pytest.mark.parametrize(
    "identity_map",
    [
        {"Jian Kuang": "person:jian_kuang"},
        {"id:webui-user-123": "location:fort_cerritos"},
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
