from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class CalendarBindingSettings(BaseModel):
    """Maps a Cortex calendar ID to a provider calendar and authorized readers."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    person_id: str = Field(pattern=r"^person:[A-Za-z0-9_-]+$")
    provider_calendar_id: str = Field(min_length=1)
    readers: tuple[str, ...] = ()

    @field_validator("readers")
    @classmethod
    def validate_readers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for reader in value:
            if not reader.startswith("person:"):
                raise ValueError("calendar readers must be person record IDs")
        return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "home-cortex"
    surreal_url: str = "ws://surrealdb:8000"
    surreal_user: str = "root"
    surreal_pass: str = "root"
    surreal_namespace: str = "home_cortex"
    surreal_database: str = "home_cortex"
    ollama_url: str = "http://ollama:11434"
    ollama_model: str | None = None
    llm_provider: Literal["ollama", "openrouter"] = "ollama"
    openrouter_api_key: SecretStr | None = None
    openrouter_model: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_http_referer: str | None = None
    openrouter_app_title: str = "home-cortex"
    data_dir: Path = Path("/app/data")
    edge_schema_dir: Path = Path("/app/schemas/edge")
    retrieval_limit: int = Field(default=100, ge=1, le=1000)
    cortex_api_key: str | None = None
    cortex_identity_map: dict[str, str] = Field(default_factory=dict)
    google_calendar_client_id: str | None = None
    google_calendar_client_secret: SecretStr | None = None
    google_calendar_refresh_token: SecretStr | None = None
    calendar_timezone: str = "America/Los_Angeles"
    calendar_bindings: tuple[CalendarBindingSettings, ...] = ()

    @field_validator(
        "cortex_api_key",
        "google_calendar_client_id",
        "openrouter_http_referer",
        "openrouter_model",
        "ollama_model",
        mode="before",
    )
    @classmethod
    def normalize_optional_string(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator(
        "google_calendar_client_secret",
        "google_calendar_refresh_token",
        "openrouter_api_key",
        mode="before",
    )
    @classmethod
    def normalize_optional_secret(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("calendar_timezone")
    @classmethod
    def validate_calendar_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"Unknown calendar timezone {value!r}") from error
        return value

    @field_validator("calendar_bindings")
    @classmethod
    def validate_calendar_bindings(
        cls,
        value: tuple[CalendarBindingSettings, ...],
    ) -> tuple[CalendarBindingSettings, ...]:
        ids = [binding.id for binding in value]
        if len(ids) != len(set(ids)):
            raise ValueError("calendar binding IDs must be unique")
        return value

    @field_validator("cortex_identity_map")
    @classmethod
    def validate_identity_map(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for external_identity, entity_id in value.items():
            key = external_identity.strip()
            if not key.startswith(("id:", "email:")):
                raise ValueError(
                    "identity-map keys must start with 'id:' or 'email:'"
                )
            if key.startswith("email:"):
                key = key.casefold()
            entity_id = entity_id.strip()
            if not entity_id.startswith("person:"):
                raise ValueError(
                    "identity-map values must be person record IDs"
                )
            normalized[key] = entity_id
        return normalized

    @model_validator(mode="after")
    def require_api_key_for_identity_mapping(self) -> "Settings":
        if self.cortex_identity_map and self.cortex_api_key is None:
            raise ValueError(
                "CORTEX_API_KEY is required when CORTEX_IDENTITY_MAP is set"
            )
        if self.llm_provider == "ollama" and not self.ollama_model:
            raise ValueError("OLLAMA_MODEL is required when LLM_PROVIDER=ollama")
        if self.llm_provider == "openrouter":
            if self.openrouter_api_key is None:
                raise ValueError(
                    "OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter"
                )
            if not self.openrouter_model:
                raise ValueError(
                    "OPENROUTER_MODEL is required when LLM_PROVIDER=openrouter"
                )
        credentials = (
            self.google_calendar_client_id,
            self.google_calendar_client_secret,
            self.google_calendar_refresh_token,
        )
        present = [item for item in credentials if item]
        if present and len(present) != 3:
            raise ValueError(
                "Google Calendar OAuth requires client id, client secret, "
                "and refresh token"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
