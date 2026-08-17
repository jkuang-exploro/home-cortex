from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    ollama_model: str
    data_dir: Path = Path("/app/data")
    retrieval_limit: int = Field(default=100, ge=1, le=1000)
    cortex_api_key: str | None = None
    cortex_identity_map: dict[str, str] = Field(default_factory=dict)

    @field_validator("cortex_api_key", mode="before")
    @classmethod
    def normalize_api_key(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
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
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
