from functools import lru_cache
from pathlib import Path

from pydantic import Field
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
