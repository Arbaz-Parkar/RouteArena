"""Validated application settings loaded from environment variables or a local .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration; LLM fields are optional for offline operation."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RouteArena"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None


settings = Settings()
