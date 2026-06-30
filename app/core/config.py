from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Innovationist Tech Demo API"
    app_version: str = "1.0.0"
    environment: str = "development"
    port: int = 8181

    # MongoDB — optional for now. Set MONGO_URI to enable database features.
    mongo_uri: str | None = None
    mongo_db_name: str = "innovationist_demo"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
