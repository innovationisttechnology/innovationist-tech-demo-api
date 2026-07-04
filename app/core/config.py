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

    # Comma-separated list of origins allowed to call the API (the demo UI).
    # When unset, defaults depend on `environment`: the deployed demo in
    # production, localhost in development. Override via the CORS_ORIGINS env var.
    cors_origins: str | None = None

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_origins:
            origins = [origin.strip() for origin in self.cors_origins.split(",")]
        elif self.environment == "production":
            origins = ["https://demo.innovationisttech.com"]
        else:
            origins = ["http://localhost:3001"]
        # An Origin is scheme+host+port only — strip any trailing slash/path so
        # a stray "https://site.com/" still matches the browser's Origin header.
        normalized = [origin.rstrip("/") for origin in origins if origin.strip()]
        if "*" in normalized:
            raise ValueError("CORS_ORIGINS cannot be '*' — list explicit origins")
        return normalized

    # MongoDB — optional for now. Set MONGO_URI to enable database features.
    mongo_uri: str | None = None
    mongo_db_name: str = "innovationist_demo"

    # How long soft-deleted flags (tombstones) linger before MongoDB's TTL
    # monitor purges them. Default 1 hour.
    tombstone_ttl_seconds: int = 3600

    # How long a flag survives without any activity before it's purged, so
    # abandoned demo sessions self-clean. Refreshed on every create/edit/toggle
    # via updated_at. Default 8 hours (a workday).
    session_ttl_seconds: int = 28800


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
