from app.core.config import settings
from app.core.db import get_client, is_db_configured
from app.core.utils.time import utc_now
from app.health.schemas import HealthResponse


async def check_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        timestamp=utc_now(),
        database=await _check_database(),
    )


async def _check_database() -> str:
    if not is_db_configured():
        return "not_configured"
    try:
        await get_client().admin.command("ping")
        return "connected"
    except Exception:
        return "unavailable"
