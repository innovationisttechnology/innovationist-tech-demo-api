from fastapi import APIRouter

from app.health.schemas import HealthResponse
from app.health.service import check_health

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health() -> HealthResponse:
    return await check_health()
