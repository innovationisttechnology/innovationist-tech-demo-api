from fastapi import APIRouter

from app.content_sync.router import router as content_sync_router
from app.health.router import router as health_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(content_sync_router)
