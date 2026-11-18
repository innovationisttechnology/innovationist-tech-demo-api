from fastapi import APIRouter

from app.content_sync.router import router as content_sync_router

api_router = APIRouter()

api_router.include_router(content_sync_router)
