from fastapi import APIRouter

from app.content_sync.router import router as content_sync_router
from app.ziza_chat.router import router as ziza_chat_router

api_router = APIRouter()

api_router.include_router(content_sync_router)
api_router.include_router(ziza_chat_router)
