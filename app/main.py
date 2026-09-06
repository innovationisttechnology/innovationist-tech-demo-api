from dotenv import load_dotenv

from app.core.utils.log_config import init_logging

# Provider SDKs (pydantic-ai, etc.) read keys like ANTHROPIC_API_KEY straight
# from the process environment, which pydantic-settings does not populate —
# so .env has to be loaded here. Real env vars win, keeping Docker unaffected.
load_dotenv()

init_logging()

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.content_sync.change_stream import watch_sync_flags
from app.content_sync.connection_manager import connection_manager
from app.core.config import settings
from app.core.db import close_db, init_db, is_db_configured
from app.core.exceptions import (
    ErrorResponse,
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.routes.routes import api_router
from app.ziza_chat.service import SessionLimitError
from app.ziza_chat.vector_store.index import ensure_vector_index

logger = logging.getLogger(__name__)


def warn_about_missing_provider_key() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error(
            "ANTHROPIC_API_KEY is not set — the /api/ziza endpoints will fail "
            "on every request. Set it in the deployment environment."
        )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    warn_about_missing_provider_key()
    watcher_task: asyncio.Task[None] | None = None
    if is_db_configured():
        await init_db()
        logger.info("Database initialized")
        await ensure_vector_index()
        watcher_task = asyncio.create_task(watch_sync_flags(connection_manager))
    else:
        logger.warning("MONGO_URI not set — running without a database")
    yield
    connection_manager.close_all()
    if watcher_task is not None:
        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass
        except Exception:
            # The watcher only restarts itself for transient database errors;
            # anything else ends the task and is re-raised here. Shutdown must
            # still complete, so it is reported rather than propagated.
            logger.exception("Sync flag watcher had already failed")
    await close_db()



_expose_api_docs = settings.environment != "production"

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if _expose_api_docs else None,
    redoc_url="/redoc" if _expose_api_docs else None,
    openapi_url="/openapi.json" if _expose_api_docs else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def session_limit_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=429,
        content=ErrorResponse(
            error="session_limit", message=str(exc), status_code=429
        ).model_dump(),
    )


app.add_exception_handler(SessionLimitError, session_limit_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)


@app.get("/")
async def root() -> dict[str, Any]:
    return {"status": "SUCCESS", "message": f"{settings.app_name} is running smoothly!"}


app.include_router(api_router, prefix="/api")
