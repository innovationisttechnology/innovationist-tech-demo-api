from app.core.utils.log_config import init_logging

init_logging()

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.content_sync.change_stream import watch_sync_flags
from app.content_sync.connection_manager import connection_manager
from app.core.config import settings
from app.core.db import close_db, init_db, is_db_configured
from app.core.exceptions import register_exception_handlers
from app.routes.routes import api_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    watcher_task: asyncio.Task[None] | None = None
    if is_db_configured():
        await init_db()
        logger.info("Database initialized")
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

register_exception_handlers(app)


@app.get("/")
async def root() -> dict[str, Any]:
    return {"status": "SUCCESS", "message": f"{settings.app_name} is running smoothly!"}


app.include_router(api_router, prefix="/api")
