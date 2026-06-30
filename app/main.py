from app.core.utils.log_config import init_logging

init_logging()

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError

from app.content_sync.change_stream import watch_sync_flags
from app.content_sync.connection_manager import connection_manager
from app.core.config import settings
from app.core.db import close_db, init_db, is_db_configured
from app.core.exceptions import (
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
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
        with contextlib.suppress(asyncio.CancelledError):
            await watcher_task
    await close_db()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)


@app.get("/")
async def root() -> dict[str, Any]:
    return {"status": "SUCCESS", "message": f"{settings.app_name} is running smoothly!"}


app.include_router(api_router, prefix="/api")
