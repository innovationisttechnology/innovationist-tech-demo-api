import logging
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from .errors import ErrorResponse

logger = logging.getLogger(__name__)


def http_exception_handler(request: Request, exc: Exception) -> Response:
    http_exc = cast(HTTPException, exc)
    return JSONResponse(
        status_code=http_exc.status_code,
        content=ErrorResponse(
            error="http_error",
            message=str(http_exc.detail),
            status_code=http_exc.status_code,
        ).model_dump(),
    )


def validation_exception_handler(request: Request, exc: Exception) -> Response:
    val_exc = cast(RequestValidationError, exc)
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="validation_error",
            message="Request validation failed",
            status_code=422,
            details=val_exc.errors(),
        ).model_dump(),
    )


def unhandled_exception_handler(request: Request, exc: Exception) -> Response:
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="Something went wrong",
            status_code=500,
        ).model_dump(),
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
