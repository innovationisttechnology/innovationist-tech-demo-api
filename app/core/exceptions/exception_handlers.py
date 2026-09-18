import logging
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)

from .errors import ErrorResponse, SessionLimitError

logger = logging.getLogger(__name__)

MODEL_UNAVAILABLE_MESSAGE = (
    "The assistant is unavailable right now. Please try again in a moment."
)


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


def session_limit_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=429,
        content=ErrorResponse(
            error="session_limit", message=str(exc), status_code=429
        ).model_dump(),
    )


def model_error_status(exc: Exception) -> int:
    if isinstance(exc, ModelHTTPError) and exc.status_code == 429:
        return 429
    return 503


def model_error_handler(request: Request, exc: Exception) -> Response:
    logger.warning("model call failed: %s: %s", type(exc).__name__, exc)
    status_code = model_error_status(exc)
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error="model_unavailable",
            message=MODEL_UNAVAILABLE_MESSAGE,
            status_code=status_code,
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
    app.add_exception_handler(SessionLimitError, session_limit_handler)
    app.add_exception_handler(ModelAPIError, model_error_handler)
    app.add_exception_handler(UsageLimitExceeded, model_error_handler)
    app.add_exception_handler(UnexpectedModelBehavior, model_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
