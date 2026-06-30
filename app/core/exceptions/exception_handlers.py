from typing import cast

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from .errors import ErrorResponse


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
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="Something went wrong",
            status_code=500,
        ).model_dump(),
    )
