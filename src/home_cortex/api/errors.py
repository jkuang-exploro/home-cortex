"""Stable HTTP errors, request IDs, and exception handlers."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from ..providers.ir import ModelProviderError
from .schemas import REQUEST_ID_HEADER


logger = logging.getLogger("uvicorn.error.home_cortex.api")


async def model_provider_error_handler(request: Request, error: ModelProviderError) -> JSONResponse:
    logger.warning("model_provider_error request_id=%s status=%d", request_id(request), error.status_code)
    return error_response(request, error.status_code, "model_provider_error", str(error))


class APIError(HTTPException):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: Any | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.details = details


def request_id(request: Request | None) -> str:
    if request is None:
        return "unknown"
    return getattr(request.state, "request_id", "unknown")


def error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: Any | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "request_id": request_id(request),
    }
    if details is not None:
        error["details"] = details
    response_headers = dict(headers or {})
    response_headers[REQUEST_ID_HEADER] = request_id(request)
    return JSONResponse(
        status_code=status_code,
        content={"error": error},
        headers=response_headers,
    )


async def http_error_handler(
    request: Request,
    error: StarletteHTTPException,
) -> JSONResponse:
    code = getattr(error, "code", f"http_{error.status_code}")
    details = getattr(error, "details", None)
    message = error.detail if isinstance(error.detail, str) else "Request failed"
    return error_response(
        request,
        error.status_code,
        code,
        message,
        details,
        headers=error.headers,
    )


async def validation_error_handler(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    details = [
        {
            "field": ".".join(str(part) for part in item["loc"]),
            "message": item["msg"],
            "type": item["type"],
        }
        for item in error.errors()
    ]
    logger.info(
        "request_validation_failed request_id=%s fields=%s types=%s",
        request_id(request),
        ",".join(detail["field"] for detail in details),
        ",".join(detail["type"] for detail in details),
    )
    return error_response(
        request,
        422,
        "request_validation_error",
        "Request validation failed",
        details,
    )


async def unexpected_error_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    logger.error(
        "unhandled_error request_id=%s exception_type=%s",
        request_id(request),
        type(error).__name__,
    )
    return error_response(
        request,
        500,
        "internal_server_error",
        "An unexpected server error occurred",
    )
