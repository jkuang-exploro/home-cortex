import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, StreamingResponse

from . import __version__
from .agent import AgentLimitError, AgentService, AgentStreamingError
from .config import get_settings
from .db import Database
from .ingestion import ingest_directory
from .ollama import OllamaService
from .retrieval import RetrievalService
from .tools import ToolDispatcher

VIRTUAL_MODEL = "home-cortex"
MODEL_CREATED = int(time.time())
REQUEST_ID_HEADER = "X-Request-ID"

logger = logging.getLogger("uvicorn.error.home_cortex.api")


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


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    database = Database(settings)
    ollama = OllamaService(settings.ollama_url, settings.ollama_model)
    await database.connect()
    app.state.database = database
    app.state.ollama = ollama
    retrieval = RetrievalService(
        database,
        settings.retrieval_limit,
        settings.data_dir,
    )
    app.state.retrieval = retrieval
    app.state.agent = AgentService(ollama, ToolDispatcher(retrieval))
    try:
        yield
    finally:
        await ollama.close()
        await database.close()


app = FastAPI(
    title="Home Cortex API",
    version=__version__,
    description="Graph-grounded RAG service for SurrealDB.",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - started) * 1_000
        logger.info(
            "request_complete request_id=%s method=%s path=%s status=500 "
            "duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise
    duration_ms = (time.perf_counter() - started) * 1_000
    response.headers[REQUEST_ID_HEADER] = request_id
    logger.info(
        "request_complete request_id=%s method=%s path=%s status=%d "
        "duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(
    request: Request,
    error: StarletteHTTPException,
) -> JSONResponse:
    code = getattr(error, "code", f"http_{error.status_code}")
    details = getattr(error, "details", None)
    message = error.detail if isinstance(error.detail, str) else "Request failed"
    return _error_response(
        request,
        error.status_code,
        code,
        message,
        details,
        headers=error.headers,
    )


@app.exception_handler(RequestValidationError)
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
    return _error_response(
        request,
        422,
        "request_validation_error",
        "Request validation failed",
        details,
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    logger.error(
        "unhandled_error request_id=%s exception_type=%s",
        _request_id(request),
        type(error).__name__,
    )
    return _error_response(
        request,
        500,
        "internal_server_error",
        "An unexpected server error occurred",
    )


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    try:
        surreal_version = await request.app.state.database.version()
        return {
            "status": "ok",
            "surrealdb": surreal_version,
        }
    except Exception as error:
        logger.warning(
            "health_check_failed request_id=%s dependency=surrealdb "
            "exception_type=%s",
            _request_id(request),
            type(error).__name__,
        )
        raise APIError(
            503,
            "database_unavailable",
            "SurrealDB health check failed",
        ) from error


@app.post("/admin/ingest")
async def ingest(request: Request) -> dict[str, Any]:
    settings = get_settings()
    try:
        result = await ingest_directory(request.app.state.database, settings.data_dir)
        return {"status": "ok", **asdict(result)}
    except (FileNotFoundError, ValueError) as error:
        raise APIError(400, "ingestion_failed", str(error)) from error


@app.post("/v1/retrieve")
async def retrieve(body: dict[str, Any], request: Request) -> dict[str, Any]:
    question = body.get("query") or body.get("question")
    if not isinstance(question, str) or not question.strip():
        raise APIError(422, "invalid_request", "Provide a non-empty 'query'")
    result = await request.app.state.retrieval.retrieve(question.strip())
    return {
        "query": result.question,
        "nodes": result.nodes,
        "edges": result.edges,
        "context": result.text,
    }


@app.post("/v1/chat")
async def chat(body: dict[str, Any], request: Request) -> dict[str, Any]:
    question = body.get("message") or body.get("question")
    if not isinstance(question, str) or not question.strip():
        raise APIError(422, "invalid_request", "Provide a non-empty 'message'")
    try:
        result = await request.app.state.agent.answer(
            question,
            request_id=_request_id(request),
        )
    except AgentLimitError as error:
        raise APIError(502, error.stop_reason, str(error)) from error
    return {
        "answer": result.answer,
        "steps": result.steps,
        "tool_calls": result.tool_calls,
        "stop_reason": result.stop_reason,
    }


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": VIRTUAL_MODEL,
                "object": "model",
                "created": MODEL_CREATED,
                "owned_by": "home-cortex",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
):
    if body.model != VIRTUAL_MODEL:
        raise APIError(
            404,
            "model_not_found",
            f"Model {body.model!r} was not found",
        )
    completion_id = f"chatcmpl-{uuid4().hex}"
    created = int(time.time())
    if body.stream:
        answer_stream = request.app.state.agent.stream_answer_messages(
            [message.model_dump() for message in body.messages],
            request_id=_request_id(request),
        )
        return StreamingResponse(
            _stream_chat_completion(
                completion_id,
                created,
                answer_stream,
                request,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        result = await request.app.state.agent.answer_messages(
            [message.model_dump() for message in body.messages],
            request_id=_request_id(request),
        )
    except AgentLimitError as error:
        raise APIError(502, error.stop_reason, str(error)) from error

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": VIRTUAL_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.answer,
                },
                "finish_reason": "stop",
            }
        ],
    }


async def _stream_chat_completion(
    completion_id: str,
    created: int,
    answer_stream: AsyncIterator[str],
    request: Request | None = None,
) -> AsyncIterator[str]:
    try:
        yield _sse_data(
            _chat_completion_chunk(
                completion_id,
                created,
                delta={"role": "assistant"},
            )
        )
        async for content in answer_stream:
            if content:
                yield _sse_data(
                    _chat_completion_chunk(
                        completion_id,
                        created,
                        delta={"content": content},
                    )
                )
        yield _sse_data(
            _chat_completion_chunk(
                completion_id,
                created,
                delta={},
                finish_reason="stop",
            )
        )
        yield "data: [DONE]\n\n"
    except asyncio.CancelledError:
        if request is not None:
            _log_client_disconnect(request, phase="ollama_stream")
        raise
    except (AgentLimitError, AgentStreamingError) as error:
        yield _sse_error(
            error.stop_reason,
            str(error),
            request,
        )
        yield "data: [DONE]\n\n"
    except Exception as error:
        logger.error(
            "stream_error request_id=%s exception_type=%s",
            _request_id(request) if request is not None else "unknown",
            type(error).__name__,
        )
        yield _sse_error(
            "internal_server_error",
            "An unexpected server error occurred",
            request,
        )
        yield "data: [DONE]\n\n"
    finally:
        close = getattr(answer_stream, "aclose", None)
        if close is not None:
            with suppress(asyncio.CancelledError, Exception):
                await close()


def _chat_completion_chunk(
    completion_id: str,
    created: int,
    *,
    delta: dict[str, str],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": VIRTUAL_MODEL,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _sse_data(value: dict[str, Any]) -> str:
    return f"data: {json.dumps(value, ensure_ascii=False)}\n\n"


def _sse_error(
    code: str,
    message: str,
    request: Request | None,
) -> str:
    return _sse_data(
        {
            "error": {
                "code": code,
                "message": message,
                "request_id": (
                    _request_id(request) if request is not None else "unknown"
                ),
            }
        }
    )


def _log_client_disconnect(request: Request, *, phase: str) -> None:
    logger.info(
        "stream_cancelled request_id=%s phase=%s",
        _request_id(request),
        phase,
    )


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: Any | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "request_id": request_id,
    }
    if details is not None:
        error["details"] = details
    response_headers = dict(headers or {})
    response_headers[REQUEST_ID_HEADER] = request_id
    return JSONResponse(
        status_code=status_code,
        content={"error": error},
        headers=response_headers,
    )
