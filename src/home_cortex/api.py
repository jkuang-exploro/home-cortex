import asyncio
import json
import logging
import secrets
import time
from collections import OrderedDict
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
from .agents import (
    AgentDefinition,
    UnknownAgentError,
    get_agent,
    get_agent_by_display_name,
    list_agents,
)
from .config import Settings, get_settings
from .db import Database
from .edge_schema import EdgeSchemaRegistry
from .display import conversation_language
from .greetings import GreetingService
from .ingestion import ingest_directory
from .identity import resolve_user_entity_id
from .ollama import OllamaService
from .retrieval import RetrievalService
from .tools import ToolDispatcher

DEFAULT_AGENT_ID = "steward"
VIRTUAL_MODEL = get_agent(DEFAULT_AGENT_ID).display_name
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


class ConversationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = Field(
        default="en",
        min_length=2,
        max_length=35,
        pattern=r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$",
    )


class ConversationStore:
    """Keep bounded conversation initialization state for this API process."""

    def __init__(self, maximum: int = 1_000) -> None:
        self.maximum = maximum
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def create(
        self,
        *,
        agent_id: str,
        person_id: str | None,
        language: str,
        greeting: str,
    ) -> dict[str, Any]:
        conversation = {
            "id": uuid4().hex,
            "agent_id": agent_id,
            "person_id": person_id,
            "language": language,
            "greeting": greeting,
        }
        self._items[conversation["id"]] = conversation
        while len(self._items) > self.maximum:
            self._items.popitem(last=False)
        return dict(conversation)

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        conversation = self._items.get(conversation_id)
        return dict(conversation) if conversation is not None else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    database = Database(settings)
    await database.connect()
    app.state.database = database
    edge_registry = EdgeSchemaRegistry.from_directory(settings.edge_schema_dir)
    app.state.edge_registry = edge_registry
    retrieval = RetrievalService(
        database,
        settings.retrieval_limit,
        settings.data_dir,
        edge_registry,
    )
    app.state.retrieval = retrieval
    app.state.greetings = GreetingService(retrieval)
    app.state.conversations = ConversationStore()
    runtimes: dict[str, AgentService] = {}
    ollama_services: list[OllamaService] = []
    for definition in list_agents():
        if definition.model.provider != "ollama":
            raise RuntimeError(
                f"Unsupported model provider {definition.model.provider!r}"
            )
        model_name = definition.model.name or settings.ollama_model
        ollama = OllamaService(settings.ollama_url, model_name)
        ollama_services.append(ollama)
        runtimes[definition.id] = AgentService(
            ollama,
            ToolDispatcher(retrieval, definition.allowed_tools),
            system_prompt=definition.prompt,
            tools=definition.tool_definitions,
        )
    app.state.agents = runtimes
    app.state.agent = runtimes[DEFAULT_AGENT_ID]
    try:
        yield
    finally:
        for ollama in ollama_services:
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
    _authenticate_request(request)
    settings = get_settings()
    try:
        result = await ingest_directory(
            request.app.state.database,
            settings.data_dir,
            getattr(request.app.state, "edge_registry", None),
        )
        return {"status": "ok", **asdict(result)}
    except (FileNotFoundError, ValueError) as error:
        raise APIError(400, "ingestion_failed", str(error)) from error


@app.post("/v1/retrieve")
async def retrieve(body: dict[str, Any], request: Request) -> dict[str, Any]:
    _authenticate_request(request)
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
    return await _agent_chat(DEFAULT_AGENT_ID, body, request)


@app.post("/agent/{agent_id}/chat")
async def agent_chat(
    agent_id: str,
    body: dict[str, Any],
    request: Request,
) -> dict[str, Any]:
    return await _agent_chat(agent_id, body, request)


@app.post("/agent/{agent_id}/conversations", status_code=201)
async def create_agent_conversation(
    agent_id: str,
    body: ConversationCreateRequest,
    request: Request,
) -> dict[str, Any]:
    definition = _agent_definition(agent_id)
    user_entity = await _resolve_identity(request)
    greeting = await _greeting_service(request).resolve(
        definition,
        user_entity,
        body.language,
    )
    person_id = user_entity.get("id") if user_entity is not None else None
    conversation = _conversation_store(request).create(
        agent_id=definition.id,
        person_id=person_id if isinstance(person_id, str) else None,
        language=greeting.language,
        greeting=greeting.text,
    )
    return _conversation_response(conversation, definition)


@app.get("/agent/{agent_id}/conversations/{conversation_id}")
async def get_agent_conversation(
    agent_id: str,
    conversation_id: str,
    request: Request,
) -> dict[str, Any]:
    definition = _agent_definition(agent_id)
    user_entity = await _resolve_identity(request)
    person_id = user_entity.get("id") if user_entity is not None else None
    conversation = _authorize_conversation_access(
        _conversation_store(request).get(conversation_id),
        agent_id=definition.id,
        person_id=person_id if isinstance(person_id, str) else None,
    )
    return _conversation_response(conversation, definition)


async def _agent_chat(
    agent_id: str,
    body: dict[str, Any],
    request: Request,
) -> dict[str, Any]:
    definition = _agent_definition(agent_id)
    question = body.get("message") or body.get("question")
    if not isinstance(question, str) or not question.strip():
        raise APIError(422, "invalid_request", "Provide a non-empty 'message'")
    user_entity = await _resolve_identity(request)
    try:
        result = await _agent_runtime(request, definition).answer(
            question,
            request_id=_request_id(request),
            user_entity=user_entity,
        )
    except AgentLimitError as error:
        raise APIError(502, error.stop_reason, str(error)) from error
    return {
        "agent": {
            "id": definition.id,
            "display_name": definition.display_name,
        },
        "answer": result.answer,
        "steps": result.steps,
        "tool_calls": result.tool_calls,
        "stop_reason": result.stop_reason,
    }


@app.get("/v1/models")
async def models(request: Request) -> dict[str, Any]:
    _authenticate_request(request)
    return {
        "object": "list",
        "data": [
            {
                "id": definition.display_name,
                "object": "model",
                "created": MODEL_CREATED,
                "owned_by": "home-cortex",
            }
            for definition in list_agents()
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
):
    user_entity = await _resolve_identity(request)
    try:
        definition = get_agent_by_display_name(body.model)
    except UnknownAgentError:
        raise APIError(
            404,
            "model_not_found",
            f"Model {body.model!r} was not found",
        )
    agent = _agent_runtime(request, definition)
    completion_id = f"chatcmpl-{uuid4().hex}"
    created = int(time.time())
    messages = [
        message.model_dump()
        for message in body.messages
        if message.role != "system"
    ]
    if not messages or not any(message["role"] == "user" for message in messages):
        raise APIError(422, "invalid_request", "Provide at least one user message")
    greeting = None
    if _is_new_conversation(messages):
        greeting = await _greeting_service(request).resolve(
            definition,
            user_entity,
            conversation_language(messages),
        )
    if greeting is not None and _is_standalone_greeting(messages):
        if body.stream:
            return StreamingResponse(
                _stream_chat_completion(
                    completion_id,
                    created,
                    _single_answer(greeting.text),
                    request,
                    model=definition.display_name,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        return _chat_completion_response(
            completion_id,
            created,
            definition.display_name,
            greeting.text,
        )
    agent_messages = messages
    if body.stream:
        answer_stream = agent.stream_answer_messages(
            agent_messages,
            request_id=_request_id(request),
            user_entity=user_entity,
        )
        if greeting is not None:
            answer_stream = _prepend_answer(greeting.text, answer_stream)
        return StreamingResponse(
            _stream_chat_completion(
                completion_id,
                created,
                answer_stream,
                request,
                model=definition.display_name,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        result = await agent.answer_messages(
            agent_messages,
            request_id=_request_id(request),
            user_entity=user_entity,
        )
    except AgentLimitError as error:
        raise APIError(502, error.stop_reason, str(error)) from error

    return _chat_completion_response(
        completion_id,
        created,
        definition.display_name,
        _greeted_answer(
            greeting.text if greeting is not None else None,
            result.answer,
        ),
    )


def _chat_completion_response(
    completion_id: str,
    created: int,
    model: str,
    content: str,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
    }


def _is_new_conversation(messages: list[dict[str, Any]]) -> bool:
    user_messages = sum(message.get("role") == "user" for message in messages)
    has_assistant = any(message.get("role") == "assistant" for message in messages)
    return user_messages == 1 and not has_assistant


def _is_standalone_greeting(messages: list[dict[str, Any]]) -> bool:
    user_content = next(
        (
            str(message.get("content", "")).strip().casefold()
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )
    normalized = user_content.rstrip("!！.。?？,， ")
    return normalized in {"hello", "hi", "hey", "你好", "您好", "嗨"}


def _greeted_answer(greeting: str | None, answer: str) -> str:
    return f"{greeting}\n\n{answer}" if greeting else answer


async def _prepend_answer(
    greeting: str,
    answer_stream: AsyncIterator[str],
) -> AsyncIterator[str]:
    try:
        yield f"{greeting}\n\n"
        async for content in answer_stream:
            yield content
    finally:
        close = getattr(answer_stream, "aclose", None)
        if close is not None:
            with suppress(asyncio.CancelledError, Exception):
                await close()


async def _single_answer(content: str) -> AsyncIterator[str]:
    yield content


async def _stream_chat_completion(
    completion_id: str,
    created: int,
    answer_stream: AsyncIterator[str],
    request: Request | None = None,
    *,
    model: str = VIRTUAL_MODEL,
) -> AsyncIterator[str]:
    try:
        yield _sse_data(
            _chat_completion_chunk(
                completion_id,
                created,
                delta={"role": "assistant"},
                model=model,
            )
        )
        async for content in answer_stream:
            if content:
                yield _sse_data(
                    _chat_completion_chunk(
                        completion_id,
                        created,
                        delta={"content": content},
                        model=model,
                    )
                )
        yield _sse_data(
            _chat_completion_chunk(
                completion_id,
                created,
                delta={},
                finish_reason="stop",
                model=model,
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
    model: str = VIRTUAL_MODEL,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
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


def _request_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


def _authenticate_request(request: Request) -> None:
    """Verify the shared household API key when one is configured.

    V1 uses a single household key. It proves the caller is a trusted
    client (typically the Open WebUI proxy), not which person they are.
    """
    expected_key = _request_settings(request).cortex_api_key
    if expected_key is None:
        return
    authorization = request.headers.get("Authorization", "")
    scheme, _, supplied_key = authorization.partition(" ")
    if scheme.casefold() != "bearer" or not secrets.compare_digest(
        supplied_key,
        expected_key,
    ):
        raise APIError(
            401,
            "authentication_required",
            "A valid Cortex API key is required",
        )


def _mapped_person_id(request: Request) -> str | None:
    """Map trusted Open WebUI headers through CORTEX_IDENTITY_MAP.

    A client-supplied person record ID is not consulted. Anyone holding
    the household API key can present any mapped Open WebUI user header;
    per-person credentials are out of scope for V1.
    """
    settings = _request_settings(request)
    _authenticate_request(request)
    if not settings.cortex_identity_map:
        return None
    entity_id = resolve_user_entity_id(
        request.headers,
        settings.cortex_identity_map,
    )
    if entity_id is None:
        raise APIError(
            403,
            "identity_not_mapped",
            "The authenticated Open WebUI user is not mapped to a home-graph person",
        )
    return entity_id


async def _resolve_identity(request: Request) -> dict[str, Any] | None:
    """Load the mapped Person by exact record ID. Missing records fail closed."""
    entity_id = _mapped_person_id(request)
    if entity_id is None:
        return None
    try:
        entity = await request.app.state.retrieval.get_entity(entity_id)
    except ValueError:
        entity = None
    if entity is None or entity.get("id") != entity_id:
        logger.info(
            "identity_resolution request_id=%s success=false reason=record_not_found",
            _request_id(request),
        )
        raise APIError(
            403,
            "identity_record_not_found",
            "The mapped home-graph person record was not found",
        )
    logger.info(
        "identity_resolution request_id=%s success=true has_name=%s "
        "has_address_as=%s",
        _request_id(request),
        str("name" in entity).lower(),
        str("address_as" in entity).lower(),
    )
    return {
        key: entity[key]
        for key in ("id", "name", "address_as")
        if key in entity
    }


def _authorize_conversation_access(
    conversation: dict[str, Any] | None,
    *,
    agent_id: str,
    person_id: str | None,
) -> dict[str, Any]:
    """Owner-only conversation access. Cross-user and missing both 404."""
    if (
        conversation is None
        or conversation["agent_id"] != agent_id
        or conversation["person_id"] != person_id
    ):
        raise APIError(404, "conversation_not_found", "Conversation was not found")
    return conversation


def _agent_definition(agent_id: str) -> AgentDefinition:
    try:
        return get_agent(agent_id)
    except UnknownAgentError as error:
        raise APIError(
            404,
            "agent_not_found",
            f"Agent {agent_id!r} was not found",
        ) from error


def _agent_runtime(
    request: Request,
    definition: AgentDefinition,
) -> AgentService:
    runtimes = getattr(request.app.state, "agents", None)
    if isinstance(runtimes, dict) and definition.id in runtimes:
        return runtimes[definition.id]
    if definition.id == DEFAULT_AGENT_ID:
        runtime = getattr(request.app.state, "agent", None)
        if runtime is not None:
            return runtime
    raise RuntimeError(f"Agent runtime {definition.id!r} is not initialized")


def _greeting_service(request: Request) -> GreetingService:
    service = getattr(request.app.state, "greetings", None)
    if isinstance(service, GreetingService):
        return service
    raise RuntimeError("Greeting service is not initialized")


def _conversation_store(request: Request) -> ConversationStore:
    store = getattr(request.app.state, "conversations", None)
    if isinstance(store, ConversationStore):
        return store
    raise RuntimeError("Conversation store is not initialized")


def _conversation_response(
    conversation: dict[str, Any],
    definition: AgentDefinition,
) -> dict[str, Any]:
    return {
        "id": conversation["id"],
        "object": "agent.conversation",
        "agent": {
            "id": definition.id,
            "display_name": definition.display_name,
        },
        "language": conversation["language"],
        "greeting": conversation["greeting"],
    }


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
