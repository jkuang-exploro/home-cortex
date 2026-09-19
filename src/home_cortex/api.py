import asyncio
import json
import logging
import os
import secrets
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import FileResponse, JSONResponse, StreamingResponse

from . import __version__
from .agent_service import AgentLimitError, AgentService, AgentStreamingError
from .agents import (
    AgentDefinition,
    UnknownAgentError,
    get_agent,
    get_agent_by_display_name,
    list_agents,
)
from .config import Settings, get_settings
from .db import Database
from .request_tracing import RequestTraceMiddleware
from .edge_schema import EdgeSchemaRegistry
from .schema_catalog import RuntimeSchemaCatalog
from .display import conversation_language
from .text import latest_user_message
from .greetings import GreetingService
from .export import export_directory
from .ingestion import ingest_directory
from .conversations import ConversationStore, SurrealConversationStore
from .gui_session import (
    COOKIE_NAME as GUI_COOKIE_NAME,
    SESSION_SECONDS as GUI_SESSION_SECONDS,
    parse_session,
    session_token as gui_session_token,
    valid_session as valid_gui_session,
)
from .identity import (
    OPENWEBUI_USER_EMAIL_HEADER,
    OPENWEBUI_USER_ID_HEADER,
    resolve_user_entity_id,
)
from .ollama import language_model_from_settings
from .retrieval import RetrievalService
from .calendar import calendar_service_from_settings
from .tools import ToolDispatcher
from .writing import ItemWritingService
from .vision.camera.errors import CameraError
from .vision.camera.hub import SharedTapoHub
from .vision.camera.sources import CameraSource, source_from_settings
from .vision.relay import (
    COOKIE_NAME,
    SESSION_SECONDS,
    normalize_stream_source,
    open_relay,
    session_source,
    session_token,
    valid_session,
)

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
    # OpenAI-compatible clients may retain a null/empty assistant placeholder
    # after an interrupted or failed streamed response. It is safe to accept
    # that history entry and discard it before invoking the agent. User
    # messages are still required to contain text below.
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)


class ConversationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = Field(
        default="en",
        min_length=2,
        max_length=35,
        pattern=r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$",
    )
    model: str | None = Field(default=None, min_length=1, max_length=256)


class ConversationMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=32_000)
    stream: bool = True


class SessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str | None = Field(default=None, min_length=3, max_length=320)
    user_id: str | None = Field(default=None, min_length=1, max_length=128)


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
    schema_catalog = RuntimeSchemaCatalog.from_data_dir(
        settings.data_dir,
        edge_registry,
    )
    app.state.schema_catalog = schema_catalog
    writing = ItemWritingService(database, schema_catalog, edge_registry)
    app.state.writing = writing
    app.state.greetings = GreetingService(retrieval)
    app.state.conversations = SurrealConversationStore(database)
    calendar = calendar_service_from_settings(settings)
    app.state.calendar = calendar
    runtimes: dict[str, AgentService] = {}
    language_models = []
    for definition in list_agents():
        if definition.model.provider not in {"ollama", "openrouter"}:
            raise RuntimeError(
                f"Unsupported model provider {definition.model.provider!r}"
            )
        language_model = language_model_from_settings(
            settings, definition.model.name
        )
        language_models.append(language_model)
        runtimes[definition.id] = AgentService(
            language_model,
            ToolDispatcher(
                retrieval,
                definition.allowed_tools,
                calendar=calendar,
                writing=writing,
                household_id=definition.settings.get("home_entity_id"),
            ),
            system_prompt=definition.prompt,
            tools=definition.tool_definitions,
            localized_identity=definition.settings.get("localized_identity"),
            assistant_id=definition.id,
            assistant_display_name=definition.display_name,
            home_entity_id=definition.settings.get("home_entity_id"),
            household_timezone=settings.calendar_timezone,
            schema_catalog=schema_catalog,
        )
    app.state.agents = runtimes
    app.state.agent = runtimes[DEFAULT_AGENT_ID]
    app.state.camera_hub = SharedTapoHub()
    try:
        yield
    finally:
        await app.state.camera_hub.close()
        for language_model in language_models:
            await language_model.close()
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


app.add_middleware(
    RequestTraceMiddleware, enabled=os.environ.get("CORTEX_PROFILE_REQUESTS") == "1"
)


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
    logger.info(
        "request_validation_failed request_id=%s fields=%s types=%s",
        _request_id(request),
        ",".join(detail["field"] for detail in details),
        ",".join(detail["type"] for detail in details),
    )
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


@app.get("/vision", response_class=FileResponse, include_in_schema=False)
async def vision_page() -> FileResponse:
    """Serve the public, data-free shell; data endpoints retain authentication."""
    return FileResponse(
        Path(__file__).parent / "vision" / "web" / "index.html",
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/vision/assets/vision.js", response_class=FileResponse, include_in_schema=False)
async def vision_script() -> FileResponse:
    """Serve the build-free player without exposing arbitrary package files."""
    return FileResponse(
        Path(__file__).parent / "vision" / "web" / "vision.js",
        media_type="text/javascript",
        headers={"Cache-Control": "no-store"},
    )


def _vision_signing_key(request: Request) -> str:
    return _request_settings(request).cortex_api_key or "home-cortex-vision"


def _authenticate_vision(request: Request) -> None:
    key = _request_settings(request).cortex_api_key
    if key and valid_session(request.cookies.get(COOKIE_NAME, ""), key):
        return
    _authenticate_request(request)


def _vision_stream_url(request: Request, *, source: str | None = None) -> str:
    if source and source.strip():
        try:
            return normalize_stream_source(source)
        except ValueError as error:
            raise APIError(422, "invalid_stream_source", str(error)) from error
    key = _vision_signing_key(request)
    bound = session_source(request.cookies.get(COOKIE_NAME, ""), key)
    if bound:
        return bound
    url = getattr(_request_settings(request), "vision_stream_url", None)
    if url:
        return str(url)
    raise APIError(
        503,
        "vision_not_configured",
        "Provide the camera IP or URL when connecting",
    )


@app.post("/vision/session", include_in_schema=False)
async def vision_session(request: Request) -> JSONResponse:
    _authenticate_vision(request)
    payload: dict[str, Any] = {}
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        with suppress(Exception):
            body = await request.json()
            if isinstance(body, dict):
                payload = body
    configured = source_from_settings(_request_settings(request))
    if configured is not None and configured.kind == "tapo":
        url = ""
    else:
        url = _vision_stream_url(
            request,
            source=payload.get("source") if isinstance(payload.get("source"), str) else None,
        )
    response = JSONResponse({"stream_url": "/vision/stream"}, headers={"Cache-Control": "no-store"})
    response.set_cookie(
        COOKIE_NAME,
        session_token(_vision_signing_key(request), url),
        max_age=SESSION_SECONDS,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
        path="/vision",
    )
    return response


@app.get("/vision/stream", include_in_schema=False)
async def vision_stream(request: Request) -> StreamingResponse:
    _authenticate_vision(request)
    try:
        return await open_relay(
            _vision_camera_source(request),
            hub=getattr(request.app.state, "camera_hub", None),
        )
    except CameraError as error:
        status = 503 if error.code == "vision_not_configured" else 502
        raise APIError(status, error.code, error.message) from None


def _vision_camera_source(request: Request) -> str | CameraSource:
    settings = _request_settings(request)
    configured = source_from_settings(settings)
    if configured is not None and configured.kind == "tapo":
        return configured
    return _vision_stream_url(request)


@app.post("/session")
async def create_session(body: SessionRequest, request: Request) -> JSONResponse:
    _authenticate_bearer(request)
    settings = _request_settings(request)
    kind, value = _session_identity(body, settings)
    payload: dict[str, Any] = {"object": "session"}
    if kind == "email":
        payload["email"] = value
    elif kind == "id":
        payload["user_id"] = value
    response = JSONResponse(payload)
    key = settings.cortex_api_key
    if key:
        response.set_cookie(
            GUI_COOKIE_NAME,
            gui_session_token(key, kind, value),
            max_age=GUI_SESSION_SECONDS,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="strict",
            path="/",
        )
    return response


@app.get("/session")
async def read_session(request: Request) -> dict[str, Any]:
    _authenticate_request(request)
    settings = _request_settings(request)
    key = settings.cortex_api_key
    parsed = parse_session(request.cookies.get(GUI_COOKIE_NAME, ""), key) if key else None
    payload: dict[str, Any] = {"object": "session"}
    if parsed is None:
        if key is None:
            payload["anonymous"] = True
            return payload
        user_id = request.headers.get(OPENWEBUI_USER_ID_HEADER)
        email = request.headers.get(OPENWEBUI_USER_EMAIL_HEADER)
        if user_id:
            payload["user_id"] = user_id
        if email:
            payload["email"] = email
        if not user_id and not email:
            payload["anonymous"] = True
        return payload
    kind, value = parsed
    if kind == "email":
        payload["email"] = value
    elif kind == "id":
        payload["user_id"] = value
    else:
        payload["anonymous"] = True
    return payload


@app.delete("/session")
async def delete_session() -> JSONResponse:
    response = JSONResponse({"status": "ok"})
    response.delete_cookie(GUI_COOKIE_NAME, path="/")
    return response


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


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_dir: Path


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


@app.post("/admin/export")
async def export(body: ExportRequest, request: Request) -> dict[str, Any]:
    _authenticate_request(request)
    if not body.target_dir.is_absolute():
        raise APIError(
            400,
            "export_failed",
            "target_dir must be an absolute path on the API server. "
            "From Docker Compose use /app/export (host directory tmp/db-export).",
        )
    try:
        result = await export_directory(
            request.app.state.database,
            body.target_dir,
            getattr(request.app.state, "edge_registry", None),
        )
        return {"status": "ok", **asdict(result)}
    except (FileNotFoundError, ValueError, OSError) as error:
        raise APIError(400, "export_failed", str(error)) from error


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
    conversation = await _conversation_store(request).create(
        agent_id=definition.id,
        model=definition.display_name,
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
        await _conversation_store(request).get(conversation_id),
        agent_id=definition.id,
        person_id=person_id if isinstance(person_id, str) else None,
    )
    return _conversation_response(conversation, definition)


@app.get("/conversations")
async def list_conversations(request: Request) -> dict[str, Any]:
    user_entity = await _resolve_identity(request)
    person_id = user_entity.get("id") if user_entity is not None else None
    items = await _conversation_store(request).list_for_person(
        person_id if isinstance(person_id, str) else None
    )
    return {"object": "list", "data": items}


@app.post("/conversations", status_code=201)
async def create_conversation(
    body: ConversationCreateRequest,
    request: Request,
) -> dict[str, Any]:
    user_entity = await _resolve_identity(request)
    person_id = user_entity.get("id") if user_entity is not None else None
    model_id = (body.model or VIRTUAL_MODEL).strip()
    agent = _agent_for_model(model_id)
    if agent is not None:
        greeting = await _greeting_service(request).resolve(
            agent,
            user_entity,
            body.language,
        )
        conversation = await _conversation_store(request).create(
            agent_id=agent.id,
            model=agent.display_name,
            person_id=person_id if isinstance(person_id, str) else None,
            language=greeting.language,
            greeting=greeting.text,
        )
        return _transcript_response(conversation, messages=conversation.get("messages", []))
    await _require_bare_model(request, model_id)
    conversation = await _conversation_store(request).create(
        agent_id=None,
        model=model_id,
        person_id=person_id if isinstance(person_id, str) else None,
        language=body.language,
        greeting=None,
    )
    return _transcript_response(conversation, messages=[])


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request) -> dict[str, Any]:
    conversation = await _owned_conversation(request, conversation_id)
    return _transcript_response(conversation, messages=conversation.get("messages", []))


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, request: Request) -> dict[str, Any]:
    await _owned_conversation(request, conversation_id)
    await _conversation_store(request).delete(conversation_id)
    return {"status": "ok", "id": conversation_id}


@app.post("/conversations/{conversation_id}/messages")
async def post_conversation_message(
    conversation_id: str,
    body: ConversationMessageRequest,
    request: Request,
):
    conversation = await _owned_conversation(request, conversation_id)
    store = _conversation_store(request)
    content = body.content.strip()
    if not content:
        raise APIError(422, "invalid_request", "Provide a non-empty 'content'")
    await store.append_message(conversation_id, role="user", content=content)
    conversation = await _owned_conversation(request, conversation_id)
    messages = [
        {"role": item["role"], "content": item["content"]}
        for item in conversation.get("messages", [])
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]
    model_id = str(conversation.get("model") or VIRTUAL_MODEL)
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid4().hex}"

    async def persist(answer: str) -> None:
        if answer:
            await store.append_message(conversation_id, role="assistant", content=answer)

    agent = _agent_for_model(model_id) if conversation.get("agent_id") else None
    if agent is not None:
        user_entity = await _resolve_identity(request)
        answer_stream = _agent_runtime(request, agent).stream_answer_messages(
            messages,
            request_id=_request_id(request),
            user_entity=user_entity,
            conversation_id=conversation_id,
        )
        if body.stream:
            return StreamingResponse(
                _stream_and_persist(
                    completion_id,
                    created,
                    answer_stream,
                    persist,
                    request,
                    model=agent.display_name,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        answer = "".join([chunk async for chunk in answer_stream])
        await persist(answer)
        return _chat_completion_response(completion_id, created, agent.display_name, answer)

    language_model = _bare_language_model(request, model_id)
    if body.stream:
        return StreamingResponse(
            _stream_and_persist(
                completion_id,
                created,
                language_model.stream_chat(messages),
                persist,
                request,
                model=model_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    collected: list[str] = []
    async for chunk in language_model.stream_chat(messages):
        if chunk:
            collected.append(chunk)
    answer = "".join(collected)
    await persist(answer)
    return _chat_completion_response(completion_id, created, model_id, answer)


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
    conversation_id = await _chat_conversation_id(request, definition.id, user_entity, body.get("conversation_id"))
    try:
        result = await _agent_runtime(request, definition).answer(
            question,
            request_id=_request_id(request),
            user_entity=user_entity,
            **({"conversation_id": conversation_id} if conversation_id else {}),
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
    agents = [
        {
            "id": definition.display_name,
            "object": "model",
            "created": MODEL_CREATED,
            "owned_by": "home-cortex",
            "kind": "agent",
        }
        for definition in list_agents()
    ]
    agent_ids = {item["id"] for item in agents}
    bare = [
        {
            "id": item["id"],
            "object": "model",
            "created": MODEL_CREATED,
            "owned_by": item["owned_by"],
            "kind": "model",
        }
        for item in await _list_bare_models(request)
        if item["id"] not in agent_ids
    ]
    return {"object": "list", "data": [*agents, *bare]}


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
    conversation_id = await _chat_conversation_id(request, definition.id, user_entity, body.conversation_id)
    agent = _agent_runtime(request, definition)
    completion_id = f"chatcmpl-{uuid4().hex}"
    created = int(time.time())
    messages: list[dict[str, str]] = []
    for message in body.messages:
        if message.role == "system":
            continue
        content = message.content
        if message.role == "assistant" and (
            content is None or not content.strip()
        ):
            continue
        if message.role == "user" and (
            content is None or not content.strip()
        ):
            raise APIError(
                422,
                "invalid_request",
                "User messages must contain non-empty text",
            )
        if content is not None:
            messages.append({"role": message.role, "content": content})
    if not messages or not any(message["role"] == "user" for message in messages):
        raise APIError(422, "invalid_request", "Provide at least one user message")
    greeting = None
    if _is_new_conversation(messages) and conversation_id is None:
        greeting = await _greeting_service(request).resolve(
            definition,
            user_entity,
            conversation_language(messages),
        )
    standalone_greeting = None
    if _is_standalone_greeting(messages):
        standalone_greeting = (
            greeting.text
            if greeting is not None
            else _continued_greeting(conversation_language(messages))
        )
    if standalone_greeting is not None and conversation_id is None:
        if body.stream:
            return StreamingResponse(
                _stream_chat_completion(
                    completion_id,
                    created,
                    _single_answer(standalone_greeting),
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
            standalone_greeting,
        )
    agent_messages = messages
    if body.stream:
        answer_stream = agent.stream_answer_messages(
            agent_messages,
            request_id=_request_id(request),
            user_entity=user_entity,
            **({"conversation_id": conversation_id} if conversation_id else {}),
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
            **({"conversation_id": conversation_id} if conversation_id else {}),
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
    normalized = latest_user_message(messages).strip().casefold().rstrip("!！.。?？,， ")
    return normalized in {"hello", "hi", "hey", "你好", "您好", "嗨"}


def _continued_greeting(language: str) -> str:
    return (
        "您好。有什么需要我处理的吗？"
        if language == "zh"
        else "Hello. How may I help?"
    )


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
        # Pad past common proxy buffers so the browser sees the stream start
        # before the steward's first language-model token.
        yield f":{ ' ' * 2048}\n\n" + _sse_data(
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


def _authenticate_bearer(request: Request) -> None:
    """Require the household API key in the Authorization header."""
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


def _authenticate_request(request: Request) -> None:
    """Verify the shared household API key when one is configured.

    V1 uses a single household key. It proves the caller is a trusted
    client (GUI session cookie or Authorization bearer), not which person
    they are.
    """
    expected_key = _request_settings(request).cortex_api_key
    if expected_key is None:
        return
    if valid_gui_session(request.cookies.get(GUI_COOKIE_NAME, ""), expected_key):
        return
    _authenticate_bearer(request)


def _mapped_person_id(request: Request) -> str | None:
    """Map trusted user id/email through CORTEX_IDENTITY_MAP.

    A client-supplied person record ID is not consulted. Anyone holding
    the household API key can present any mapped user header; per-person
    credentials are out of scope for V1. A GUI session stores the same
    map keys, not a person record ID.
    """
    settings = _request_settings(request)
    _authenticate_request(request)
    if not settings.cortex_identity_map:
        return None
    session_user_id = None
    session_email = None
    if settings.cortex_api_key:
        parsed = parse_session(
            request.cookies.get(GUI_COOKIE_NAME, ""),
            settings.cortex_api_key,
        )
        if parsed is not None:
            kind, value = parsed
            if kind == "id":
                session_user_id = value
            elif kind == "email":
                session_email = value
    entity_id = resolve_user_entity_id(
        request.headers,
        settings.cortex_identity_map,
        user_id=session_user_id,
        email=session_email,
    )
    if entity_id is None:
        raise APIError(
            403,
            "identity_not_mapped",
            "The authenticated user is not mapped to a home-graph person",
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


async def _chat_conversation_id(request: Request, agent_id: str, user_entity: Mapping[str, Any] | None, value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 128:
        raise APIError(422, "invalid_request", "Invalid conversation_id")
    _authorize_conversation_access(
        await _conversation_store(request).get(value),
        agent_id=agent_id,
        person_id=str(user_entity["id"]) if user_entity else None,
    )
    return value


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
    if store is None or not hasattr(store, "create"):
        raise RuntimeError("Conversation store is not initialized")
    return store


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


def _transcript_response(
    conversation: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "id": conversation["id"],
        "object": "conversation",
        "model": conversation.get("model"),
        "agent_id": conversation.get("agent_id"),
        "language": conversation.get("language"),
        "greeting": conversation.get("greeting"),
        "title": conversation.get("title") or "",
        "created_at": conversation.get("created_at"),
        "updated_at": conversation.get("updated_at"),
        "messages": [
            {
                "id": item.get("id"),
                "role": item.get("role"),
                "content": item.get("content"),
                "created_at": item.get("created_at"),
            }
            for item in messages
        ],
    }
    return payload


async def _owned_conversation(request: Request, conversation_id: str) -> dict[str, Any]:
    user_entity = await _resolve_identity(request)
    person_id = user_entity.get("id") if user_entity is not None else None
    conversation = await _conversation_store(request).get(conversation_id)
    if conversation is None or conversation.get("person_id") != person_id:
        raise APIError(404, "conversation_not_found", "Conversation was not found")
    return conversation


def _agent_for_model(model_id: str) -> AgentDefinition | None:
    try:
        return get_agent_by_display_name(model_id)
    except UnknownAgentError:
        try:
            return get_agent(model_id)
        except UnknownAgentError:
            return None


def _session_identity(body: SessionRequest, settings: Settings) -> tuple[Any, str]:
    email = body.email.strip() if body.email else None
    user_id = body.user_id.strip() if body.user_id else None
    if email and user_id:
        raise APIError(422, "invalid_request", "Provide email or user_id, not both")
    if not settings.cortex_identity_map:
        if email:
            return "email", email
        if user_id:
            return "id", user_id
        return "none", ""
    entity_id = resolve_user_entity_id(
        {},
        settings.cortex_identity_map,
        user_id=user_id,
        email=email,
    )
    if entity_id is None:
        raise APIError(
            403,
            "identity_not_mapped",
            "The authenticated user is not mapped to a home-graph person",
        )
    if email:
        return "email", email
    if user_id:
        return "id", user_id
    raise APIError(422, "invalid_request", "Provide a mapped email or user_id")


async def _list_bare_models(request: Request) -> list[dict[str, str]]:
    listed = getattr(request.app.state, "bare_models", None)
    if listed is not None:
        return [dict(item) for item in listed]
    settings = _request_settings(request)
    provider = getattr(settings, "llm_provider", "ollama")
    if provider == "openrouter":
        name = getattr(settings, "openrouter_model", None)
        return [{"id": name, "owned_by": "openrouter"}] if name else []
    configured = getattr(settings, "ollama_model", None)
    url = getattr(settings, "ollama_url", None)
    models: list[dict[str, str]] = []
    if url:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{str(url).rstrip('/')}/api/tags")
                response.raise_for_status()
                for item in response.json().get("models") or []:
                    name = item.get("name") if isinstance(item, dict) else None
                    if name:
                        models.append({"id": str(name), "owned_by": "ollama"})
        except Exception:
            models = []
    if configured and not any(item["id"] == configured for item in models):
        models.insert(0, {"id": configured, "owned_by": "ollama"})
    return models


async def _require_bare_model(request: Request, model_id: str) -> None:
    names = {item["id"] for item in await _list_bare_models(request)}
    if model_id not in names:
        raise APIError(404, "model_not_found", f"Model {model_id!r} was not found")


def _bare_language_model(request: Request, model_id: str):
    cached = getattr(request.app.state, "bare_language_models", None)
    if not isinstance(cached, dict):
        cached = {}
        request.app.state.bare_language_models = cached
    if model_id in cached:
        return cached[model_id]
    settings = _request_settings(request)
    language_model = language_model_from_settings(settings, model_id)
    cached[model_id] = language_model
    return language_model


async def _stream_and_persist(
    completion_id: str,
    created: int,
    answer_stream: AsyncIterator[str],
    persist,
    request: Request,
    *,
    model: str,
) -> AsyncIterator[str]:
    collected: list[str] = []

    async def wrapped() -> AsyncIterator[str]:
        try:
            async for chunk in answer_stream:
                if chunk:
                    collected.append(chunk)
                    yield chunk
        finally:
            close = getattr(answer_stream, "aclose", None)
            if close is not None:
                with suppress(asyncio.CancelledError, Exception):
                    await close()
            answer = "".join(collected)
            if answer:
                with suppress(Exception):
                    await persist(answer)

    async for event in _stream_chat_completion(
        completion_id,
        created,
        wrapped(),
        request,
        model=model,
    ):
        yield event


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
