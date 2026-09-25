"""FastAPI construction and runtime composition root."""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..runtime.agent import AgentService
from ..agents import list_agents
from ..capabilities.calendar import calendar_service_from_settings
from ..config import get_settings
from ..conversation.store import SurrealConversationStore
from ..persistence.db import Database
from ..persistence.edge_schema import EdgeSchemaRegistry
from ..conversation.greetings import GreetingService
from ..providers.base import model_provider_from_settings
from ..providers.ir import ModelProviderError
from ..common.tracing import RequestTraceMiddleware
from ..persistence.retrieval import RetrievalService
from ..persistence.schema_catalog import RuntimeSchemaCatalog
from ..capabilities.dispatcher import ToolDispatcher
from ..mutation.writing import ItemWritingService
from .errors import (
    http_error_handler,
    model_provider_error_handler,
    logger,
    unexpected_error_handler,
    validation_error_handler,
)
from .routes import chat, conversations, models, sessions, system
from .schemas import DEFAULT_AGENT_ID, REQUEST_ID_HEADER


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
    writing = ItemWritingService(database, schema_catalog, edge_registry)
    app.state.greetings = GreetingService(retrieval)
    app.state.conversations = SurrealConversationStore(database)
    calendar = calendar_service_from_settings(settings)
    runtimes: dict[str, AgentService] = {}
    providers = []
    for definition in list_agents():
        provider = model_provider_from_settings(settings, definition.model.name)
        providers.append(provider)
        runtimes[definition.id] = AgentService(
            provider,
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
    try:
        yield
    finally:
        bare = getattr(app.state, "bare_language_models", {})
        for provider in [*providers, *bare.values()]:
            await provider.close()
        await database.close()


def create_app() -> FastAPI:
    application = FastAPI(
        title="Home Cortex API",
        version=__version__,
        description="Graph-grounded RAG service for SurrealDB.",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_observability(request, call_next):
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

    application.add_middleware(
        RequestTraceMiddleware,
        enabled=os.environ.get("CORTEX_PROFILE_REQUESTS") == "1",
    )
    application.add_exception_handler(StarletteHTTPException, http_error_handler)
    application.add_exception_handler(RequestValidationError, validation_error_handler)
    application.add_exception_handler(ModelProviderError, model_provider_error_handler)
    application.add_exception_handler(Exception, unexpected_error_handler)
    for router in (
        sessions.router,
        system.router,
        conversations.router,
        chat.router,
        models.router,
    ):
        application.include_router(router)
    return application


app = create_app()
