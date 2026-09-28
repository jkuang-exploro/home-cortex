"""FastAPI construction and runtime composition root."""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..runtime.agent import AgentService
from ..runtime.model_warmup import ModelWarmup
from ..semantic.prompt import planner_chat_messages
from ..semantic.schema import SemanticSchemaRegistry
from ..semantic.unified_planner import unified_chat_messages, unified_output_schema
from ..mutation.ir import read_plan_schema
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
from ..spatial.presence import (
    DEFAULT_OBSERVER_STALE_AFTER_S,
    EmbodimentPresence,
    embodiment_ids_from_node_file,
    record_ids_from_node_file,
)
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
from .routes import chat, conversations, models, sessions, system, telemetry
from .schemas import DEFAULT_AGENT_ID, REQUEST_ID_HEADER


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.cortex_api_key is None:
        raise RuntimeError("CORTEX_API_KEY is required to serve the API")
    app.state.settings = settings
    definitions = list_agents()
    providers = [
        model_provider_from_settings(settings, definition.model.name)
        for definition in definitions
    ]
    local_providers = [
        provider for provider in providers
        if callable(getattr(provider, "is_resident", None))
        and callable(getattr(provider, "warmup", None))
    ]
    warmup = ModelWarmup(local_providers if settings.cortex_model_warmup else ())
    app.state.model_warmup = warmup
    database = None
    connected = False
    try:
        edge_registry = EdgeSchemaRegistry.from_directory(settings.edge_schema_dir)
        schema_catalog = RuntimeSchemaCatalog.from_data_dir(
            settings.data_dir, edge_registry,
        )
        app.state.edge_registry = edge_registry
        app.state.embodiment_presence = EmbodimentPresence(
            embodiment_ids=embodiment_ids_from_node_file(
                settings.data_dir / "nodes" / "embodiment.json"
            ),
            space_ids=record_ids_from_node_file(
                settings.data_dir / "nodes" / "space.json", "space"
            ),
            stale_after_s=DEFAULT_OBSERVER_STALE_AFTER_S,
        )
        if settings.cortex_model_warmup:
            semantic_schema = SemanticSchemaRegistry(schema_catalog)
            synthetic_turn = [{"role": "user", "content": "How many members live in this household?"}]
            household_now = datetime.now(ZoneInfo(settings.calendar_timezone)).isoformat()
            for definition, provider in zip(definitions, providers):
                configure = getattr(provider, "configure_planner_warmup", None)
                if not callable(configure):
                    continue
                if "write_item" in definition.allowed_tools:
                    messages = unified_chat_messages(
                        synthetic_turn, semantic_schema, household_now=household_now,
                    )
                    output_schema = unified_output_schema(semantic_schema)
                else:
                    messages = planner_chat_messages(
                        synthetic_turn,
                        semantic_schema.planner_capability_payload(),
                        household_now=household_now,
                    )
                    output_schema = read_plan_schema(semantic_schema.planner_output_schema())
                configure(messages, output_schema)
        warmup.start()
        database = Database(settings)
        await database.connect()
        connected = True
        app.state.database = database
        retrieval = RetrievalService(
            database,
            settings.retrieval_limit,
            settings.data_dir,
            edge_registry,
        )
        app.state.retrieval = retrieval
        writing = ItemWritingService(database, schema_catalog, edge_registry)
        app.state.greetings = GreetingService(retrieval)
        app.state.conversations = SurrealConversationStore(database)
        calendar = calendar_service_from_settings(settings)
        runtimes: dict[str, AgentService] = {}
        for definition, provider in zip(definitions, providers):
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
        yield
    finally:
        await warmup.close()
        bare = getattr(app.state, "bare_language_models", {})
        for provider in [*providers, *bare.values()]:
            await provider.close()
        if connected and database is not None:
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
        telemetry.router,
    ):
        application.include_router(router)
    return application


app = create_app()
