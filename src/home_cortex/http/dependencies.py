"""Authentication, identity, and initialized-service lookup."""
from __future__ import annotations

import logging
import secrets
from collections.abc import Mapping
from typing import Any

from fastapi import Request

from ..agent_service import AgentService
from ..agents import (
    AgentDefinition,
    UnknownAgentError,
    get_agent,
    get_agent_by_display_name,
)
from ..config import Settings, get_settings
from ..conversations import ConversationStore
from ..greetings import GreetingService
from ..gui_session import (
    COOKIE_NAME as GUI_COOKIE_NAME,
    parse_session,
    valid_session as valid_gui_session,
)
from ..identity import (
    OPENWEBUI_USER_EMAIL_HEADER,
    OPENWEBUI_USER_ID_HEADER,
    resolve_user_entity_id,
)
from .errors import APIError, request_id
from .schemas import DEFAULT_AGENT_ID


logger = logging.getLogger("uvicorn.error.home_cortex.api")


def request_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


def authenticate_bearer(request: Request) -> None:
    expected_key = request_settings(request).cortex_api_key
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


def authenticate_request(request: Request) -> None:
    expected_key = request_settings(request).cortex_api_key
    if expected_key is None:
        return
    if valid_gui_session(request.cookies.get(GUI_COOKIE_NAME, ""), expected_key):
        return
    authenticate_bearer(request)


def mapped_person_id(request: Request) -> str | None:
    settings = request_settings(request)
    authenticate_request(request)
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


async def resolve_identity(request: Request) -> dict[str, Any] | None:
    entity_id = mapped_person_id(request)
    if entity_id is None:
        return None
    try:
        entity = await request.app.state.retrieval.get_entity(entity_id)
    except ValueError:
        entity = None
    if entity is None or entity.get("id") != entity_id:
        logger.info(
            "identity_resolution request_id=%s success=false reason=record_not_found",
            request_id(request),
        )
        raise APIError(
            403,
            "identity_record_not_found",
            "The mapped home-graph person record was not found",
        )
    logger.info(
        "identity_resolution request_id=%s success=true has_name=%s "
        "has_address_as=%s",
        request_id(request),
        str("name" in entity).lower(),
        str("address_as" in entity).lower(),
    )
    return {
        key: entity[key]
        for key in ("id", "name", "address_as")
        if key in entity
    }


def authorize_conversation_access(
    conversation: dict[str, Any] | None,
    *,
    agent_id: str,
    person_id: str | None,
) -> dict[str, Any]:
    if (
        conversation is None
        or conversation["agent_id"] != agent_id
        or conversation["person_id"] != person_id
    ):
        raise APIError(404, "conversation_not_found", "Conversation was not found")
    return conversation


async def chat_conversation_id(
    request: Request,
    agent_id: str,
    user_entity: Mapping[str, Any] | None,
    value: Any,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 128:
        raise APIError(422, "invalid_request", "Invalid conversation_id")
    authorize_conversation_access(
        await conversation_store(request).get(value),
        agent_id=agent_id,
        person_id=str(user_entity["id"]) if user_entity else None,
    )
    return value


def agent_definition(agent_id: str) -> AgentDefinition:
    try:
        return get_agent(agent_id)
    except UnknownAgentError as error:
        raise APIError(
            404,
            "agent_not_found",
            f"Agent {agent_id!r} was not found",
        ) from error


def agent_for_model(model_id: str) -> AgentDefinition | None:
    try:
        return get_agent_by_display_name(model_id)
    except UnknownAgentError:
        try:
            return get_agent(model_id)
        except UnknownAgentError:
            return None


def agent_runtime(request: Request, definition: AgentDefinition) -> AgentService:
    runtimes = getattr(request.app.state, "agents", None)
    if isinstance(runtimes, dict) and definition.id in runtimes:
        return runtimes[definition.id]
    if definition.id == DEFAULT_AGENT_ID:
        runtime = getattr(request.app.state, "agent", None)
        if runtime is not None:
            return runtime
    raise RuntimeError(f"Agent runtime {definition.id!r} is not initialized")


def greeting_service(request: Request) -> GreetingService:
    service = getattr(request.app.state, "greetings", None)
    if isinstance(service, GreetingService):
        return service
    raise RuntimeError("Greeting service is not initialized")


def conversation_store(request: Request) -> ConversationStore:
    store = getattr(request.app.state, "conversations", None)
    if store is None or not hasattr(store, "create"):
        raise RuntimeError("Conversation store is not initialized")
    return store


async def owned_conversation(
    request: Request,
    conversation_id: str,
) -> dict[str, Any]:
    user_entity = await resolve_identity(request)
    person_id = user_entity.get("id") if user_entity is not None else None
    conversation = await conversation_store(request).get(conversation_id)
    if conversation is None or conversation.get("person_id") != person_id:
        raise APIError(404, "conversation_not_found", "Conversation was not found")
    return conversation


def session_identity(body: Any, settings: Settings) -> tuple[Any, str]:
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

