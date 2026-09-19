"""Conversation lifecycle and persisted message execution."""
from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from ..dependencies import (
    agent_definition,
    agent_for_model,
    agent_runtime,
    authorize_conversation_access,
    conversation_store,
    greeting_service,
    owned_conversation,
    resolve_identity,
)
from ..errors import APIError, request_id
from ..execution import AnswerExecution
from ..providers import bare_model_provider, require_bare_model
from ..schemas import (
    ConversationCreateRequest,
    ConversationMessageRequest,
    VIRTUAL_MODEL,
)
from ..sse import chat_completion_response, stream_chat_completion


router = APIRouter()


@router.post("/agent/{agent_id}/conversations", status_code=201)
async def create_agent_conversation(
    agent_id: str,
    body: ConversationCreateRequest,
    request: Request,
) -> dict[str, Any]:
    definition = agent_definition(agent_id)
    user_entity = await resolve_identity(request)
    greeting = await greeting_service(request).resolve(
        definition,
        user_entity,
        body.language,
    )
    person_id = user_entity.get("id") if user_entity is not None else None
    conversation = await conversation_store(request).create(
        agent_id=definition.id,
        model=definition.display_name,
        person_id=person_id if isinstance(person_id, str) else None,
        language=greeting.language,
        greeting=greeting.text,
    )
    return conversation_response(conversation, definition)


@router.get("/agent/{agent_id}/conversations/{conversation_id}")
async def get_agent_conversation(
    agent_id: str,
    conversation_id: str,
    request: Request,
) -> dict[str, Any]:
    definition = agent_definition(agent_id)
    user_entity = await resolve_identity(request)
    person_id = user_entity.get("id") if user_entity is not None else None
    conversation = authorize_conversation_access(
        await conversation_store(request).get(conversation_id),
        agent_id=definition.id,
        person_id=person_id if isinstance(person_id, str) else None,
    )
    return conversation_response(conversation, definition)


@router.get("/conversations")
async def list_conversations(request: Request) -> dict[str, Any]:
    user_entity = await resolve_identity(request)
    person_id = user_entity.get("id") if user_entity is not None else None
    items = await conversation_store(request).list_for_person(
        person_id if isinstance(person_id, str) else None
    )
    return {"object": "list", "data": items}


@router.post("/conversations", status_code=201)
async def create_conversation(
    body: ConversationCreateRequest,
    request: Request,
) -> dict[str, Any]:
    user_entity = await resolve_identity(request)
    person_id = user_entity.get("id") if user_entity is not None else None
    model_id = (body.model or VIRTUAL_MODEL).strip()
    agent = agent_for_model(model_id)
    store = conversation_store(request)
    if agent is not None:
        greeting = await greeting_service(request).resolve(
            agent,
            user_entity,
            body.language,
        )
        conversation = await store.create(
            agent_id=agent.id,
            model=agent.display_name,
            person_id=person_id if isinstance(person_id, str) else None,
            language=greeting.language,
            greeting=greeting.text,
        )
        return transcript_response(
            conversation, messages=conversation.get("messages", [])
        )
    await require_bare_model(request, model_id)
    conversation = await store.create(
        agent_id=None,
        model=model_id,
        person_id=person_id if isinstance(person_id, str) else None,
        language=body.language,
        greeting=None,
    )
    return transcript_response(conversation, messages=[])


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    request: Request,
) -> dict[str, Any]:
    conversation = await owned_conversation(request, conversation_id)
    return transcript_response(conversation, messages=conversation.get("messages", []))


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    request: Request,
) -> dict[str, Any]:
    await owned_conversation(request, conversation_id)
    await conversation_store(request).delete(conversation_id)
    return {"status": "ok", "id": conversation_id}


@router.post("/conversations/{conversation_id}/messages")
async def post_conversation_message(
    conversation_id: str,
    body: ConversationMessageRequest,
    request: Request,
):
    conversation = await owned_conversation(request, conversation_id)
    store = conversation_store(request)
    content = body.content.strip()
    if not content:
        raise APIError(422, "invalid_request", "Provide a non-empty 'content'")
    first_user = not any(
        item.get("role") == "user" for item in conversation.get("messages", [])
    )
    user_message = await store.append_message(
        conversation_id,
        role="user",
        content=content,
        first_user=first_user,
    )
    if user_message is None:
        raise APIError(404, "conversation_not_found", "Conversation was not found")
    conversation_messages = conversation.setdefault("messages", [])
    conversation_messages.append(user_message)
    messages = [
        {"role": item["role"], "content": item["content"]}
        for item in conversation_messages
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]
    model_id = str(conversation.get("model") or VIRTUAL_MODEL)
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid4().hex}"

    async def persist(answer: str) -> None:
        await store.append_message(
            conversation_id,
            role="assistant",
            content=answer,
        )

    agent = agent_for_model(model_id) if conversation.get("agent_id") else None
    if agent is not None:
        user_entity = await resolve_identity(request)
        source = agent_runtime(request, agent).stream_answer_messages(
            messages,
            request_id=request_id(request),
            user_entity=user_entity,
            conversation_id=conversation_id,
        )
        response_model = agent.display_name
    else:
        source = bare_model_provider(request, model_id).stream_chat(messages)
        response_model = model_id

    execution = AnswerExecution(source, persist=persist)
    if body.stream:
        return StreamingResponse(
            stream_chat_completion(
                completion_id,
                created,
                execution.tokens(suppress_persist_errors=True),
                request,
                model=response_model,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    answer = await execution.collect()
    return chat_completion_response(
        completion_id,
        created,
        response_model,
        answer,
    )


def conversation_response(conversation: dict[str, Any], definition: Any) -> dict[str, Any]:
    return {
        "id": conversation["id"],
        "object": "agent.conversation",
        "agent": {"id": definition.id, "display_name": definition.display_name},
        "language": conversation["language"],
        "greeting": conversation["greeting"],
    }


def transcript_response(
    conversation: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
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
