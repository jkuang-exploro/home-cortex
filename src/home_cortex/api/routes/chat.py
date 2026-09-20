"""Agent and OpenAI-compatible chat routes."""
from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from ...runtime.agent import AgentLimitError
from ...agents import UnknownAgentError, get_agent_by_display_name
from ...common.display import conversation_language
from ...common.text import latest_user_message
from ..dependencies import (
    agent_definition,
    agent_runtime,
    chat_conversation_id,
    greeting_service,
    resolve_identity,
)
from ..errors import APIError, request_id
from ..execution import AnswerExecution, prepend_answer, single_answer
from ..schemas import ChatCompletionRequest, DEFAULT_AGENT_ID
from ..sse import chat_completion_response, stream_chat_completion


router = APIRouter()


@router.post("/v1/chat")
async def chat(body: dict[str, Any], request: Request) -> dict[str, Any]:
    return await agent_chat_response(DEFAULT_AGENT_ID, body, request)


@router.post("/agent/{agent_id}/chat")
async def agent_chat(
    agent_id: str,
    body: dict[str, Any],
    request: Request,
) -> dict[str, Any]:
    return await agent_chat_response(agent_id, body, request)


async def agent_chat_response(
    agent_id: str,
    body: dict[str, Any],
    request: Request,
) -> dict[str, Any]:
    definition = agent_definition(agent_id)
    question = body.get("message") or body.get("question")
    if not isinstance(question, str) or not question.strip():
        raise APIError(422, "invalid_request", "Provide a non-empty 'message'")
    user_entity = await resolve_identity(request)
    conversation_id = await chat_conversation_id(
        request,
        definition.id,
        user_entity,
        body.get("conversation_id"),
    )
    try:
        result = await agent_runtime(request, definition).answer(
            question,
            request_id=request_id(request),
            user_entity=user_entity,
            **({"conversation_id": conversation_id} if conversation_id else {}),
        )
    except AgentLimitError as error:
        raise APIError(502, error.stop_reason, str(error)) from error
    return {
        "agent": {"id": definition.id, "display_name": definition.display_name},
        "answer": result.answer,
        "steps": result.steps,
        "tool_calls": result.tool_calls,
        "stop_reason": result.stop_reason,
    }


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    user_entity = await resolve_identity(request)
    try:
        definition = get_agent_by_display_name(body.model)
    except UnknownAgentError:
        raise APIError(
            404,
            "model_not_found",
            f"Model {body.model!r} was not found",
        ) from None
    conversation_id = await chat_conversation_id(
        request,
        definition.id,
        user_entity,
        body.conversation_id,
    )
    agent = agent_runtime(request, definition)
    completion_id = f"chatcmpl-{uuid4().hex}"
    created = int(time.time())
    messages: list[dict[str, str]] = []
    for message in body.messages:
        if message.role == "system":
            continue
        content = message.content
        if message.role == "assistant" and (content is None or not content.strip()):
            continue
        if message.role == "user" and (content is None or not content.strip()):
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
    if is_new_conversation(messages) and conversation_id is None:
        greeting = await greeting_service(request).resolve(
            definition,
            user_entity,
            conversation_language(messages),
        )
    standalone_greeting = None
    if is_standalone_greeting(messages):
        standalone_greeting = (
            greeting.text
            if greeting is not None
            else continued_greeting(conversation_language(messages))
        )
    if standalone_greeting is not None and conversation_id is None:
        execution = AnswerExecution(single_answer(standalone_greeting))
    else:
        source = agent.stream_answer_messages(
            messages,
            request_id=request_id(request),
            user_entity=user_entity,
            **({"conversation_id": conversation_id} if conversation_id else {}),
        )
        if greeting is not None:
            source = prepend_answer(greeting.text, source)
        execution = AnswerExecution(source)

    if body.stream:
        return StreamingResponse(
            stream_chat_completion(
                completion_id,
                created,
                execution.tokens(),
                request,
                model=definition.display_name,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        answer = await execution.collect()
    except AgentLimitError as error:
        raise APIError(502, error.stop_reason, str(error)) from error
    return chat_completion_response(
        completion_id,
        created,
        definition.display_name,
        answer,
    )


def is_new_conversation(messages: list[dict[str, Any]]) -> bool:
    user_messages = sum(message.get("role") == "user" for message in messages)
    has_assistant = any(message.get("role") == "assistant" for message in messages)
    return user_messages == 1 and not has_assistant


def is_standalone_greeting(messages: list[dict[str, Any]]) -> bool:
    normalized = latest_user_message(messages).strip().casefold().rstrip(
        "!！.。?？,， "
    )
    return normalized in {"hello", "hi", "hey", "你好", "您好", "嗨"}


def continued_greeting(language: str) -> str:
    return "您好。有什么需要我处理的吗？" if language == "zh" else "Hello. How may I help?"
