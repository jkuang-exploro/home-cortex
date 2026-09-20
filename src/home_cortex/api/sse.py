"""OpenAI-compatible JSON and SSE response rendering."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

from fastapi import Request

from ..runtime.agent import AgentLimitError, AgentStreamingError
from .errors import request_id
from .schemas import VIRTUAL_MODEL


logger = logging.getLogger("uvicorn.error.home_cortex.api")


def chat_completion_response(
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
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


async def stream_chat_completion(
    completion_id: str,
    created: int,
    answer_stream: AsyncIterator[str],
    request: Request | None = None,
    *,
    model: str = VIRTUAL_MODEL,
) -> AsyncIterator[str]:
    try:
        yield f":{' ' * 2048}\n\n" + sse_data(
            chat_completion_chunk(
                completion_id,
                created,
                delta={"role": "assistant"},
                model=model,
            )
        )
        async for content in answer_stream:
            if content:
                yield sse_data(
                    chat_completion_chunk(
                        completion_id,
                        created,
                        delta={"content": content},
                        model=model,
                    )
                )
        yield sse_data(
            chat_completion_chunk(
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
            log_client_disconnect(request, phase="ollama_stream")
        raise
    except (AgentLimitError, AgentStreamingError) as error:
        yield sse_error(error.stop_reason, str(error), request)
        yield "data: [DONE]\n\n"
    except Exception as error:
        logger.error(
            "stream_error request_id=%s exception_type=%s",
            request_id(request),
            type(error).__name__,
        )
        yield sse_error(
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


def chat_completion_chunk(
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


def sse_data(value: dict[str, Any]) -> str:
    return f"data: {json.dumps(value, ensure_ascii=False)}\n\n"


def sse_error(code: str, message: str, request: Request | None) -> str:
    return sse_data(
        {
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id(request),
            }
        }
    )


def log_client_disconnect(request: Request, *, phase: str) -> None:
    logger.info(
        "stream_cancelled request_id=%s phase=%s",
        request_id(request),
        phase,
    )
