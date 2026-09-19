"""Compatibility entry point for the split HTTP package.

Deployment may continue using ``home_cortex.api:app``. New HTTP code belongs in
``home_cortex.http`` route, transport, or composition modules.
"""

from .conversations import ConversationStore
from .http.app import app, create_app, lifespan
from .http.errors import APIError
from .http.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    ConversationCreateRequest,
    ConversationMessageRequest,
    DEFAULT_AGENT_ID,
    ExportRequest,
    MODEL_CREATED,
    REQUEST_ID_HEADER,
    SessionRequest,
    VIRTUAL_MODEL,
)
from .http.sse import stream_chat_completion as _stream_chat_completion


__all__ = (
    "APIError",
    "ChatCompletionRequest",
    "ChatMessage",
    "ConversationCreateRequest",
    "ConversationMessageRequest",
    "ConversationStore",
    "DEFAULT_AGENT_ID",
    "ExportRequest",
    "MODEL_CREATED",
    "REQUEST_ID_HEADER",
    "SessionRequest",
    "VIRTUAL_MODEL",
    "_stream_chat_completion",
    "app",
    "create_app",
    "lifespan",
)
