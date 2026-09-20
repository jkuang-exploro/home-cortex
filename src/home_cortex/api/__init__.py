"""Public HTTP application surface.

Deployment uses ``home_cortex.api:app``. Route and transport implementation lives
in this package; core services must not depend on it.
"""

from ..conversation.store import ConversationStore
from .app import app, create_app, lifespan
from .errors import APIError
from .schemas import (
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
from .sse import stream_chat_completion as _stream_chat_completion


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
