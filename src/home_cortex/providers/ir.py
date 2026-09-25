"""Provider-neutral chat response objects and transport errors."""

from typing import Any

from pydantic import BaseModel


class ChatFunction(BaseModel):
    name: str
    arguments: dict[str, Any]


class ChatToolCall(BaseModel):
    function: ChatFunction


class ChatMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ChatToolCall] | None = None


class ChatResponse(BaseModel):
    model: str
    message: ChatMessage


class ModelProviderError(RuntimeError):
    """A model transport failed; safe to report without response body details."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code
