"""Transport-only request models and response constants."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..agents import get_agent


DEFAULT_AGENT_ID = "steward"
VIRTUAL_MODEL = get_agent(DEFAULT_AGENT_ID).display_name
MODEL_CREATED = int(time.time())
REQUEST_ID_HEADER = "X-Request-ID"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant"]
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


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_dir: Path

