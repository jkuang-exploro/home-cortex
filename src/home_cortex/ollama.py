from collections.abc import Mapping, Sequence
from typing import Any

from ollama import AsyncClient, ChatResponse

from .tools import TOOLS


class OllamaService:
    """Make individual non-streaming Ollama calls for the future agent loop."""

    def __init__(
        self,
        base_url: str,
        model: str,
        client: AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._owns_client = client is None
        self.client = client or AsyncClient(host=self.base_url)

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> ChatResponse:
        """Send one ordinary chat request without exposing tools."""
        return await self.client.chat(
            model=self.model,
            messages=messages,
            stream=False,
        )

    async def chat_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> ChatResponse:
        """Send one request that lets the model choose a read-only Cortex tool."""
        return await self.client.chat(
            model=self.model,
            messages=messages,
            tools=TOOLS,
            stream=False,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self.client.close()

