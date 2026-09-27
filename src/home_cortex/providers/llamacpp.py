"""Local llama-server adapter using the shared OpenAI-compatible transport."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from .openai_compatible import OpenAICompatibleService


class LlamaCppService(OpenAICompatibleService):
    """A configured local model; the chat caller cannot select a filesystem path."""

    async def is_resident(self) -> bool:
        """llama-server reports healthy only after its configured model loads."""
        parsed = urlsplit(self.base_url)
        root = parsed.path.removesuffix("/v1").rstrip("/")
        url = urlunsplit((parsed.scheme, parsed.netloc, f"{root}/health", "", ""))
        response = await self.client.get(url)
        return response.status_code == 200

    async def warmup(self) -> None:
        await self._post({
            "model": self.model,
            "messages": [{"role": "user", "content": "OK"}],
            "stream": False,
            "max_tokens": 1,
            "temperature": 0,
        })
