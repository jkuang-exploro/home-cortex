import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .retrieval import RetrievedContext

SYSTEM_INSTRUCTION = """You are Home Cortex, a local assistant grounded in a home graph.
Use the supplied graph context as the source of truth for home-specific facts.
Do not invent missing people, objects, locations, or relationships.
If the graph does not contain the answer, say that the information is not recorded.

HOME GRAPH CONTEXT:
{context}
"""


class OllamaService:
    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model_override: str | None = None,
    ) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.model_override = model_override

    async def models(self) -> Any:
        try:
            response = await self.client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise HTTPException(status_code=502, detail=f"Ollama is unavailable: {error}")

    async def chat_completion(
        self,
        payload: dict[str, Any],
        context: RetrievedContext,
    ) -> JSONResponse | StreamingResponse:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=422, detail="'messages' must be a non-empty list")

        upstream_payload = dict(payload)
        upstream_payload["messages"] = [
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION.format(context=context.text),
            },
            *messages,
        ]
        if self.model_override:
            upstream_payload["model"] = self.model_override
        elif not isinstance(upstream_payload.get("model"), str):
            raise HTTPException(status_code=422, detail="'model' must be provided")

        if bool(upstream_payload.get("stream", False)):
            return await self._stream(upstream_payload)
        return await self._complete(upstream_payload)

    async def _complete(self, payload: dict[str, Any]) -> JSONResponse:
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            return JSONResponse(content=response.json(), status_code=response.status_code)
        except httpx.HTTPStatusError as error:
            detail = _upstream_detail(error.response)
            raise HTTPException(status_code=error.response.status_code, detail=detail)
        except (httpx.HTTPError, ValueError) as error:
            raise HTTPException(status_code=502, detail=f"Ollama is unavailable: {error}")

    async def _stream(self, payload: dict[str, Any]) -> StreamingResponse:
        request = self.client.build_request(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        )
        try:
            response = await self.client.send(request, stream=True)
        except httpx.HTTPError as error:
            raise HTTPException(status_code=502, detail=f"Ollama is unavailable: {error}")

        if response.is_error:
            await response.aread()
            detail = _upstream_detail(response)
            await response.aclose()
            raise HTTPException(status_code=response.status_code, detail=detail)

        async def chunks() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()

        return StreamingResponse(chunks(), media_type="text/event-stream")


def _upstream_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict):
            error = body.get("error") or body.get("detail")
            if isinstance(error, dict):
                return str(error.get("message", error))
            if error:
                return str(error)
        return json.dumps(body)
    except ValueError:
        return response.text or f"Ollama returned HTTP {response.status_code}"

