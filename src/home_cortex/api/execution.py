"""One answer stream used by streamed and collected HTTP responses."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress


PersistAnswer = Callable[[str], Awaitable[None]]


class AnswerExecution:
    """Consume one model/agent stream and persist its assembled answer once."""

    def __init__(
        self,
        source: AsyncIterator[str],
        *,
        persist: PersistAnswer | None = None,
    ) -> None:
        self.source = source
        self.persist = persist
        self._started = False

    async def tokens(
        self, *, suppress_persist_errors: bool = False
    ) -> AsyncIterator[str]:
        if self._started:
            raise RuntimeError("AnswerExecution can only be consumed once")
        self._started = True
        collected: list[str] = []
        try:
            async for chunk in self.source:
                if chunk:
                    collected.append(chunk)
                    yield chunk
        finally:
            close = getattr(self.source, "aclose", None)
            if close is not None:
                with suppress(asyncio.CancelledError, Exception):
                    await close()
            answer = "".join(collected)
            if answer and self.persist is not None:
                if suppress_persist_errors:
                    with suppress(Exception):
                        await self.persist(answer)
                else:
                    await self.persist(answer)

    async def collect(self) -> str:
        return "".join([chunk async for chunk in self.tokens()])


async def single_answer(content: str) -> AsyncIterator[str]:
    yield content


async def prepend_answer(
    prefix: str,
    source: AsyncIterator[str],
) -> AsyncIterator[str]:
    try:
        yield f"{prefix}\n\n"
        async for content in source:
            yield content
    finally:
        close = getattr(source, "aclose", None)
        if close is not None:
            with suppress(asyncio.CancelledError, Exception):
                await close()
