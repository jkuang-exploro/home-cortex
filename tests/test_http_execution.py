import pytest

from home_cortex.api.execution import AnswerExecution


async def _source(*chunks: str):
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_collect_consumes_one_stream_and_persists_once() -> None:
    persisted: list[str] = []

    async def persist(answer: str) -> None:
        persisted.append(answer)

    execution = AnswerExecution(_source("one", "", " two"), persist=persist)
    assert await execution.collect() == "one two"
    assert persisted == ["one two"]
    with pytest.raises(RuntimeError, match="only be consumed once"):
        await execution.collect()


@pytest.mark.asyncio
async def test_collect_preserves_nonstream_persistence_errors() -> None:
    async def persist(_answer: str) -> None:
        raise RuntimeError("store unavailable")

    execution = AnswerExecution(_source("answer"), persist=persist)
    with pytest.raises(RuntimeError, match="store unavailable"):
        await execution.collect()


@pytest.mark.asyncio
async def test_stream_can_finish_when_post_response_persistence_fails() -> None:
    async def persist(_answer: str) -> None:
        raise RuntimeError("store unavailable")

    execution = AnswerExecution(_source("answer"), persist=persist)
    assert "".join([
        token async for token in execution.tokens(suppress_persist_errors=True)
    ]) == "answer"
