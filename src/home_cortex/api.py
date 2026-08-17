import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import StreamingResponse

from . import __version__
from .agent import AgentLimitError, AgentService
from .config import get_settings
from .db import Database
from .ingestion import ingest_directory
from .ollama import OllamaService
from .retrieval import RetrievalService
from .tools import ToolDispatcher

VIRTUAL_MODEL = "home-cortex"
MODEL_CREATED = int(time.time())


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    database = Database(settings)
    ollama = OllamaService(settings.ollama_url, settings.ollama_model)
    await database.connect()
    app.state.database = database
    app.state.ollama = ollama
    retrieval = RetrievalService(
        database,
        settings.retrieval_limit,
        settings.data_dir,
    )
    app.state.retrieval = retrieval
    app.state.agent = AgentService(ollama, ToolDispatcher(retrieval))
    try:
        yield
    finally:
        await ollama.close()
        await database.close()


app = FastAPI(
    title="Home Cortex API",
    version=__version__,
    description="Graph-grounded RAG service for SurrealDB.",
    lifespan=lifespan,
)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    try:
        surreal_version = await request.app.state.database.version()
        return {
            "status": "ok",
            "surrealdb": surreal_version,
        }
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error))


@app.post("/admin/ingest")
async def ingest(request: Request) -> dict[str, Any]:
    settings = get_settings()
    try:
        result = await ingest_directory(request.app.state.database, settings.data_dir)
        return {"status": "ok", **asdict(result)}
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/v1/retrieve")
async def retrieve(body: dict[str, Any], request: Request) -> dict[str, Any]:
    question = body.get("query") or body.get("question")
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(status_code=422, detail="Provide a non-empty 'query'")
    result = await request.app.state.retrieval.retrieve(question.strip())
    return {
        "query": result.question,
        "nodes": result.nodes,
        "edges": result.edges,
        "context": result.text,
    }


@app.post("/v1/chat")
async def chat(body: dict[str, Any], request: Request) -> dict[str, Any]:
    question = body.get("message") or body.get("question")
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(status_code=422, detail="Provide a non-empty 'message'")
    try:
        result = await request.app.state.agent.answer(question)
    except AgentLimitError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {
        "answer": result.answer,
        "steps": result.steps,
        "tool_calls": result.tool_calls,
    }


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": VIRTUAL_MODEL,
                "object": "model",
                "created": MODEL_CREATED,
                "owned_by": "home-cortex",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
):
    if body.model != VIRTUAL_MODEL:
        raise HTTPException(
            status_code=404,
            detail=f"Model {body.model!r} was not found",
        )
    try:
        result = await request.app.state.agent.answer_messages(
            [message.model_dump() for message in body.messages]
        )
    except AgentLimitError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    completion_id = f"chatcmpl-{uuid4().hex}"
    created = int(time.time())
    if body.stream:
        return StreamingResponse(
            _stream_chat_completion(completion_id, created, result.answer),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": VIRTUAL_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.answer,
                },
                "finish_reason": "stop",
            }
        ],
    }


async def _stream_chat_completion(
    completion_id: str,
    created: int,
    answer: str,
) -> AsyncIterator[str]:
    chunks = [
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": VIRTUAL_MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": VIRTUAL_MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": answer},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": VIRTUAL_MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        },
    ]
    for chunk in chunks:
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
