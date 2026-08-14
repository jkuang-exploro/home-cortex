from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request

from . import __version__
from .config import get_settings
from .db import Database
from .ingestion import ingest_directory
from .ollama import OllamaService
from .retrieval import RetrievalService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    database = Database(settings)
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    await database.connect()
    app.state.database = database
    app.state.retrieval = RetrievalService(
        database,
        settings.retrieval_limit,
        settings.data_dir,
    )
    app.state.ollama = OllamaService(
        http_client,
        settings.ollama_url,
        settings.ollama_model,
    )
    try:
        yield
    finally:
        await http_client.aclose()
        await database.close()


app = FastAPI(
    title="Home Cortex API",
    version=__version__,
    description="Graph-grounded RAG gateway for SurrealDB and Ollama.",
    lifespan=lifespan,
)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    try:
        surreal_version = await request.app.state.database.version()
        models = await request.app.state.ollama.models()
        model_count = len(models.get("data", [])) if isinstance(models, dict) else 0
        return {
            "status": "ok",
            "surrealdb": surreal_version,
            "ollama_models": model_count,
        }
    except HTTPException:
        raise
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


@app.get("/v1/models")
async def models(request: Request) -> Any:
    return await request.app.state.ollama.models()


@app.post("/v1/chat/completions")
async def chat_completions(body: dict[str, Any], request: Request) -> Any:
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=422, detail="'messages' must be a list")

    question = next(
        (
            message.get("content")
            for message in reversed(messages)
            if isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ),
        None,
    )
    if not question:
        raise HTTPException(status_code=422, detail="A text user message is required")

    context = await request.app.state.retrieval.retrieve(question)
    return await request.app.state.ollama.chat_completion(body, context)
