from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from . import __version__
from .agent import AgentLimitError, AgentService
from .config import get_settings
from .db import Database
from .ingestion import ingest_directory
from .ollama import OllamaService
from .retrieval import RetrievalService
from .tools import ToolDispatcher


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
