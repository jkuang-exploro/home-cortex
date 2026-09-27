"""Measure streamed first-answer latency on an isolated synthetic HTTP API.

Runs the real API routes and model provider, but uses an in-memory conversation
store and the fixed invented graph. It never opens the household database or
dispatches writes. Run from an isolated package on the GPU host.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import httpx
import uvicorn
import yaml

from benchmarks.harness.environment import (
    git_metadata, hash_tree, instruction_prompt_fingerprint, ollama_metadata,
    repo_root, sha256_file,
)
from benchmarks.harness.stats import percentile
from home_cortex.agents import get_agent
from home_cortex.api.app import create_app
from home_cortex.conversation.store import ConversationStore
from home_cortex.persistence.edge_schema import EdgeSchemaRegistry
from home_cortex.persistence.schema_catalog import RuntimeSchemaCatalog
from home_cortex.providers.llamacpp import LlamaCppService
from home_cortex.providers.ollama import OllamaService
from home_cortex.runtime.agent import AgentService
from home_cortex.runtime.model_warmup import ModelWarmup
from scripts.benchmarks.json_graph import JsonGraphDispatcher


KEY = "synthetic-benchmark-key"
USER = "synthetic-benchmark-user"


class _SyntheticRetrieval:
    def __init__(self, graph: JsonGraphDispatcher):
        self.graph = graph

    async def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        return self.graph.entities.get(entity_id)


class _TraceCollector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.profiles: dict[str, dict[str, Any]] = {}

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith("request_profile "):
            payload = json.loads(record.getMessage().removeprefix("request_profile "))
            if payload.get("request_id"):
                self.profiles[payload["request_id"]] = payload


def _app(args: argparse.Namespace, dataset: dict[str, Any], provider: Any):
    os.environ["CORTEX_PROFILE_REQUESTS"] = "1"
    app = create_app()
    root = repo_root()
    fixture = root / "benchmarks" / dataset["fixture"]
    edges = EdgeSchemaRegistry.from_directory(root / "schemas" / "edge")
    graph = JsonGraphDispatcher(fixture, edges)
    catalog = RuntimeSchemaCatalog.from_data_dir(fixture, edges)
    steward = get_agent("steward")
    app.state.agents = {
        steward.id: AgentService(
            provider, graph,
            system_prompt=steward.prompt,
            tools=steward.tool_definitions,
            schema_catalog=catalog,
            localized_identity=steward.settings.get("localized_identity"),
            assistant_id=steward.id,
            assistant_display_name=steward.display_name,
            home_entity_id=dataset["household_id"],
        )
    }
    app.state.settings = SimpleNamespace(
        cortex_api_key=KEY,
        cortex_identity_map={f"id:{USER}": dataset["speaker_id"]},
        llm_provider=args.runtime,
        ollama_model=args.model,
        ollama_url=args.base_url,
        local_llm_model=args.model,
        local_llm_base_url=args.base_url,
    )
    app.state.retrieval = _SyntheticRetrieval(graph)
    app.state.conversations = ConversationStore()
    app.state.bare_models = [{"id": args.model, "owned_by": args.runtime}]
    app.state.bare_language_models = {args.model: provider}
    return app


async def _measure(
    client: httpx.AsyncClient, conversation_id: str, utterance: str,
) -> dict[str, Any]:
    started = perf_counter()
    clock = lambda: round((perf_counter() - started) * 1000, 3)
    row: dict[str, Any] = {"headers_ms": None, "first_byte_ms": None,
                           "first_content_ms": None, "complete_ms": None}
    async with client.stream(
        "POST", f"/conversations/{conversation_id}/messages",
        json={"content": utterance, "stream": True},
    ) as response:
        row["headers_ms"] = clock()
        row["status_code"] = response.status_code
        row["request_id"] = response.headers.get("X-Request-ID")
        buffer = ""
        for_bytes = response.aiter_bytes()
        async for chunk in for_bytes:
            if row["first_byte_ms"] is None and chunk:
                row["first_byte_ms"] = clock()
            buffer += chunk.decode("utf-8")
            while "\n\n" in buffer:
                event, buffer = buffer.split("\n\n", 1)
                data = next((line[6:] for line in event.splitlines()
                             if line.startswith("data: ")), None)
                if not data or data == "[DONE]":
                    continue
                payload = json.loads(data)
                if payload.get("error"):
                    row["stream_error"] = payload["error"].get("code")
                choices = payload.get("choices") or []
                if (row["first_content_ms"] is None and choices
                        and (choices[0].get("delta") or {}).get("content")):
                    row["first_content_ms"] = clock()
        row["complete_ms"] = clock()
    return row


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    dataset_path = root / "benchmarks" / "first_answer_latency.yaml"
    dataset = yaml.safe_load(dataset_path.read_text())
    if dataset.get("version") != 1:
        raise ValueError("Unknown first-answer benchmark dataset version")
    provider = (OllamaService(args.base_url, args.model) if args.runtime == "ollama"
                else LlamaCppService(args.base_url, args.model))
    app = _app(args, dataset, provider)
    collector = _TraceCollector()
    warmup = ModelWarmup([provider] if args.warmup else [])
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, lifespan="off",
        access_log=False, log_level="error",
    ))
    logger = logging.getLogger("uvicorn.error.home_cortex.common.tracing")
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.addHandler(collector)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    task = asyncio.create_task(server.serve(sockets=[sock]))
    rows: list[dict[str, Any]] = []
    try:
        async with asyncio.timeout(10):
            while not server.started:
                await asyncio.sleep(0.01)
        if args.warmup:
            warmup.start()
            async with asyncio.timeout(120):
                while warmup.snapshot()["status"] != "warm":
                    await asyncio.sleep(0.05)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}",
            timeout=httpx.Timeout(120),
            headers={"Authorization": f"Bearer {KEY}",
                     "X-OpenWebUI-User-Id": USER},
        ) as client:
            for iteration in range(args.repetitions + 1):
                for case in dataset["cases"]:
                    conversation = await app.state.conversations.create(
                        agent_id=None if case.get("bare") else "steward",
                        model=args.model if case.get("bare") else get_agent("steward").display_name,
                        person_id=dataset["speaker_id"], language="en",
                    )
                    for turn, utterance in enumerate(case["turns"]):
                        row = await _measure(client, conversation["id"], utterance)
                        if iteration == 0:
                            continue
                        row.update(case_id=case["id"], cohort=case["cohort"],
                                   turn=turn, iteration=iteration)
                        rows.append(row)
            if args.concurrency > 1:
                for iteration in range(1, args.repetitions + 1):
                    conversations = [
                        await app.state.conversations.create(
                            agent_id="steward", model=get_agent("steward").display_name,
                            person_id=dataset["speaker_id"], language="en",
                        )
                        for _ in range(args.concurrency)
                    ]
                    concurrent = await asyncio.gather(*(
                        _measure(client, item["id"], "Who am I?")
                        for item in conversations
                    ))
                    for row in concurrent:
                        row.update(case_id="fact_self_en_concurrent",
                                   cohort="fact_concurrent", turn=0,
                                   iteration=iteration)
                        rows.append(row)
            # The ASGI trace closes after the final body send, which can lag the
            # client-side stream close by a scheduling turn.
            await asyncio.sleep(0.05)
    finally:
        server.should_exit = True
        await task
        await warmup.close()
        await provider.close()
        logger.removeHandler(collector)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
    for row in rows:
        profile = collector.profiles.get(row["request_id"])
        row["stages"] = profile["events"] if profile else []
    cohorts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cohorts[row["cohort"]].append(row)
    metrics = {}
    for name, items in cohorts.items():
        first = [r["first_content_ms"] for r in items if r["first_content_ms"] is not None]
        first_byte = [r["first_byte_ms"] for r in items if r["first_byte_ms"] is not None]
        complete = [r["complete_ms"] for r in items]
        metrics[name] = {
            "n": len(items), "answer_n": len(first),
            "http_error_n": sum(r["status_code"] >= 400 for r in items),
            "stream_error_n": sum(bool(r.get("stream_error")) for r in items),
            "first_byte_p50_ms": percentile(first_byte, 0.50) if first_byte else None,
            "first_byte_p95_ms": percentile(first_byte, 0.95) if first_byte else None,
            "first_content_p50_ms": percentile(first, 0.50) if first else None,
            "first_content_p95_ms": percentile(first, 0.95) if first else None,
            "complete_p50_ms": percentile(complete, 0.50),
            "complete_p95_ms": percentile(complete, 0.95),
        }
    return {
        "scope": "loopback streamed HTTP, synthetic graph; excludes proxy/browser paint",
        "runtime": args.runtime, "model": args.model,
        "warmup": warmup.snapshot(), "metrics": metrics, "rows": rows,
        "provenance": {
            "git": git_metadata(root),
            "package_sha256": hash_tree(root / "src" / "home_cortex"),
            "scripts_sha256": hash_tree(root / "scripts"),
            "schema_sha256": hash_tree(root / "schemas"),
            "fixture_sha256": hash_tree(root / "benchmarks" / dataset["fixture"]),
            "dataset_sha256": sha256_file(dataset_path),
            "prompt_sha256": instruction_prompt_fingerprint(),
            "model_runtime": (ollama_metadata(args.base_url, args.model)
                              if args.runtime == "ollama" else None),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("ollama", "llamacpp"), default="ollama")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-warmup", action="store_false", dest="warmup")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    report = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
