#!/usr/bin/env python3
"""Isolated Ollama warm-request load_duration probe.

Does not unload models, change Ollama settings, or send a mismatched num_ctx.
Inference calls keep the requested model at the resident context and keep_alive 24h.

Direct HTTP experiments use only the standard library. Cortex comparison
requires PYTHONPATH to an isolated package copy.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

MODEL = "qwen3.5:4b"
NUM_CTX = 8192
KEEP_ALIVE = "24h"
TINY_USER = "Reply with the single word: ok"


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256_text(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode()
    return hashlib.sha256(data).hexdigest()


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize(values: list[float] | None) -> dict[str, Any]:
    numbers = [float(v) for v in values or [] if v is not None]
    if not numbers:
        return {"n": 0}
    return {
        "n": len(numbers),
        "mean": statistics.mean(numbers),
        "stdev": statistics.pstdev(numbers) if len(numbers) > 1 else 0.0,
        "min": min(numbers),
        "p50": percentile(numbers, 50),
        "p95": percentile(numbers, 95),
        "max": max(numbers),
    }


def ns_to_ms(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value) / 1_000_000.0


class OllamaHttp:
    def __init__(self, base_url: str, reuse: bool) -> None:
        parsed = urlparse(base_url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 11434
        self.reuse = reuse
        self._conn: http.client.HTTPConnection | None = None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _connection(self) -> http.client.HTTPConnection:
        if self.reuse and self._conn is not None:
            return self._conn
        conn = http.client.HTTPConnection(self.host, self.port, timeout=180)
        if self.reuse:
            self._conn = conn
        return conn

    def request(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        send = None if body is None else {key: value for key, value in body.items() if not str(key).startswith("_")}
        payload = compact(send).encode() if send is not None else b""
        headers = {"Content-Type": "application/json", "Connection": "keep-alive" if self.reuse else "close"}
        started = time.perf_counter()
        conn = self._connection()
        try:
            conn.request("POST" if body is not None else "GET", path, payload, headers)
            response = conn.getresponse()
            raw = response.read()
        except Exception:
            self.close()
            raise
        if not self.reuse:
            conn.close()
        elif response.will_close:
            self.close()
        wall_ms = (time.perf_counter() - started) * 1000.0
        if response.status >= 400:
            raise RuntimeError(f"{path} HTTP {response.status}: {raw[:300]!r}")
        parsed = json.loads(raw.decode()) if raw else {}
        parsed["_wall_ms"] = wall_ms
        parsed["_bytes_in"] = len(payload)
        parsed["_bytes_out"] = len(raw)
        return parsed


def metrics(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "wall_ms": response.get("_wall_ms"),
        "load_ms": ns_to_ms(response.get("load_duration")),
        "prefill_ms": ns_to_ms(response.get("prompt_eval_duration")),
        "generation_ms": ns_to_ms(response.get("eval_duration")),
        "total_ms": ns_to_ms(response.get("total_duration")),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
        "done_reason": response.get("done_reason"),
        "bytes_in": response.get("_bytes_in"),
        "bytes_out": response.get("_bytes_out"),
    }


def snapshot(client: OllamaHttp) -> dict[str, Any]:
    version = client.request("/api/version")
    ps = client.request("/api/ps")
    models = ps.get("models") or []
    resident = next((item for item in models if item.get("name") == MODEL or item.get("model") == MODEL), None)
    return {
        "ollama_version": version.get("version"),
        "ps": ps,
        "resident": {
            "name": (resident or {}).get("name") or (resident or {}).get("model"),
            "digest": (resident or {}).get("digest"),
            "size": (resident or {}).get("size"),
            "size_vram": (resident or {}).get("size_vram"),
            "expires_at": (resident or {}).get("expires_at"),
            "context": ((resident or {}).get("context_length")
                        or ((resident or {}).get("details") or {}).get("context_length")),
            "details": (resident or {}).get("details"),
        },
    }


def assert_safe_resident(before: dict[str, Any], after: dict[str, Any], experiment: str) -> None:
    before_res = before["resident"]
    after_res = after["resident"]
    if not after_res.get("name"):
        raise RuntimeError(f"{experiment}: model is no longer resident; aborting")
    before_ctx = before_res.get("context")
    after_ctx = after_res.get("context")
    if after_ctx not in (None, NUM_CTX) and after_ctx != NUM_CTX:
        raise RuntimeError(f"{experiment}: context changed {before_ctx} -> {after_ctx}; aborting")
    if before_res.get("digest") and after_res.get("digest") and before_res["digest"] != after_res["digest"]:
        raise RuntimeError(f"{experiment}: digest changed; aborting")


def run_series(
    client: OllamaHttp,
    name: str,
    path: str,
    bodies: list[dict[str, Any]],
    warmup: int,
) -> dict[str, Any]:
    measured: list[dict[str, Any]] = []
    for index, body in enumerate(bodies):
        response = client.request(path, body)
        row = metrics(response)
        row["index"] = index
        row["warmup"] = index < warmup
        if index >= warmup:
            measured.append(row)
    keys = ("wall_ms", "load_ms", "prefill_ms", "generation_ms", "total_ms")
    return {
        "name": name,
        "path": path,
        "warmup": warmup,
        "sample_count": len(measured),
        "request_bytes": bodies[warmup]["_probe_bytes"] if warmup < len(bodies) and "_probe_bytes" in bodies[warmup] else bodies[0].get("_probe_bytes"),
        "options": bodies[warmup].get("options") if warmup < len(bodies) else bodies[0].get("options"),
        "has_format": bool((bodies[warmup] if warmup < len(bodies) else bodies[0]).get("format")),
        "stream": (bodies[warmup] if warmup < len(bodies) else bodies[0]).get("stream"),
        "keep_alive": (bodies[warmup] if warmup < len(bodies) else bodies[0]).get("keep_alive"),
        "distributions": {key: summarize([row[key] for row in measured if row.get(key) is not None]) for key in keys},
        "prompt_eval_count": summarize([row["prompt_eval_count"] for row in measured if row.get("prompt_eval_count") is not None]),
        "eval_count": summarize([row["eval_count"] for row in measured if row.get("eval_count") is not None]),
        "rows": measured,
    }


def chat_body(
    messages: list[dict[str, Any]],
    *,
    num_predict: int,
    fmt: Any | None,
    stream: bool = False,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
        "think": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "num_ctx": NUM_CTX,
            "num_predict": num_predict,
            "seed": 0,
        },
    }
    if fmt is not None:
        body["format"] = fmt
    encoded = compact(body)
    body["_probe_bytes"] = len(encoded.encode())
    body["_messages_bytes"] = len(compact(messages).encode())
    body["_format_bytes"] = len(compact(fmt).encode()) if fmt is not None else 0
    return body


def generate_body(prompt: str, num_predict: int) -> dict[str, Any]:
    body = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "num_ctx": NUM_CTX,
            "num_predict": num_predict,
            "seed": 0,
        },
    }
    encoded = compact(body)
    body["_probe_bytes"] = len(encoded.encode())
    return body


def dump_planner_payload(root: Path, utterance: str) -> dict[str, Any]:
    sys.path.insert(0, str(root / "src"))
    from home_cortex.ollama import (  # type: ignore
        PLANNER_KEEP_ALIVE,
        PLANNER_NUM_CTX,
        PLANNER_NUM_PREDICT,
        PLANNER_SEED,
        planner_chat_messages,
    )
    from home_cortex.semantic_planner_benchmark import build_json_fact_service  # type: ignore

    service, context = build_json_fact_service(
        root / "benchmarks/fixtures/semantic-contract",
        root / "schemas/edge",
        None,
    )
    capabilities = service.engine.schema.planner_capability_payload()
    output_schema = dict(service.engine.schema.planner_output_schema())
    messages = planner_chat_messages(
        [{"role": "user", "content": utterance}],
        capabilities,
        household_now=context.current_time.isoformat(),
        output_schema=output_schema,
    )
    from home_cortex.semantic_transport import transport_for
    output_schema = transport_for(output_schema).schema
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": KEEP_ALIVE,
        "format": output_schema,
        "options": {
            "temperature": 0,
            "num_ctx": NUM_CTX,
            "num_predict": PLANNER_NUM_PREDICT,
            "seed": PLANNER_SEED,
        },
    }
    encoded = compact(body)
    return {
        "utterance": utterance,
        "household_now": context.current_time.isoformat(),
        "message_count": len(messages),
        "messages_bytes": len(compact(messages).encode()),
        "format_bytes": len(compact(output_schema).encode()),
        "request_bytes": len(encoded.encode()),
        "request_sha256": sha256_text(encoded),
        "keep_alive": PLANNER_KEEP_ALIVE,
        "options": body["options"],
        "body": body,
    }


async def run_cortex_planner(root: Path, url: str, utterance: str, warmup: int, repeat: int) -> dict[str, Any]:
    sys.path.insert(0, str(root / "src"))
    from home_cortex.ollama import OllamaService, PLANNER_NUM_CTX  # type: ignore
    from home_cortex.profiling import trace_request  # type: ignore
    from home_cortex.semantic_conversation import SemanticConversationService  # type: ignore
    from home_cortex.semantic_planner_benchmark import build_json_fact_service  # type: ignore

    if PLANNER_NUM_CTX != NUM_CTX:
        raise RuntimeError(f"planner num_ctx {PLANNER_NUM_CTX} != probe {NUM_CTX}")
    model = OllamaService(url, MODEL)
    service, context = build_json_fact_service(
        root / "benchmarks/fixtures/semantic-contract",
        root / "schemas/edge",
        model,
    )
    conversations = SemanticConversationService(service)
    rows = []
    try:
        for index in range(warmup + repeat):
            with trace_request() as trace:
                started = time.perf_counter()
                answer = await conversations.try_answer(
                    [{"role": "user", "content": utterance}],
                    context=context,
                )
                elapsed = (time.perf_counter() - started) * 1000.0
            if index < warmup:
                continue
            calls = [event for event in trace.events if event["stage"].startswith("llm.")]
            rows.append({
                "index": index - warmup,
                "wall_ms": elapsed,
                "status": None if answer is None else str(getattr(answer.result, "status", None)),
                "llm_calls": len(calls),
                "calls": [{
                    "duration_ms": event.get("duration_ms"),
                    "load_ms": event.get("load_ms"),
                    "prefill_ms": event.get("prefill_ms"),
                    "generation_ms": event.get("generation_ms"),
                    "input_tokens": event.get("input_tokens"),
                    "output_tokens": event.get("output_tokens"),
                } for event in calls],
            })
    finally:
        await model.close()
    load_values = [call["load_ms"] for row in rows for call in row["calls"] if call.get("load_ms") is not None]
    wall_values = [row["wall_ms"] for row in rows]
    return {
        "name": "cortex_isolated_planner",
        "utterance": utterance,
        "warmup": warmup,
        "sample_count": len(rows),
        "distributions": {
            "wall_ms": summarize(wall_values),
            "load_ms": summarize(load_values),
            "prefill_ms": summarize([call["prefill_ms"] for row in rows for call in row["calls"] if call.get("prefill_ms") is not None]),
            "generation_ms": summarize([call["generation_ms"] for row in rows for call in row["calls"] if call.get("generation_ms") is not None]),
            "llm_duration_ms": summarize([call["duration_ms"] for row in rows for call in row["calls"] if call.get("duration_ms") is not None]),
        },
        "rows": rows,
    }


def gpu_snapshot() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
    except Exception as error:
        return {"error": str(error)}
    parts = [item.strip() for item in output.split(",")]
    if len(parts) < 5:
        return {"raw": output}
    return {
        "name": parts[0],
        "memory_used_mib": float(parts[1]),
        "memory_total_mib": float(parts[2]),
        "utilization_gpu_pct": float(parts[3]),
        "utilization_memory_pct": float(parts[4]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-url", default="http://ollama:11434")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isolated-root", type=Path, help="Isolated Cortex package root for planner dump/run")
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--planner-repeat", type=int, default=8)
    parser.add_argument("--skip-cortex", action="store_true")
    parser.add_argument("--utterance", default="Who am I?")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--num-ctx", type=int, default=NUM_CTX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global MODEL, NUM_CTX
    MODEL = args.model
    NUM_CTX = args.num_ctx
    client = OllamaHttp(args.ollama_url, reuse=True)
    started = time.time()
    report: dict[str, Any] = {
        "objective": "Explain warm-request Ollama load_duration with the model resident",
        "constraints": {
            "matched_num_ctx": NUM_CTX,
            "keep_alive": KEEP_ALIVE,
            "model": MODEL,
            "no_unload": True,
            "no_settings_change": True,
        },
        "host_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_before": gpu_snapshot(),
    }
    try:
        env_before = snapshot(client)
        report["environment_before"] = env_before
        if not env_before["resident"].get("name"):
            raise RuntimeError(f"{MODEL} is not resident; refusing to start a cold load")
        resident_ctx = env_before["resident"].get("context")
        if resident_ctx not in (None, NUM_CTX):
            raise RuntimeError(f"resident context is {resident_ctx}, not {NUM_CTX}; refusing to mismatch")

        planner = None
        if args.isolated_root:
            planner = dump_planner_payload(args.isolated_root, args.utterance)
            planner_meta = {key: value for key, value in planner.items() if key != "body"}
            report["planner_payload"] = planner_meta
            large_messages = planner["body"]["messages"]
            planner_format = planner["body"]["format"]
        else:
            large_messages = [
                {"role": "system", "content": "x" * 18000},
                {"role": "user", "content": args.utterance},
            ]
            planner_format = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}

        tiny_messages = [{"role": "user", "content": TINY_USER}]
        n = args.warmup + args.repeat
        planner_n = args.warmup + args.planner_repeat
        experiments = []

        def add(name: str, path: str, bodies: list[dict[str, Any]]) -> None:
            before = snapshot(client)
            gpu_pre = gpu_snapshot()
            series = run_series(client, name, path, bodies, args.warmup)
            gpu_post = gpu_snapshot()
            after = snapshot(client)
            assert_safe_resident(before, after, name)
            series["gpu_before"] = gpu_pre
            series["gpu_after"] = gpu_post
            series["resident_after"] = after["resident"]
            experiments.append(series)

        show_bodies = [{"model": MODEL, "_probe_bytes": len(compact({"model": MODEL}).encode())} for _ in range(n)]
        add("api_show_cached", "/api/show", show_bodies)
        uncached_show = {"model": MODEL, "options": {"seed": 0}, "system": "probe-bypass-show-cache"}
        add("api_show_uncached", "/api/show",
            [{**uncached_show, "_probe_bytes": len(compact(uncached_show).encode())} for _ in range(n)])
        add("tiny_generate_predict1", "/api/generate", [generate_body(TINY_USER, 1) for _ in range(n)])
        add("tiny_chat_predict1", "/api/chat", [chat_body(tiny_messages, num_predict=1, fmt=None) for _ in range(n)])
        add("tiny_chat_with_format_predict1", "/api/chat",
            [chat_body(tiny_messages, num_predict=1, fmt=planner_format) for _ in range(n)])
        add("large_chat_no_format_predict1", "/api/chat",
            [chat_body(large_messages, num_predict=1, fmt=None) for _ in range(n)])
        add("large_chat_with_format_predict1", "/api/chat",
            [chat_body(large_messages, num_predict=1, fmt=planner_format) for _ in range(n)])
        add("planner_matched_predict384", "/api/chat",
            [chat_body(large_messages, num_predict=384, fmt=planner_format) for _ in range(planner_n)])

        fresh = OllamaHttp(args.ollama_url, reuse=False)
        try:
            before = snapshot(fresh)
            series = run_series(
                fresh,
                "tiny_chat_predict1_new_tcp",
                "/api/chat",
                [chat_body(tiny_messages, num_predict=1, fmt=None) for _ in range(n)],
                args.warmup,
            )
            after = snapshot(fresh)
            assert_safe_resident(before, after, series["name"])
            series["resident_after"] = after["resident"]
            experiments.append(series)
        finally:
            fresh.close()

        report["experiments"] = experiments

        if args.isolated_root and not args.skip_cortex:
            import asyncio
            try:
                report["cortex_isolated_planner"] = asyncio.run(
                    run_cortex_planner(args.isolated_root, args.ollama_url, args.utterance, args.warmup, args.planner_repeat)
                )
                after = snapshot(client)
                assert_safe_resident(env_before, after, "cortex_isolated_planner")
            except Exception as error:
                report["cortex_isolated_planner_error"] = f"{type(error).__name__}: {error}"

        report["environment_after"] = snapshot(client)
        report["gpu_after"] = gpu_snapshot()
        report["elapsed_s"] = time.time() - started
        report["pid"] = os.getpid()
        report["python"] = sys.version
    finally:
        client.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if "experiments" in report or "environment_before" in report:
            args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(compact({
        "wrote": str(args.output),
        "experiments": [item["name"] for item in report.get("experiments", [])],
        "resident": (report.get("environment_after") or report.get("environment_before") or {}).get("resident"),
        "cortex_error": report.get("cortex_isolated_planner_error"),
    }))


if __name__ == "__main__":
    main()
