"""Opt-in, bounded request traces. Never retain prompts, identities or results.

Stage intervals are nested, not additive. Model usage is provider-reported;
missing metrics remain None, and non-streaming calls cannot measure TTFT.
"""
from __future__ import annotations

import inspect
import json
import logging
import math
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter
from typing import Any

_call: ContextVar[dict | None] = ContextVar("model_call", default=None)

_active: ContextVar[RequestTrace | None] = ContextVar("request_trace", default=None)


class RequestTrace:
    def __init__(self, limit: int = 512):
        self.started = perf_counter()
        self.events: list[dict[str, Any]] = []
        self.limit = limit
        self.dropped = 0

    def begin(self, name: str) -> dict[str, Any]:
        event = {"stage": name, "start_ms": (perf_counter() - self.started) * 1000}
        if len(self.events) < self.limit:
            self.events.append(event)
        else:
            self.dropped += 1
        return event

    def end(self, event: dict[str, Any]) -> None:
        event["duration_ms"] = (perf_counter() - self.started) * 1000 - event["start_ms"]


@contextmanager
def trace_request(limit: int = 512):
    trace = RequestTrace(limit)
    token = _active.set(trace)
    try:
        yield trace
    finally:
        _active.reset(token)


def stage(name: str):
    """Time sync/async stages only inside an explicit trace_request scope."""
    def decorate(function):
        @wraps(function)
        async def asynchronous(*args, **kwargs):
            trace = _active.get()
            if trace is None:
                return await function(*args, **kwargs)
            event = trace.begin(name)
            try:
                return await function(*args, **kwargs)
            finally:
                trace.end(event)

        @wraps(function)
        def synchronous(*args, **kwargs):
            trace = _active.get()
            if trace is None:
                return function(*args, **kwargs)
            event = trace.begin(name)
            try:
                return function(*args, **kwargs)
            finally:
                trace.end(event)
        return asynchronous if inspect.iscoroutinefunction(function) else synchronous
    return decorate


def _usage(event, response):
    if isinstance(response, dict):
        usage = response.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        details = usage.get("completion_tokens_details")
        details = details if isinstance(details, dict) else {}
        pairs = {"input_tokens": usage.get("prompt_tokens"),
                 "output_tokens": usage.get("completion_tokens"),
                 "reasoning_tokens": details.get("reasoning_tokens")}
    else:
        pairs = {"input_tokens": getattr(response, "prompt_eval_count", None),
                 "output_tokens": getattr(response, "eval_count", None)}
        message = getattr(response, 'message', None)
        if message is not None:
            for target, source in (('content_chars', 'content'), ('reasoning_chars', 'thinking')):
                value = getattr(message, source, None)
                if isinstance(value, str):
                    event[target] = event.get(target, 0) + len(value)
        for target, source in (("prefill_ms", "prompt_eval_duration"),
                               ("generation_ms", "eval_duration"),
                               ("load_ms", "load_duration")):
            value = getattr(response, source, None)
            if isinstance(value, (int, float)):
                pairs[target] = value / 1_000_000
    event.update({key: value for key, value in pairs.items()
                  if isinstance(value, (int, float)) and not isinstance(value, bool)
                  and math.isfinite(value) and value >= 0})


def model_call(provider: str):
    """Record each transport call, including retries and streamed completion.

Ollama streams expose tokens and generation timing in their final chunk.
The wall interval includes transport/SDK work; it is not inference time.
"""
    def decorate(function):
        @wraps(function)
        async def wrapped(*args, **kwargs):
            trace = _active.get()
            if trace is None:
                return await function(*args, **kwargs)
            event = trace.begin("llm." + provider)
            payload = kwargs if provider == 'ollama' else (args[1] if len(args) > 1 else {})
            event['purpose'] = ('semantic_interpretation' if isinstance(payload, dict) and
                                (payload.get('format') or payload.get('response_format'))
                                else 'chat_or_tool_selection')
            event.update(input_tokens=None, output_tokens=None, reasoning_tokens=None,
                         ttft_ms=None, generation_ms=None, prefill_ms=None, load_ms=None,
                         post_ttft_ms=None)
            try:
                response = await function(*args, **kwargs)
            except BaseException as error:
                event["error"] = type(error).__name__
                trace.end(event)
                raise
            if hasattr(response, "__aiter__"):
                async def stream():
                    token = _call.set(event)
                    try:
                        async for chunk in response:
                            message = getattr(chunk, "message", None)
                            if event["ttft_ms"] is None and message and (
                                message.content or getattr(message, "thinking", None)
                                or message.tool_calls
                            ):
                                event["ttft_ms"] = (perf_counter() - trace.started) * 1000 - event["start_ms"]
                            _usage(event, chunk)
                            yield chunk
                    except BaseException as error:
                        event["error"] = type(error).__name__
                        raise
                    finally:
                        try:
                            close = getattr(response, "aclose", None)
                            if close:
                                await close()
                        finally:
                            _call.reset(token)
                            trace.end(event)
                            if event["ttft_ms"] is not None:
                                event["post_ttft_ms"] = event["duration_ms"] - event["ttft_ms"]
                return stream()
            _usage(event, response)
            trace.end(event)
            return response
        return wrapped
    return decorate


def observe_usage(response):
    """Capture raw provider usage before an adapter discards its envelope."""
    event = _call.get()
    if event is not None:
        _usage(event, response)
        choices = response.get('choices') or []
        delta = choices[0].get('delta') if choices and isinstance(choices[0], dict) else None
        trace = _active.get()
        if trace and event['ttft_ms'] is None and isinstance(delta, dict) and any(
            delta.get(key) for key in ('content', 'reasoning', 'reasoning_content', 'tool_calls')
        ):
            event['ttft_ms'] = (perf_counter() - trace.started) * 1000 - event['start_ms']


def stream_model_call(provider: str):
    def decorate(function):
        @model_call(provider)
        async def open_stream(*args, **kwargs):
            return function(*args, **kwargs)

        @wraps(function)
        async def wrapped(*args, **kwargs):
            stream = await open_stream(*args, **kwargs)
            try:
                async for chunk in stream:
                    yield chunk
            finally:
                await stream.aclose()
        return wrapped
    return decorate


class RequestTraceMiddleware:
    """ASGI tracing includes the final streaming body and cancellation cleanup."""

    def __init__(self, app, enabled: bool = False):
        self.app = app
        self.enabled = enabled

    async def __call__(self, scope, receive, send):
        if not self.enabled or scope['type'] != 'http':
            return await self.app(scope, receive, send)
        with trace_request() as trace:
            event = trace.begin('http.total')
            try:
                await self.app(scope, receive, send)
            except BaseException as error:
                event['error'] = type(error).__name__
                raise
            finally:
                trace.end(event)
                logging.getLogger('uvicorn.error.home_cortex.profiling').info('request_profile %s', json.dumps({
                    'request_id': scope.get('state', {}).get('request_id'),
                    'events': trace.events, 'dropped': trace.dropped,
                }, separators=(',', ':')))
