import asyncio
import json

import pytest
from ollama import ChatResponse

from home_cortex.ollama import OllamaService, _semantic_planner_examples
from home_cortex.request_tracing import model_call, stage, trace_request


def test_example_cache_does_not_expose_mutable_messages():
    before = _semantic_planner_examples()
    first = before[0].copy()
    before[0]['content'] = 'mutated'
    before.clear()
    assert _semantic_planner_examples()[0] == first


@pytest.mark.asyncio
async def test_traces_are_isolated_bounded_and_do_not_capture_arguments():
    @stage('test')
    async def operation(secret):
        await asyncio.sleep(0)
        return secret

    async def run(secret, count):
        with trace_request(limit=1) as trace:
            for _ in range(count):
                assert await operation(secret) == secret
        return trace

    one, two = await asyncio.gather(run('private-a', 1), run('private-b', 3))
    assert len(one.events) == len(two.events) == 1
    assert one.dropped == 0 and two.dropped == 2
    assert 'private' not in json.dumps(one.events + two.events)
    assert await operation('outside') == 'outside'
    assert len(two.events) == 1


@pytest.mark.asyncio
async def test_each_call_preserves_usage_and_errors_without_stale_metrics():
    @model_call('test')
    async def request(value):
        if value is None:
            raise ValueError('sensitive error detail')
        return {'usage': {'prompt_tokens': value, 'completion_tokens': 2}}

    with trace_request() as trace:
        await request(10)
        await request(20)
        with pytest.raises(ValueError):
            await request(None)
    assert [e['input_tokens'] for e in trace.events] == [10, 20, None]
    assert all(e['ttft_ms'] is None for e in trace.events)
    assert trace.events[-1]['error'] == 'ValueError'
    assert 'sensitive' not in json.dumps(trace.events)


@pytest.mark.asyncio
async def test_stream_ttft_ignores_empty_chunks_and_preserves_final_usage():
    closed = []

    async def chunks():
        try:
            yield ChatResponse(message={'role':'assistant','content':''})
            yield ChatResponse(message={'role':'assistant','content':'hello'})
            yield ChatResponse(message={'role':'assistant','content':''}, done=True,
                               prompt_eval_count=11, eval_count=3, eval_duration=1_000_000)
        finally:
            closed.append(True)

    class Client:
        async def chat(self, **kwargs):
            return chunks()

    service = OllamaService('http://unused', 'test', client=Client())
    with trace_request() as trace:
        output = [c async for c in service.stream_chat_with_tools([], [])]
    event = trace.events[0]
    assert len(output) == 3 and closed == [True]
    assert event['input_tokens'] == 11 and event['output_tokens'] == 3
    assert event['generation_ms'] == 1
    assert 0 <= event['ttft_ms'] <= event['duration_ms']
    assert event['post_ttft_ms'] >= 0


@pytest.mark.asyncio
async def test_stream_cancellation_closes_underlying_stream():
    closed = []

    async def chunks():
        try:
            yield ChatResponse(message={'role':'assistant','content':'first'})
            yield ChatResponse(message={'role':'assistant','content':'second'})
        finally:
            closed.append(True)

    class Client:
        async def chat(self, **kwargs):
            return chunks()

    with trace_request() as trace:
        stream = OllamaService('http://unused', 'test', client=Client()).stream_chat_with_tools([], [])
        await anext(stream)
        await stream.aclose()
    assert closed == [True]
    assert trace.events[0]['error'] == 'GeneratorExit'
    assert trace.events[0]['input_tokens'] is None


@pytest.mark.asyncio
async def test_http_trace_includes_stream_completion_and_omits_request_data(caplog):
    from home_cortex.request_tracing import RequestTraceMiddleware

    @stage('stream.work')
    async def produce():
        await asyncio.sleep(0)

    async def application(scope, receive, send):
        scope['state']['request_id'] = 'server-generated'
        await send({'type':'http.response.start','status':200,'headers':[]})
        await produce()
        await send({'type':'http.response.body','body':b'private answer','more_body':False})

    sent = []
    async def send(message):
        sent.append(message)
    async def receive():
        return {'type':'http.request','body':b'private query'}

    with caplog.at_level('INFO', logger='uvicorn.error.home_cortex.request_tracing'):
        await RequestTraceMiddleware(application, enabled=True)(
            {'type':'http','state':{},'path':'/private-path'}, receive, send)
    payload = json.loads(caplog.records[-1].message.removeprefix('request_profile '))
    outer, inner = payload['events']
    assert outer['stage'] == 'http.total'
    assert outer['duration_ms'] >= inner['start_ms'] + inner['duration_ms'] - outer['start_ms']
    assert sent[-1]['more_body'] is False
    assert 'private' not in caplog.text


@pytest.mark.asyncio
async def test_openrouter_stream_usage_is_captured_before_adapter_discards_it():
    import httpx
    from home_cortex.openrouter import OpenRouterService

    def handler(request):
        return httpx.Response(200, text=(
            'data: {"choices":[{"delta":{"reasoning":"private reasoning"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":30,"completion_tokens":9,"completion_tokens_details":{"reasoning_tokens":4}}}\n\n'
            'data: [DONE]\n\n'
        ))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = OpenRouterService('http://test', 'model', api_key='private-key', client=client)
        with trace_request() as trace:
            chunks = [chunk async for chunk in service.stream_chat_with_tools([], [])]
    assert len(chunks) == 1
    event = trace.events[0]
    assert (event['input_tokens'], event['output_tokens'], event['reasoning_tokens']) == (30,9,4)
    assert event['ttft_ms'] is not None
    assert 'private' not in json.dumps(trace.events)
