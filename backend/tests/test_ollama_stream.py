import json

import httpx
import pytest

from services.ollama_service import ollama_service


def ndjson_transport(lines, status=200):
    body = "".join(json.dumps(line) + "\n" for line in lines)

    def handler(request):
        return httpx.Response(status, text=body)

    return httpx.MockTransport(handler)


@pytest.fixture
def stream_from(monkeypatch):
    def install(lines, status=200):
        transport = ndjson_transport(lines, status)
        original = httpx.AsyncClient

        def client(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", client)

    return install


async def collect(messages=None):
    return [c async for c in ollama_service.chat_stream(messages or [{"role": "user", "content": "hi"}])]


@pytest.mark.asyncio
async def test_tokens_are_yielded_in_order(stream_from):
    stream_from([
        {"message": {"content": "Hel"}},
        {"message": {"content": "lo"}},
        {"done": True},
    ])

    assert await collect() == ["Hel", "lo"]


@pytest.mark.asyncio
async def test_an_error_line_raises_rather_than_ending_the_stream(stream_from):
    # Ignoring it left the caller with an empty reply and no error at all.
    stream_from([{"error": "model 'nope' not found"}])

    with pytest.raises(RuntimeError, match="not found"):
        await collect()


@pytest.mark.asyncio
async def test_an_error_after_some_tokens_still_raises(stream_from):
    stream_from([
        {"message": {"content": "partial"}},
        {"error": "context length exceeded"},
    ])

    with pytest.raises(RuntimeError, match="context length"):
        await collect()


@pytest.mark.asyncio
async def test_malformed_lines_are_skipped(stream_from, monkeypatch):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text='not json\n{"message":{"content":"ok"}}\n{"done":true}\n')
    )
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **k: original(*a, **{**k, "transport": transport})
    )

    assert await collect() == ["ok"]


@pytest.mark.asyncio
async def test_nothing_is_yielded_after_done(stream_from):
    stream_from([
        {"message": {"content": "first"}},
        {"done": True},
        {"message": {"content": "should not arrive"}},
    ])

    assert await collect() == ["first"]
