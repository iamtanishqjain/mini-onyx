import json

import httpx
import pytest

from services.ollama_service import ollama_service


def ndjson(*objects):
    return "".join(json.dumps(o) + "\n" for o in objects)


@pytest.fixture
def stream_body(monkeypatch):
    """Make chat_stream read a canned NDJSON body instead of calling Ollama."""

    def install(body):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
        original = httpx.AsyncClient

        def client(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", client)

    return install


async def collect():
    messages = [{"role": "user", "content": "hi"}]
    return [chunk async for chunk in ollama_service.chat_stream(messages)]


@pytest.mark.asyncio
async def test_tokens_are_yielded_in_order(stream_body):
    stream_body(
        ndjson(
            {"message": {"content": "Hel"}},
            {"message": {"content": "lo"}},
            {"done": True},
        )
    )

    assert await collect() == ["Hel", "lo"]


@pytest.mark.asyncio
async def test_an_error_line_raises_rather_than_ending_the_stream(stream_body):
    # Skipping it ended the stream with no tokens, so the caller showed an
    # empty reply and no sign that anything had failed.
    stream_body(ndjson({"error": "model 'nope' not found"}))

    with pytest.raises(RuntimeError, match="not found"):
        await collect()


@pytest.mark.asyncio
async def test_an_error_after_some_tokens_still_raises(stream_body):
    stream_body(
        ndjson(
            {"message": {"content": "partial"}},
            {"error": "context length exceeded"},
        )
    )

    with pytest.raises(RuntimeError, match="context length"):
        await collect()


@pytest.mark.asyncio
async def test_malformed_lines_are_skipped(stream_body):
    stream_body("not json\n" + ndjson({"message": {"content": "ok"}}, {"done": True}))

    assert await collect() == ["ok"]


@pytest.mark.asyncio
async def test_nothing_is_yielded_after_done(stream_body):
    stream_body(
        ndjson(
            {"message": {"content": "first"}},
            {"done": True},
            {"message": {"content": "should not arrive"}},
        )
    )

    assert await collect() == ["first"]
