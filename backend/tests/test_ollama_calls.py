import asyncio

import httpx
import pytest

from services.ollama_service import ollama_service


@pytest.fixture
def respond(monkeypatch):
    """Answer every Ollama call with one canned JSON response."""

    def install(payload, status=200):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(status, json=payload)
        )
        original = httpx.AsyncClient

        def client(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", client)

    return install


@pytest.mark.asyncio
async def test_embed_returns_the_vector(respond):
    respond({"embedding": [0.1, 0.2, 0.3]})

    assert await ollama_service.embed("hello") == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_names_the_model_when_it_cannot_embed(respond):
    # Pointing OLLAMA_EMBED_MODEL at a chat model used to surface as a bare
    # KeyError('embedding') reading "Ingestion failed: 'embedding'".
    respond({"error": "does not support embeddings"})

    with pytest.raises(RuntimeError, match="did not return an embedding"):
        await ollama_service.embed("hello")


@pytest.mark.asyncio
async def test_embed_rejects_an_empty_vector(respond):
    respond({"embedding": []})

    with pytest.raises(RuntimeError, match="did not return an embedding"):
        await ollama_service.embed("hello")


@pytest.mark.asyncio
async def test_embed_batch_keeps_the_input_order(respond):
    respond({"embedding": [1.0]})

    assert await ollama_service.embed_batch(["a", "b", "c"]) == [[1.0], [1.0], [1.0]]


@pytest.mark.asyncio
async def test_chat_returns_the_message_content(respond):
    respond({"message": {"content": "Hello there"}})

    assert await ollama_service.chat([{"role": "user", "content": "hi"}]) == "Hello there"


@pytest.mark.asyncio
async def test_list_models_returns_the_models(respond):
    respond({"models": [{"name": "llama3"}]})

    assert await ollama_service.list_models() == [{"name": "llama3"}]


@pytest.mark.asyncio
async def test_is_available_is_true_on_a_200(respond):
    respond({"models": []})

    assert await ollama_service.is_available() is True


@pytest.mark.asyncio
async def test_embed_batch_sends_several_at_once(monkeypatch):
    # One request per chunk, strictly sequential, cost the sum of every round
    # trip: 16.8s for 359 chunks at 40ms against 4.2s four at a time.
    from app.core import config

    monkeypatch.setattr(config.settings, "embed_concurrency", 4)
    in_flight, peak = 0, 0

    async def embed(text):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return [1.0]

    monkeypatch.setattr(ollama_service, "embed", embed)

    await ollama_service.embed_batch([str(i) for i in range(12)])

    assert peak > 1, "the requests went out one at a time"
    assert peak <= 4, "the concurrency limit was not respected"


@pytest.mark.asyncio
async def test_embed_batch_stops_the_queue_when_one_call_fails(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "embed_concurrency", 2)
    started = []

    async def embed(text):
        started.append(text)
        await asyncio.sleep(0.01)
        raise RuntimeError("model gone")

    monkeypatch.setattr(ollama_service, "embed", embed)

    with pytest.raises(RuntimeError, match="model gone"):
        await ollama_service.embed_batch([str(i) for i in range(20)])

    # The ones still waiting on the semaphore are cancelled, not sent.
    assert len(started) < 20


@pytest.mark.asyncio
async def test_ingest_embeds_through_embed_batch(monkeypatch, tmp_path):
    # ingest_document had its own sequential loop, so the helper written for
    # this job was never used and the two could drift.
    from app.core import config
    from services.rag_service import RAGService

    monkeypatch.setattr(config.settings, "chroma_persist_dir", str(tmp_path))
    service = RAGService()
    calls = []

    async def embed_batch(texts):
        calls.append(len(texts))
        return [[1.0] for _ in texts]

    monkeypatch.setattr("services.rag_service.ollama_service.embed_batch", embed_batch)

    _, chunks_added = await service.ingest_document(
        "report.txt", b"one sentence. two sentence.", "docs"
    )

    assert calls == [chunks_added]
