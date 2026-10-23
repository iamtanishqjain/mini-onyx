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
