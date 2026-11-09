import asyncio
import json
from collections.abc import AsyncGenerator

import httpx

from app.core.config import settings


class OllamaService:
    """Wraps the Ollama REST API for chat and embeddings.

    The module-level instance is created at import time, so these read from
    settings on each access rather than copying them in __init__. Otherwise a
    setting changed after import, which is what tests and any runtime reload
    do, would never reach the service.
    """

    @property
    def base_url(self) -> str:
        return settings.ollama_base_url

    @property
    def chat_model(self) -> str:
        return settings.ollama_chat_model

    @property
    def embed_model(self) -> str:
        return settings.ollama_embed_model

    # ── Connection check ────────────────────────────────────────────────────

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    # ── Available models ────────────────────────────────────────────────────

    async def list_models(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            # `or []` rather than a default, since a null models key returns
            # the null and the caller then iterates None.
            models = resp.json().get("models") or []
            return [m for m in models if isinstance(m, dict)]

    # ── Chat (streaming) ────────────────────────────────────────────────────

    async def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yields text chunks as they arrive from Ollama."""
        payload = {
            "model": model or self.chat_model,
            "messages": messages,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=120) as client, client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Ollama reports a mid-stream failure, a model that will not
                # load or a context that overflows, as an error line rather
                # than a status code. Ignoring it ended the stream with no
                # tokens at all, so the caller showed an empty reply and no
                # sign that anything had gone wrong.
                if data.get("error"):
                    raise RuntimeError(f"Ollama: {data['error']}")

                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done"):
                    break

    # ── Chat (non-streaming, returns full response) ─────────────────────────

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
    ) -> str:
        payload = {
            "model": model or self.chat_model,
            "messages": messages,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    # ── Embeddings ──────────────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """Get embedding vector for a piece of text."""
        payload = {
            "model": self.embed_model,
            "prompt": text,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{self.base_url}/api/embeddings", json=payload)
            resp.raise_for_status()
            body = resp.json()

        embedding = body.get("embedding")
        if not embedding:
            # Usually OLLAMA_EMBED_MODEL pointing at a chat model. Indexing the
            # key straight out raised a bare KeyError('embedding'), which
            # reached the user as "Ingestion failed: 'embedding'".
            reason = body.get("error") or "the response contained no embedding"
            raise RuntimeError(f"Model '{self.embed_model}' did not return an embedding: {reason}")
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed several texts, a few in flight at a time.

        Ollama has no batch endpoint, so this is one request per text. Sending
        them strictly one after another cost the sum of every round trip: at
        40ms a call, 359 chunks took 16.8s against 4.2s four at a time.

        The limit is per call, so two documents embedding at once can have
        twice EMBED_CONCURRENCY in flight.
        """
        semaphore = asyncio.Semaphore(settings.embed_concurrency)

        async def embed_one(text: str) -> list[float]:
            async with semaphore:
                return await self.embed(text)

        tasks = [asyncio.create_task(embed_one(text)) for text in texts]
        try:
            # gather keeps the order of tasks, so chunk i stays embedding i.
            return await asyncio.gather(*tasks)
        except Exception:
            # Otherwise the requests still queued behind the semaphore carry on
            # against a model that has already failed once.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


# Singleton
ollama_service = OllamaService()
