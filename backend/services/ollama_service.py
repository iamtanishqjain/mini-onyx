import httpx
import json
from typing import AsyncGenerator, List, Optional
from app.core.config import settings


class OllamaService:
    """Wraps the Ollama REST API for chat and embeddings."""

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.chat_model = settings.ollama_chat_model
        self.embed_model = settings.ollama_embed_model

    # ── Connection check ────────────────────────────────────────────────────

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    # ── Available models ────────────────────────────────────────────────────

    async def list_models(self) -> List[dict]:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            return resp.json().get("models", [])

    # ── Chat (streaming) ────────────────────────────────────────────────────

    async def chat_stream(
        self,
        messages: List[dict],
        model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Yields text chunks as they arrive from Ollama."""
        payload = {
            "model": model or self.chat_model,
            "messages": messages,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
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
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

    # ── Chat (non-streaming, returns full response) ─────────────────────────

    async def chat(
        self,
        messages: List[dict],
        model: Optional[str] = None,
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

    async def embed(self, text: str) -> List[float]:
        """Get embedding vector for a piece of text."""
        payload = {
            "model": self.embed_model,
            "prompt": text,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{self.base_url}/api/embeddings", json=payload)
            resp.raise_for_status()
            return resp.json()["embedding"]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts (sequential — Ollama has no batch endpoint)."""
        embeddings = []
        for text in texts:
            emb = await self.embed(text)
            embeddings.append(emb)
        return embeddings


# Singleton
ollama_service = OllamaService()
