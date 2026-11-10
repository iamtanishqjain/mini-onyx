from fastapi import APIRouter

from app.core.config import settings
from schemas.schemas import HealthResponse
from services.ollama_service import ollama_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the backend and Ollama are reachable."""
    ollama_ok, reason = await ollama_service.probe()
    return HealthResponse(
        status="ok" if ollama_ok else "degraded",
        ollama_connected=ollama_ok,
        ollama_error=reason,
        ollama_url=settings.ollama_base_url,
        chat_model=settings.ollama_chat_model,
        embed_model=settings.ollama_embed_model,
        version=settings.app_version,
    )
