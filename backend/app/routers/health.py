from fastapi import APIRouter
from schemas.schemas import HealthResponse
from services.ollama_service import ollama_service
from app.core.config import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the backend and Ollama are reachable."""
    ollama_ok = await ollama_service.is_available()
    return HealthResponse(
        status="ok" if ollama_ok else "degraded",
        ollama_connected=ollama_ok,
        ollama_url=settings.ollama_base_url,
        chat_model=settings.ollama_chat_model,
        embed_model=settings.ollama_embed_model,
        version=settings.app_version,
    )
