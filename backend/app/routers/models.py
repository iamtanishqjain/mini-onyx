from fastapi import APIRouter, HTTPException
from app.models.schemas import ModelsResponse, ModelInfo
from app.services.ollama_service import ollama_service

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/", response_model=ModelsResponse)
async def list_models():
    """List all models available in Ollama."""
    try:
        raw_models = await ollama_service.list_models()
        models = [
            ModelInfo(
                name=m.get("name", "unknown"),
                size=str(round(m.get("size", 0) / 1e9, 1)) + " GB" if m.get("size") else None,
                modified_at=m.get("modified_at"),
            )
            for m in raw_models
        ]
        return ModelsResponse(models=models)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not reach Ollama: {str(e)}")
