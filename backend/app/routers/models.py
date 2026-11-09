import httpx
from fastapi import APIRouter, HTTPException

from schemas.schemas import ModelInfo, ModelsResponse
from services.ollama_service import ollama_service

router = APIRouter(prefix="/models", tags=["models"])


def _size_in_gb(size) -> str | None:
    """Ollama reports a size in bytes, when it reports one at all."""
    if not isinstance(size, int | float) or isinstance(size, bool) or size <= 0:
        return None
    return f"{round(size / 1e9, 1)} GB"


def _model_info(entry: dict) -> ModelInfo:
    modified = entry.get("modified_at")
    return ModelInfo(
        name=str(entry.get("name") or "unknown"),
        size=_size_in_gb(entry.get("size")),
        modified_at=modified if isinstance(modified, str) else None,
    )


@router.get("/", response_model=ModelsResponse)
async def list_models():
    """List all models available in Ollama."""
    # Reading the reply used to sit in the same try as the call, so a listing
    # this code could not parse was reported as "Could not reach Ollama" and
    # sent people to check their networking over a field holding the wrong
    # type. Only a transport failure means unreachable.
    try:
        raw_models = await ollama_service.list_models()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama answered {e.response.status_code} when asked for its models.",
        ) from e
    except (httpx.RequestError, OSError) as e:
        raise HTTPException(status_code=503, detail=f"Could not reach Ollama: {e}") from e
    except Exception as e:
        # Reached it, could not make sense of the reply: a body that is not
        # JSON, or a shape this code does not expect.
        raise HTTPException(
            status_code=502, detail=f"Could not read Ollama's model list: {e}"
        ) from e

    return ModelsResponse(models=[_model_info(m) for m in raw_models])
