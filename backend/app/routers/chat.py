from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json

from schemas.schemas import ChatRequest, ChatResponse
from services.ollama_service import ollama_service
from services.rag_service import rag_service
from app.core.config import settings

router = APIRouter(prefix="/chat", tags=["chat"])

RAG_SYSTEM_PROMPT = """You are a helpful AI assistant. You have been given relevant context below to help answer the user's question. Use this context to provide accurate, grounded answers. Always cite which source you are referencing.

CONTEXT:
{context}

---
If the context doesn't contain enough information to answer, say so clearly and answer from your general knowledge."""

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise, accurate, and friendly."


async def prepare_messages(request: ChatRequest):
    """Build the messages sent to the model, plus the RAG sources used.

    A system message supplied by the caller is kept, unless RAG found context,
    in which case the grounding prompt has to take its place.
    """
    messages = [{"role": m.role.value, "content": m.content} for m in request.messages]
    caller_system = messages[0]["content"] if messages and messages[0]["role"] == "system" else None

    context, source_chunks = "", []
    if request.use_rag and request.collection_name:
        user_query = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        if user_query:
            context, source_chunks = await rag_service.build_context(
                query_text=user_query,
                collection_name=request.collection_name,
                top_k=settings.rag_top_k,
            )

    system_content = (
        RAG_SYSTEM_PROMPT.format(context=context)
        if context
        else caller_system or DEFAULT_SYSTEM_PROMPT
    )

    if caller_system is not None:
        messages[0]["content"] = system_content
    else:
        messages.insert(0, {"role": "system", "content": system_content})

    return messages, source_chunks


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint.
    Set use_rag=true and provide collection_name to enable RAG.
    Returns Server-Sent Events (text/event-stream).
    """
    messages, source_chunks = await prepare_messages(request)

    # ── Stream ────────────────────────────────────────────────────────────
    async def event_generator():
        # First, send the source chunks so the UI can show citations
        if source_chunks:
            sources_payload = json.dumps({
                "type": "sources",
                "sources": [
                    {
                        "id": c.id,
                        "content": c.content[:200] + "..." if len(c.content) > 200 else c.content,
                        "source": c.metadata.get("source", "unknown"),
                        "score": c.score,
                    }
                    for c in source_chunks
                ],
            })
            yield f"data: {sources_payload}\n\n"

        # Then stream the LLM response token by token
        try:
            async for chunk in ollama_service.chat_stream(
                messages=messages,
                model=request.model,
            ):
                payload = json.dumps({"type": "token", "content": chunk})
                yield f"data: {payload}\n\n"
        except Exception as e:
            error_payload = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_payload}\n\n"

        # Signal done
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Non-streaming chat (returns full response at once).
    Good for testing via Swagger UI.
    """
    messages, _ = await prepare_messages(request)

    try:
        content = await ollama_service.chat(messages=messages, model=request.model)
        return ChatResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")
