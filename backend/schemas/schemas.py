from enum import Enum

from pydantic import BaseModel, Field

# ── Chat ────────────────────────────────────────────────────────────────────

class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Message(BaseModel):
    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    # A chat with no messages would still reach the model, costing an
    # inference call for a prompt with no user turn in it.
    messages: list[Message] = Field(min_length=1)
    model: str | None = None          # overrides default if provided
    use_rag: bool = False                 # attach retrieved context
    collection_name: str | None = None  # which RAG collection to query
    stream: bool = True


class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str


# ── RAG ─────────────────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    id: str
    content: str
    metadata: dict
    score: float | None = None


class IngestResponse(BaseModel):
    collection_name: str
    chunks_added: int
    filename: str
    message: str


class QueryRequest(BaseModel):
    # Embedding whitespace costs a model call and can only return noise.
    query: str = Field(min_length=1, pattern=r"\S")
    collection_name: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    query: str
    results: list[DocumentChunk]


class CollectionInfo(BaseModel):
    name: str
    document_count: int


class CollectionsResponse(BaseModel):
    collections: list[CollectionInfo]


# ── Models ───────────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    name: str
    size: str | None = None
    modified_at: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    ollama_url: str
    chat_model: str
    embed_model: str
    version: str
