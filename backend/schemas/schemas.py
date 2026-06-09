from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


# ── Chat ────────────────────────────────────────────────────────────────────

class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Message(BaseModel):
    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None          # overrides default if provided
    use_rag: bool = False                 # attach retrieved context
    collection_name: Optional[str] = None  # which RAG collection to query
    stream: bool = True


class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str


# ── RAG ─────────────────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    id: str
    content: str
    metadata: dict
    score: Optional[float] = None


class IngestResponse(BaseModel):
    collection_name: str
    chunks_added: int
    filename: str
    message: str


class QueryRequest(BaseModel):
    query: str
    collection_name: str
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    query: str
    results: List[DocumentChunk]


class CollectionInfo(BaseModel):
    name: str
    document_count: int


class CollectionsResponse(BaseModel):
    collections: List[CollectionInfo]


# ── Models ───────────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    name: str
    size: Optional[str] = None
    modified_at: Optional[str] = None


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    ollama_url: str
    chat_model: str
    embed_model: str
    version: str
