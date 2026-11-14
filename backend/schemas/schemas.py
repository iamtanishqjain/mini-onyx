from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """A request body that refuses fields it does not know.

    Pydantic drops unknown keys by default, so a misspelled collection_name
    was accepted and ignored, and the answer came back ungrounded with
    nothing to say the field had not been read.
    """

    model_config = ConfigDict(extra="forbid")


# ── Chat ────────────────────────────────────────────────────────────────────

class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Message(StrictModel):
    role: MessageRole
    content: str


class ChatRequest(StrictModel):
    # A chat with no messages would still reach the model, costing an
    # inference call for a prompt with no user turn in it.
    messages: list[Message] = Field(min_length=1)
    model: str | None = None          # overrides default if provided
    use_rag: bool = False                 # attach retrieved context
    collection_name: str | None = None  # which RAG collection to query

    @model_validator(mode="after")
    def _has_a_question_to_answer(self):
        # A blank user turn passed the length check and reached the model
        # anyway, and with use_rag it skipped retrieval silently too, the
        # empty string being falsy. Earlier turns may be blank: only the
        # request as a whole has to carry something to answer.
        if not any(m.role is MessageRole.user and m.content.strip() for m in self.messages):
            raise ValueError("at least one user message must have content")
        return self

    @model_validator(mode="after")
    def _rag_needs_a_collection(self):
        # Asking for RAG without naming a collection fell through to an
        # ungrounded answer carrying no sources, which reads exactly like a
        # grounded one that found nothing. A collection that does not exist
        # is already a 404; not naming one at all should not be quieter.
        if self.use_rag and not (self.collection_name or "").strip():
            raise ValueError("use_rag needs a collection_name")
        return self


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


class QueryRequest(StrictModel):
    # Embedding whitespace costs a model call and can only return noise.
    query: str = Field(min_length=1, pattern=r"\S")
    collection_name: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    query: str
    results: list[DocumentChunk]


class CollectionInfo(BaseModel):
    name: str
    document_count: int   # distinct source files
    chunk_count: int      # pieces they were split into


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
    ollama_error: str | None = None   # why not, when not connected
    ollama_url: str
    chat_model: str
    embed_model: str
    version: str
