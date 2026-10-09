import uuid
import json
import logging
import math
from pathlib import Path
from typing import List, Tuple

from app.core.config import settings
from schemas.schemas import DocumentChunk
from services.ollama_service import ollama_service

logger = logging.getLogger(__name__)


# ── Simple vector store (pure Python, no deps) ───────────────────────────────

class SimpleVectorStore:
    """In-memory vector store with cosine similarity. Persists to JSON."""

    def __init__(self, persist_dir: str):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collections: dict = {}
        self._load_all()

    def _collection_path(self, name: str) -> Path:
        return self.persist_dir / f"{name}.json"

    def _load_all(self):
        for f in self.persist_dir.glob("*.json"):
            try:
                with open(f, "r") as fp:
                    self.collections[f.stem] = json.load(fp)
            except (json.JSONDecodeError, OSError) as e:
                # The store is built at import time, so raising here stops the
                # whole app from starting. One damaged collection should cost
                # that collection, not the service.
                logger.warning("Skipping unreadable collection %s: %s", f.name, e)

    def _save(self, name: str):
        with open(self._collection_path(name), "w") as fp:
            json.dump(self.collections[name], fp)

    def get_or_create(self, name: str):
        if name not in self.collections:
            self.collections[name] = []
        return self.collections[name]

    def add(self, name: str, ids, embeddings, documents, metadatas):
        col = self.get_or_create(name)
        for id_, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            col.append({"id": id_, "embedding": emb, "document": doc, "metadata": meta})
        self._save(name)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def query(self, name: str, query_embedding: List[float], top_k: int):
        col = self.collections.get(name, [])
        if not col:
            return []
        scored = [
            (item, self._cosine_similarity(query_embedding, item["embedding"]))
            for item in col
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def list_collections(self):
        return [
            {"name": name, "document_count": len(items)}
            for name, items in self.collections.items()
        ]

    def delete_collection(self, name: str) -> bool:
        if name in self.collections:
            del self.collections[name]
            path = self._collection_path(name)
            if path.exists():
                path.unlink()
            return True
        return False


# ── Text helpers ─────────────────────────────────────────────────────────────

def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            for sep in ["\n\n", ".\n", ". ", "\n"]:
                idx = text.rfind(sep, start, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # A separator close to the window start makes `end` small, so stepping
        # back by `overlap` would move `start` backwards (even negative) and skip
        # the text in between. Only overlap when it still moves forward.
        start = end - overlap if end - overlap > start else end
    return chunks


def parse_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n\n".join(p.extract_text() or "" for p in reader.pages)


def parse_docx(file_bytes: bytes) -> str:
    from docx import Document
    import io
    doc = Document(io.BytesIO(file_bytes))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def parse_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")


def parse_file(filename: str, file_bytes: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":       return parse_pdf(file_bytes)
    elif ext in (".docx",): return parse_docx(file_bytes)
    elif ext in (".txt", ".md"): return parse_txt(file_bytes)
    else: raise ValueError(f"Unsupported file type: {ext}")


# ── RAG Service ───────────────────────────────────────────────────────────────

def sanitize_collection_name(name: str) -> str:
    """Collection names become file names, so keep them to safe characters.

    Applied on every path into the store. Sanitizing only on ingest meant a
    collection stored as "my_docs" could never be queried or deleted by the
    "my docs" the caller originally sent.
    """
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return safe[:63] or "default"


class RAGService:
    def __init__(self):
        self.store = SimpleVectorStore(settings.chroma_persist_dir)

    async def ingest_document(self, filename, file_bytes, collection_name) -> Tuple[str, int]:
        collection_name = sanitize_collection_name(collection_name)
        raw_text = parse_file(filename, file_bytes)
        if not raw_text.strip():
            raise ValueError("Could not extract text from document.")
        chunks = split_text(raw_text, settings.chunk_size, settings.chunk_overlap)
        embeddings = [await ollama_service.embed(c) for c in chunks]
        self.store.add(
            name=collection_name,
            ids=[str(uuid.uuid4()) for _ in chunks],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"source": filename, "chunk_index": i} for i in range(len(chunks))],
        )
        return collection_name, len(chunks)

    async def query(self, query_text, collection_name, top_k=5) -> List[DocumentChunk]:
        collection_name = sanitize_collection_name(collection_name)
        query_emb = await ollama_service.embed(query_text)
        results = self.store.query(collection_name, query_emb, top_k)
        return [
            DocumentChunk(
                id=item["id"],
                content=item["document"],
                metadata=item["metadata"],
                score=round(score, 4),
            )
            for item, score in results
        ]

    async def build_context(self, query_text, collection_name, top_k=5) -> Tuple[str, List[DocumentChunk]]:
        chunks = await self.query(query_text, collection_name, top_k)
        if not chunks:
            return "", []
        parts = [f"[Source {i+1}: {c.metadata.get('source','unknown')}]\n{c.content}"
                 for i, c in enumerate(chunks)]
        return "\n\n---\n\n".join(parts), chunks

    def list_collections(self):
        return self.store.list_collections()

    def delete_collection(self, name):
        return self.store.delete_collection(sanitize_collection_name(name))


rag_service = RAGService()