import json
import logging
import math
import os
import uuid
from pathlib import Path

from app.core.config import settings
from schemas.schemas import DocumentChunk
from services.ollama_service import ollama_service

logger = logging.getLogger(__name__)


class CollectionNotFoundError(LookupError):
    """Raised for a collection name nothing has ever been ingested into."""


class EmbeddingMismatchError(ValueError):
    """Raised when a collection and the current embedding model disagree."""


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
                with open(f) as fp:
                    self.collections[f.stem] = json.load(fp)
            except (json.JSONDecodeError, OSError) as e:
                # The store is built at import time, so raising here stops the
                # whole app from starting. One damaged collection should cost
                # that collection, not the service.
                logger.warning("Skipping unreadable collection %s: %s", f.name, e)

    def _save(self, name: str):
        # Write beside the target and rename, so an interrupted save leaves the
        # previous collection intact rather than a half-written file. os.replace
        # is atomic on both POSIX and Windows.
        path = self._collection_path(name)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as fp:
            json.dump(self.collections[name], fp)
        os.replace(tmp, path)

    def get_or_create(self, name: str):
        if name not in self.collections:
            self.collections[name] = []
        return self.collections[name]

    def add(self, name: str, ids, embeddings, documents, metadatas):
        col = self.get_or_create(name)
        # strict, so a short embeddings list raises instead of zip quietly
        # truncating and storing fewer chunks than the document produced.
        for id_, emb, doc, meta in zip(ids, embeddings, documents, metadatas, strict=True):
            col.append({"id": id_, "embedding": emb, "document": doc, "metadata": meta})
        self._save(name)

    def remove_source(self, name: str, source: str) -> int:
        """Drop every chunk that came from one source file. Returns how many."""
        col = self.collections.get(name)
        if not col:
            return 0
        kept = [c for c in col if c.get("metadata", {}).get("source") != source]
        removed = len(col) - len(kept)
        if removed:
            self.collections[name] = kept
            self._save(name)
        return removed

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        # Vectors of different lengths mean the collection was built with a
        # different embedding model. zip would compare the shared prefix and
        # return a plausible-looking but meaningless score.
        if len(a) != len(b):
            raise EmbeddingMismatchError(
                f"embedding length {len(b)} does not match the query's {len(a)}; "
                "the collection was built with a different embedding model"
            )
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def query(self, name: str, query_embedding: list[float], top_k: int):
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

def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
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
    import io

    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n\n".join(p.extract_text() or "" for p in reader.pages)


def parse_docx(file_bytes: bytes) -> str:
    import io

    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def parse_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")


def parse_file(filename: str, file_bytes: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in (".pdf", ".docx", ".txt", ".md"):
        raise ValueError(f"Unsupported file type: {ext}")

    parser = {".pdf": parse_pdf, ".docx": parse_docx}.get(ext, parse_txt)
    try:
        return parser(file_bytes)
    except Exception as e:
        # A file the reader cannot open is a bad upload, not a server fault.
        # Raised as ValueError so the route answers 422 rather than 500.
        raise ValueError(f"Could not read '{filename}': {e}") from e


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

    async def ingest_document(self, filename, file_bytes, collection_name) -> tuple[str, int]:
        collection_name = sanitize_collection_name(collection_name)
        raw_text = parse_file(filename, file_bytes)
        if not raw_text.strip():
            raise ValueError("Could not extract text from document.")
        chunks = split_text(raw_text, settings.chunk_size, settings.chunk_overlap)
        embeddings = [await ollama_service.embed(c) for c in chunks]
        # Re-uploading a file replaces its chunks. Appending them left the old
        # copy in place, so retrieval returned the same passage twice and spent
        # the context window saying it.
        self.store.remove_source(collection_name, filename)
        self.store.add(
            name=collection_name,
            ids=[str(uuid.uuid4()) for _ in chunks],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"source": filename, "chunk_index": i} for i in range(len(chunks))],
        )
        return collection_name, len(chunks)

    async def query(self, query_text, collection_name, top_k=5) -> list[DocumentChunk]:
        collection_name = sanitize_collection_name(collection_name)
        # A name that was never ingested into used to return no results, which
        # the caller could not tell apart from a collection that simply had
        # nothing to match. Checked before embedding, so a typo costs no call.
        if collection_name not in self.store.collections:
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")
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

    async def build_context(
        self, query_text, collection_name, top_k=5
    ) -> tuple[str, list[DocumentChunk]]:
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
