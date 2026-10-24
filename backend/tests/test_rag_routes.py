import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import rag as rag_routes

client = TestClient(app)


@pytest.fixture
def empty_store(monkeypatch):
    monkeypatch.setattr(rag_routes.rag_service, "list_collections", lambda: [])


@pytest.fixture
def store(monkeypatch, tmp_path):
    """A real service on a temp dir, with a stub embedder of a chosen width."""
    from app.core import config
    from services.rag_service import RAGService

    monkeypatch.setattr(config.settings, "chroma_persist_dir", str(tmp_path))
    service = RAGService()
    monkeypatch.setattr(rag_routes, "rag_service", service)

    def install(dimensions=1):
        async def embed(text):
            return [float(len(text))] + [0.0] * (dimensions - 1)

        monkeypatch.setattr("services.rag_service.ollama_service.embed", embed)
        return service

    return install


def test_an_unsupported_extension_is_rejected():
    response = client.post(
        "/rag/ingest",
        files={"file": ("notes.exe", b"binary", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_a_file_with_no_extension_is_rejected():
    response = client.post(
        "/rag/ingest",
        files={"file": ("README", b"text", "text/plain")},
    )

    assert response.status_code == 400


def test_an_empty_file_is_rejected():
    response = client.post(
        "/rag/ingest",
        files={"file": ("notes.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_listing_collections_is_empty_rather_than_failing(empty_store):
    response = client.get("/rag/collections")

    assert response.status_code == 200
    assert response.json()["collections"] == []


def test_deleting_an_unknown_collection_is_a_404(monkeypatch):
    monkeypatch.setattr(rag_routes.rag_service, "delete_collection", lambda name: False)

    response = client.delete("/rag/collections/does-not-exist")

    assert response.status_code == 404


def test_an_empty_query_is_rejected():
    response = client.post("/rag/query", json={"query": "", "collection_name": "docs"})

    assert response.status_code == 422


def test_a_whitespace_only_query_is_rejected():
    # Embedding whitespace costs a model call and can only match noise.
    response = client.post("/rag/query", json={"query": "   ", "collection_name": "docs"})

    assert response.status_code == 422


def test_an_empty_collection_name_is_rejected():
    response = client.post("/rag/query", json={"query": "hello", "collection_name": ""})

    assert response.status_code == 422


def test_an_oversized_upload_is_rejected(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "max_upload_mb", 1)
    oversized = b"x" * (1024 * 1024 + 1)

    response = client.post(
        "/rag/ingest", files={"file": ("notes.txt", oversized, "text/plain")}
    )

    assert response.status_code == 413
    assert "1 MB limit" in response.json()["detail"]


def test_an_upload_at_the_limit_is_accepted(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "max_upload_mb", 1)

    async def ingest(filename, file_bytes, collection_name):
        return collection_name, 1

    monkeypatch.setattr(rag_routes.rag_service, "ingest_document", ingest)

    response = client.post(
        "/rag/ingest",
        files={"file": ("notes.txt", b"x" * (1024 * 1024), "text/plain")},
    )

    assert response.status_code == 200


def test_a_corrupt_pdf_is_a_422_not_a_500():
    # The reader failing on a bad upload is the user's problem to fix, and a
    # 500 would say the server broke while telling them nothing useful.
    response = client.post(
        "/rag/ingest",
        files={"file": ("broken.pdf", b"not a pdf at all", "application/pdf")},
    )

    assert response.status_code == 422
    assert "broken.pdf" in response.json()["detail"]


def test_a_corrupt_docx_is_a_422_not_a_500():
    response = client.post(
        "/rag/ingest",
        files={"file": ("broken.docx", b"not a docx", "application/octet-stream")},
    )

    assert response.status_code == 422


def test_reingesting_a_file_replaces_its_chunks_rather_than_duplicating(monkeypatch, tmp_path):
    from app.core import config
    from services.rag_service import RAGService

    monkeypatch.setattr(config.settings, "chroma_persist_dir", str(tmp_path))

    async def embed(text):
        return [float(len(text))]

    service = RAGService()
    monkeypatch.setattr("services.rag_service.ollama_service.embed", embed)

    import asyncio

    for _ in range(2):
        asyncio.run(
            service.ingest_document(
                filename="report.txt",
                file_bytes=b"chunk one. chunk two.",
                collection_name="docs",
            )
        )

    stored = service.store.collections["docs"]
    assert len({c["document"] for c in stored}) == len(stored), "chunks were duplicated"


def test_querying_an_unknown_collection_is_a_404(store):
    # It used to answer 200 with no results, which the caller could not tell
    # apart from a collection that had nothing to match.
    store()

    response = client.post(
        "/rag/query", json={"query": "hello", "collection_name": "typo"}
    )

    assert response.status_code == 404
    assert "typo" in response.json()["detail"]


def test_an_unknown_collection_is_not_embedded(store, monkeypatch):
    store()
    calls = []

    async def embed(text):
        calls.append(text)
        return [1.0]

    monkeypatch.setattr("services.rag_service.ollama_service.embed", embed)

    client.post("/rag/query", json={"query": "hello", "collection_name": "typo"})

    assert calls == []


def test_a_collection_from_another_embedding_model_is_a_422(store):
    # Scoring a 384-dim query against 768-dim chunks is not a server fault,
    # and the 500 it used to answer said nothing about what to change.
    service = store(dimensions=1)
    service.store.add(
        name="docs",
        ids=["1"],
        embeddings=[[1.0, 0.0]],
        documents=["hello"],
        metadatas=[{"source": "a.txt"}],
    )

    response = client.post(
        "/rag/query", json={"query": "hello", "collection_name": "docs"}
    )

    assert response.status_code == 422
    assert "different embedding model" in response.json()["detail"]


def test_a_corrupt_stored_chunk_is_a_500_not_a_422(store):
    # Pydantic's ValidationError is itself a ValueError, so catching that
    # broadly blamed the caller's request for the store's own bad data.
    service = store()
    service.store.add(
        name="docs",
        ids=["1"],
        embeddings=[[1.0]],
        documents=["hello"],
        metadatas=[None],
    )

    response = client.post(
        "/rag/query", json={"query": "hello", "collection_name": "docs"}
    )

    assert response.status_code == 500


def chunks(service, name, *vectors):
    service.store.add(
        name=name,
        ids=[str(i) for i in range(len(vectors))],
        embeddings=list(vectors),
        documents=[f"chunk {i}" for i in range(len(vectors))],
        metadatas=[{"source": "a.txt", "chunk_index": i} for i in range(len(vectors))],
    )


def test_a_query_returns_the_chunks_best_match_first(store, monkeypatch):
    service = store()
    chunks(service, "docs", [0.0, 1.0], [1.0, 0.0])

    async def embed(text):
        return [1.0, 0.0]

    monkeypatch.setattr("services.rag_service.ollama_service.embed", embed)

    body = client.post(
        "/rag/query", json={"query": "hello", "collection_name": "docs"}
    ).json()

    assert [c["content"] for c in body["results"]] == ["chunk 1", "chunk 0"]
    assert body["results"][0]["score"] == 1.0
    assert body["results"][0]["metadata"]["source"] == "a.txt"
    assert body["query"] == "hello"


def test_top_k_caps_the_results(store, monkeypatch):
    service = store()
    chunks(service, "docs", [1.0, 0.0], [0.9, 0.1], [0.0, 1.0])

    async def embed(text):
        return [1.0, 0.0]

    monkeypatch.setattr("services.rag_service.ollama_service.embed", embed)

    body = client.post(
        "/rag/query",
        json={"query": "hello", "collection_name": "docs", "top_k": 2},
    ).json()

    assert len(body["results"]) == 2


def test_a_collection_whose_chunks_were_all_removed_still_answers_200(store):
    # It exists, it just has nothing left to match, which is not the 404 that
    # a name nothing was ever ingested into gets.
    service = store()
    chunks(service, "docs", [1.0])
    service.store.remove_source("docs", "a.txt")

    response = client.post(
        "/rag/query", json={"query": "hello", "collection_name": "docs"}
    )

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_a_collection_can_be_queried_by_the_name_it_was_created_with(store):
    # Ingest stores it sanitized, so "my docs" is on disk as "my_docs".
    service = store()
    chunks(service, "my_docs", [1.0])

    response = client.post(
        "/rag/query", json={"query": "hello", "collection_name": "my docs"}
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == 1
