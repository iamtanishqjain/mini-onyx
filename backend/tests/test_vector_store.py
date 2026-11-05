import asyncio
import json
import time

import pytest

from services.rag_service import SimpleVectorStore


def test_a_corrupt_collection_does_not_stop_the_store_loading(tmp_path, caplog):
    # A crash part-way through a write leaves a truncated file. Loading used to
    # raise, and because the store is built at import time the whole app died.
    (tmp_path / "good.json").write_text(json.dumps([{"id": "1", "document": "hi"}]))
    (tmp_path / "broken.json").write_text("{not json")

    store = SimpleVectorStore(str(tmp_path))

    assert "good" in store.collections
    assert "broken" not in store.collections


def test_the_readable_collections_are_still_listed(tmp_path):
    (tmp_path / "good.json").write_text(json.dumps([{"id": "1", "document": "hi"}]))
    (tmp_path / "broken.json").write_text("")

    store = SimpleVectorStore(str(tmp_path))

    assert [c["name"] for c in store.list_collections()] == ["good"]


def test_saving_leaves_no_temporary_file_behind(tmp_path):
    store = SimpleVectorStore(str(tmp_path))
    store.add(
        name="docs",
        ids=["1"],
        embeddings=[[0.1, 0.2]],
        documents=["hello"],
        metadatas=[{"source": "a.txt"}],
    )

    assert (tmp_path / "docs.json").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_saved_collection_reloads(tmp_path):
    store = SimpleVectorStore(str(tmp_path))
    store.add(
        name="docs",
        ids=["1"],
        embeddings=[[0.1, 0.2]],
        documents=["hello"],
        metadatas=[{"source": "a.txt"}],
    )

    reopened = SimpleVectorStore(str(tmp_path))

    assert len(reopened.collections["docs"]) == 1


def test_a_mismatched_embedding_length_is_an_error_not_a_wrong_score(tmp_path):
    # Changing the embedding model leaves old vectors a different length.
    # zip used to compare the shared prefix and return a plausible number.
    store = SimpleVectorStore(str(tmp_path))
    store.add(
        name="docs",
        ids=["1"],
        embeddings=[[1.0, 0.0, 0.0, 99.0, 99.0]],
        documents=["hello"],
        metadatas=[{"source": "a.txt"}],
    )

    with pytest.raises(ValueError, match="different embedding model"):
        store.query("docs", [1.0, 0.0, 0.0], top_k=1)


def test_adding_fewer_embeddings_than_chunks_is_an_error(tmp_path):
    store = SimpleVectorStore(str(tmp_path))

    with pytest.raises(ValueError):
        store.add(
            name="docs",
            ids=["1", "2", "3"],
            embeddings=[[0.1], [0.2]],
            documents=["a", "b", "c"],
            metadatas=[{}, {}, {}],
        )


def test_removing_a_source_leaves_the_other_documents(tmp_path):
    store = SimpleVectorStore(str(tmp_path))
    store.add(
        name="docs",
        ids=["1", "2"],
        embeddings=[[1.0], [2.0]],
        documents=["from a", "from b"],
        metadatas=[{"source": "a.pdf"}, {"source": "b.pdf"}],
    )

    removed = store.remove_source("docs", "a.pdf")

    assert removed == 1
    assert [c["document"] for c in store.collections["docs"]] == ["from b"]


def test_removing_an_unknown_source_changes_nothing(tmp_path):
    store = SimpleVectorStore(str(tmp_path))
    store.add(
        name="docs",
        ids=["1"],
        embeddings=[[1.0]],
        documents=["hello"],
        metadatas=[{"source": "a.pdf"}],
    )

    assert store.remove_source("docs", "nope.pdf") == 0
    assert len(store.collections["docs"]) == 1


def test_a_zero_query_vector_scores_everything_zero(tmp_path):
    store = SimpleVectorStore(str(tmp_path))
    # The query norm is computed once in query() now, so the zero guard that
    # used to sit per chunk has to still hold for the whole scoring pass.
    store.add(
        name="docs",
        ids=["1", "2"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["a", "b"],
        metadatas=[{"source": "a.txt"}, {"source": "b.txt"}],
    )

    results = store.query("docs", [0.0, 0.0], top_k=5)

    assert [score for _, score in results] == [0.0, 0.0]


def test_scores_are_unchanged_by_the_hoisted_norm(tmp_path):
    store = SimpleVectorStore(str(tmp_path))
    # cos between [3,4] (norm 5) and [1,0] is 3/5, and with [0,1] is 4/5.
    store.add(
        name="docs",
        ids=["1", "2"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["a", "b"],
        metadatas=[{"source": "a.txt"}, {"source": "b.txt"}],
    )

    scores = {item["document"]: score for item, score in store.query("docs", [3.0, 4.0], 5)}

    assert scores["a"] == pytest.approx(0.6)
    assert scores["b"] == pytest.approx(0.8)


def test_replacing_a_source_writes_the_collection_once(tmp_path, monkeypatch):
    # Removing then adding wrote the whole file twice per re-ingest.
    store = SimpleVectorStore(str(tmp_path))
    store.add(
        name="docs",
        ids=["1"],
        embeddings=[[1.0]],
        documents=["old"],
        metadatas=[{"source": "a.txt"}],
    )

    saves = []
    original = SimpleVectorStore._save
    monkeypatch.setattr(
        SimpleVectorStore,
        "_save",
        lambda self, name: (saves.append(name), original(self, name))[1],
    )

    store.replace_source(
        name="docs",
        source="a.txt",
        ids=["2"],
        embeddings=[[1.0]],
        documents=["new"],
        metadatas=[{"source": "a.txt"}],
    )

    assert saves == ["docs"]
    assert [c["document"] for c in store.collections["docs"]] == ["new"]


def test_replacing_a_source_leaves_other_sources_alone(tmp_path):
    store = SimpleVectorStore(str(tmp_path))
    store.add(
        name="docs",
        ids=["1", "2"],
        embeddings=[[1.0], [1.0]],
        documents=["from a", "from b"],
        metadatas=[{"source": "a.txt"}, {"source": "b.txt"}],
    )

    store.replace_source(
        name="docs",
        source="a.txt",
        ids=["3"],
        embeddings=[[1.0]],
        documents=["new a"],
        metadatas=[{"source": "a.txt"}],
    )

    assert sorted(c["document"] for c in store.collections["docs"]) == ["from b", "new a"]


def test_a_short_embeddings_list_stores_nothing(tmp_path):
    # zip(strict=True) raises only after yielding the common prefix, so the
    # chunks were appended one by one and the first few stayed behind.
    store = SimpleVectorStore(str(tmp_path))

    with pytest.raises(ValueError):
        store.add(
            name="docs",
            ids=["1", "2", "3"],
            embeddings=[[1.0]],
            documents=["a", "b", "c"],
            metadatas=[{"source": "a.txt"}] * 3,
        )

    assert store.collections.get("docs", []) == []


def test_a_value_that_cannot_be_saved_leaves_no_temporary_file(tmp_path):
    # Streaming into the file wrote part of the collection before hitting the
    # bad value, leaving a .tmp file holding a truncated collection.
    store = SimpleVectorStore(str(tmp_path))

    with pytest.raises(TypeError):
        store.add(
            name="docs",
            ids=["1"],
            embeddings=[object()],
            documents=["hello"],
            metadatas=[{"source": "a.txt"}],
        )

    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.asyncio
async def test_saving_a_collection_does_not_stall_the_event_loop(tmp_path, monkeypatch):
    # Writing a large collection is seconds of work. On the loop thread it
    # stalls every other request for the whole write.
    from app.core import config
    from services.rag_service import RAGService

    monkeypatch.setattr(config.settings, "chroma_persist_dir", str(tmp_path))
    service = RAGService()

    async def embed(text):
        return [1.0]

    monkeypatch.setattr("services.rag_service.ollama_service.embed", embed)
    monkeypatch.setattr(SimpleVectorStore, "_save", lambda self, name: time.sleep(0.3))

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    await service.ingest_document("report.txt", b"hello there", "docs")
    task.cancel()

    assert ticks > 5, "the event loop was blocked for the whole save"


@pytest.mark.asyncio
async def test_deleting_a_collection_does_not_stall_the_event_loop(tmp_path, monkeypatch):
    # delete takes the same lock the writes do, so running it on the loop
    # thread stalled every request until a save in progress finished.
    from app.core import config
    from services.rag_service import RAGService

    monkeypatch.setattr(config.settings, "chroma_persist_dir", str(tmp_path))
    service = RAGService()

    async def embed(text):
        return [1.0]

    monkeypatch.setattr("services.rag_service.ollama_service.embed", embed)
    service.store.add(
        name="other",
        ids=["1"],
        embeddings=[[1.0]],
        documents=["x"],
        metadatas=[{"source": "x.txt"}],
    )
    monkeypatch.setattr(SimpleVectorStore, "_save", lambda self, name: time.sleep(0.3))

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    ingest = asyncio.create_task(service.ingest_document("a.txt", b"hello", "docs"))
    await asyncio.sleep(0.05)
    await service.delete_collection("other")
    await ingest
    task.cancel()

    assert ticks > 5, "the event loop was blocked waiting on the write lock"
