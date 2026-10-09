import json

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
