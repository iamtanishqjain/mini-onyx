import pytest

from services.rag_service import RAGService, sanitize_collection_name


@pytest.fixture
def service(tmp_path, monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "chroma_persist_dir", str(tmp_path))
    return RAGService()


def test_spaces_and_dots_are_replaced():
    assert sanitize_collection_name("my docs") == "my_docs"
    assert sanitize_collection_name("report.v2") == "report_v2"


def test_empty_name_falls_back_to_default():
    assert sanitize_collection_name("") == "default"


def test_name_is_capped():
    assert len(sanitize_collection_name("a" * 200)) == 63


def test_path_separators_cannot_escape_the_store():
    assert "/" not in sanitize_collection_name("../../etc/passwd")
    assert ".." not in sanitize_collection_name("../../etc/passwd")


def test_a_collection_can_be_deleted_by_the_name_it_was_created_with(service):
    # Ingest stores it sanitized. Delete used to receive the raw name and miss.
    service.store.add(
        name=sanitize_collection_name("my docs"),
        ids=["1"],
        embeddings=[[0.1, 0.2]],
        documents=["hello"],
        metadatas=[{"source": "a.txt"}],
    )

    assert service.delete_collection("my docs") is True
