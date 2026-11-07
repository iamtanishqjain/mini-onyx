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


@pytest.mark.asyncio
async def test_a_collection_can_be_deleted_by_the_name_it_was_created_with(service):
    # Ingest stores it sanitized. Delete used to receive the raw name and miss.
    service.store.add(
        name=sanitize_collection_name("my docs"),
        ids=["1"],
        embeddings=[[0.1, 0.2]],
        documents=["hello"],
        metadatas=[{"source": "a.txt"}],
    )

    assert await service.delete_collection("my docs") is True


def test_two_long_names_do_not_share_a_collection():
    # Both are 78 characters and differ only in the last two, which the old
    # 63-character truncation threw away, so they became one collection.
    a = "research-notes-" + "x" * 60 + "-v1"
    b = "research-notes-" + "x" * 60 + "-v2"

    assert sanitize_collection_name(a) != sanitize_collection_name(b)


def test_a_long_name_maps_to_the_same_collection_every_time():
    long_name = "project-" + "y" * 80

    assert sanitize_collection_name(long_name) == sanitize_collection_name(long_name)
    assert len(sanitize_collection_name(long_name)) == 63


def test_a_name_that_fits_is_left_alone():
    assert sanitize_collection_name("my docs") == "my_docs"
    assert sanitize_collection_name("a" * 63) == "a" * 63
