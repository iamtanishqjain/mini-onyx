import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import rag as rag_routes

client = TestClient(app)


@pytest.fixture
def empty_store(monkeypatch):
    monkeypatch.setattr(rag_routes.rag_service, "list_collections", lambda: [])


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
