import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import models as models_route

client = TestClient(app)


def listing(models):
    async def list_models():
        return models

    return list_models


def test_models_are_listed_with_a_human_readable_size(monkeypatch):
    monkeypatch.setattr(
        models_route.ollama_service,
        "list_models",
        listing([{"name": "llama3", "size": 4_700_000_000, "modified_at": "2026-01-01"}]),
    )

    body = client.get("/models/").json()

    assert body["models"][0]["name"] == "llama3"
    assert body["models"][0]["size"] == "4.7 GB"


def test_a_model_without_a_size_is_still_listed(monkeypatch):
    monkeypatch.setattr(
        models_route.ollama_service,
        "list_models",
        listing([{"name": "phi3"}]),
    )

    body = client.get("/models/").json()

    assert body["models"][0]["name"] == "phi3"
    assert body["models"][0]["size"] is None


def test_an_unnamed_model_does_not_break_the_listing(monkeypatch):
    monkeypatch.setattr(models_route.ollama_service, "list_models", listing([{}]))

    body = client.get("/models/").json()

    assert body["models"][0]["name"] == "unknown"


def test_an_unreachable_ollama_is_a_503(monkeypatch):
    async def boom():
        raise ConnectionError("connection refused")

    monkeypatch.setattr(models_route.ollama_service, "list_models", boom)

    response = client.get("/models/")

    assert response.status_code == 503
    assert "Could not reach Ollama" in response.json()["detail"]
