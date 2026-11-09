import httpx
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


def test_a_model_whose_size_is_not_a_number_is_still_listed(monkeypatch):
    # One odd entry used to fail the whole request, reported as if Ollama
    # could not be reached.
    monkeypatch.setattr(
        models_route.ollama_service,
        "list_models",
        listing([{"name": "llama3", "size": "4.7GB"}]),
    )

    body = client.get("/models/").json()

    assert body["models"][0]["name"] == "llama3"
    assert body["models"][0]["size"] is None


def test_a_non_string_timestamp_is_dropped_rather_than_failing(monkeypatch):
    monkeypatch.setattr(
        models_route.ollama_service,
        "list_models",
        listing([{"name": "llama3", "modified_at": {"not": "a string"}}]),
    )

    body = client.get("/models/").json()

    assert body["models"][0]["modified_at"] is None


def test_an_error_status_from_ollama_is_not_reported_as_unreachable(monkeypatch):
    async def answered_500():
        request = httpx.Request("GET", "http://ollama:11434/api/tags")
        raise httpx.HTTPStatusError(
            "500", request=request, response=httpx.Response(500, request=request)
        )

    monkeypatch.setattr(models_route.ollama_service, "list_models", answered_500)

    response = client.get("/models/")

    assert response.status_code == 502
    assert "answered 500" in response.json()["detail"]


def test_an_unreadable_reply_is_not_reported_as_unreachable(monkeypatch):
    async def gibberish():
        raise ValueError("Expecting value: line 1 column 1")

    monkeypatch.setattr(models_route.ollama_service, "list_models", gibberish)

    response = client.get("/models/")

    assert response.status_code == 502
    assert "Could not read" in response.json()["detail"]


@pytest.mark.asyncio
async def test_a_null_model_list_is_treated_as_empty(monkeypatch):
    # Ollama answering {"models": null} made the route iterate None.
    import httpx as httpx_module

    from services.ollama_service import ollama_service

    transport = httpx_module.MockTransport(
        lambda request: httpx_module.Response(200, json={"models": None})
    )
    original = httpx_module.AsyncClient
    monkeypatch.setattr(
        httpx_module,
        "AsyncClient",
        lambda *a, **k: original(*a, **{**k, "transport": transport}),
    )

    assert await ollama_service.list_models() == []
