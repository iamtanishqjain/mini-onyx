import pytest
from fastapi.testclient import TestClient

from app.main import app
from services.ollama_service import ollama_service

client = TestClient(app)


@pytest.fixture
def ollama_down(monkeypatch):
    async def unavailable():
        return False

    monkeypatch.setattr(ollama_service, "is_available", unavailable)


@pytest.fixture
def ollama_up(monkeypatch):
    async def available():
        return True

    monkeypatch.setattr(ollama_service, "is_available", available)


def test_root_points_at_the_docs():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health_is_ok_when_ollama_answers(ollama_up):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ollama_connected"] is True


def test_health_is_degraded_rather_than_failing_when_ollama_is_down(ollama_down):
    # The backend is still serving, so /health must answer 200 and report the
    # dependency separately, or a readiness probe would restart a healthy pod.
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["ollama_connected"] is False


def test_health_reports_the_configured_models(ollama_up):
    body = client.get("/health").json()

    assert body["chat_model"]
    assert body["embed_model"]
    assert body["ollama_url"].startswith("http")


def test_unknown_route_is_a_404():
    assert client.get("/does-not-exist").status_code == 404
