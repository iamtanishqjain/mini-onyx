import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_a_chat_with_no_messages_is_rejected():
    # Without this the request reaches the model as a system prompt with no
    # user turn, which costs an inference call and returns nothing useful.
    response = client.post("/chat/", json={"messages": []})

    assert response.status_code == 422


def test_an_unknown_role_is_rejected():
    response = client.post(
        "/chat/",
        json={"messages": [{"role": "wizard", "content": "hi"}]},
    )

    assert response.status_code == 422


def test_a_message_without_content_is_rejected():
    response = client.post("/chat/", json={"messages": [{"role": "user"}]})

    assert response.status_code == 422


def sse_events(body: str):
    return [line[len("data: "):] for line in body.splitlines() if line.startswith("data: ")]


def test_the_stream_emits_tokens_then_done(monkeypatch):
    from app.routers import chat

    async def chat_stream(messages, model=None):
        for chunk in ["Hel", "lo"]:
            yield chunk

    monkeypatch.setattr(chat.ollama_service, "chat_stream", chat_stream)

    response = client.post(
        "/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert response.status_code == 200
    events = [json.loads(e) for e in sse_events(response.text)]
    assert [e["content"] for e in events if e["type"] == "token"] == ["Hel", "lo"]
    assert events[-1]["type"] == "done"


def test_a_backend_failure_becomes_an_error_event_not_a_dropped_stream(monkeypatch):
    from app.routers import chat

    async def chat_stream(messages, model=None):
        raise RuntimeError("ollama exploded")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(chat.ollama_service, "chat_stream", chat_stream)

    response = client.post(
        "/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    # The response has already begun, so the failure has to arrive in-band.
    assert response.status_code == 200
    events = [json.loads(e) for e in sse_events(response.text)]
    assert any(e["type"] == "error" and "ollama exploded" in e["message"] for e in events)
    assert events[-1]["type"] == "done"


def test_the_stream_is_not_buffered_by_proxies(monkeypatch):
    from app.routers import chat

    async def chat_stream(messages, model=None):
        yield "hi"

    monkeypatch.setattr(chat.ollama_service, "chat_stream", chat_stream)

    response = client.post(
        "/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
