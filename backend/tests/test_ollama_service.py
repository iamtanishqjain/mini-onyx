import httpx
import pytest

from app.core import config
from services.ollama_service import ollama_service


@pytest.fixture(autouse=True)
def restore_settings():
    original = config.settings.ollama_base_url
    yield
    config.settings.ollama_base_url = original


def test_the_service_follows_a_changed_base_url():
    # The instance is built at import time, so copying settings in __init__
    # meant a later change never reached it.
    config.settings.ollama_base_url = "http://ollama:11434"

    assert ollama_service.base_url == "http://ollama:11434"


def test_the_service_follows_changed_model_names():
    config.settings.ollama_chat_model = "mistral"
    config.settings.ollama_embed_model = "all-minilm"

    assert ollama_service.chat_model == "mistral"
    assert ollama_service.embed_model == "all-minilm"


@pytest.mark.asyncio
async def test_the_health_probe_gives_up_inside_the_healthcheck_window(monkeypatch):
    # A real socket, because MockTransport ignores timeouts entirely and would
    # pass whatever the probe does. The container healthcheck allows 5s.
    import socket
    import threading
    import time

    from app.core import config
    from services.ollama_service import ollama_service

    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(5)
    held = []

    def accept_and_hold():
        while True:
            try:
                held.append(listener.accept()[0])
            except OSError:
                return

    threading.Thread(target=accept_and_hold, daemon=True).start()
    monkeypatch.setattr(
        config.settings, "ollama_base_url", f"http://127.0.0.1:{listener.getsockname()[1]}"
    )

    try:
        started = time.perf_counter()
        connected, _ = await ollama_service.probe()
        assert connected is False
        took = time.perf_counter() - started
    finally:
        listener.close()
        for conn in held:
            conn.close()

    assert took < 4, f"the probe took {took:.2f}s, outliving the 5s healthcheck"


@pytest.mark.asyncio
async def test_the_probe_names_the_transport_failure(monkeypatch):
    # A bare false read the same for a wrong URL, a name that does not
    # resolve, a refused connection and an Ollama answering an error. Faked
    # rather than aimed at a dead port, which refuses on one platform and
    # hangs until the timeout on another.
    def refuse(request):
        raise httpx.ConnectError("[Errno 111] Connection refused")

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **k: original(*a, **{**k, "transport": httpx.MockTransport(refuse)}),
    )
    monkeypatch.setattr(config.settings, "ollama_base_url", "http://ollama:11434")

    connected, reason = await ollama_service.probe()

    assert connected is False
    assert "ConnectError" in reason
    assert "http://ollama:11434" in reason


@pytest.mark.asyncio
async def test_the_probe_reports_an_error_status_from_ollama(monkeypatch):
    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **k: original(*a, **{**k, "transport": transport})
    )

    connected, reason = await ollama_service.probe()

    assert connected is False
    assert "answered 503" in reason


@pytest.mark.asyncio
async def test_the_probe_reports_no_reason_when_ollama_answers(monkeypatch):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"models": []}))
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **k: original(*a, **{**k, "transport": transport})
    )

    assert await ollama_service.probe() == (True, None)
