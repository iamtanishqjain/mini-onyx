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
async def test_is_available_is_false_when_ollama_cannot_be_reached():
    # Port 1 is reserved and nothing listens there, so the connection fails.
    config.settings.ollama_base_url = "http://127.0.0.1:1"

    assert await ollama_service.is_available() is False


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
        assert await ollama_service.is_available() is False
        took = time.perf_counter() - started
    finally:
        listener.close()
        for conn in held:
            conn.close()

    assert took < 4, f"the probe took {took:.2f}s, outliving the 5s healthcheck"
