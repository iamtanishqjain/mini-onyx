import pytest

from app.core.config import Settings


def settings_with(value):
    return Settings(cors_origins=value)


def test_a_json_array_is_parsed():
    s = settings_with('["http://localhost:3000","http://127.0.0.1:3000"]')

    assert s.cors_origins_list == ["http://localhost:3000", "http://127.0.0.1:3000"]


def test_a_comma_separated_list_is_parsed():
    # The natural thing to write in a .env file, and it used to crash startup.
    s = settings_with("http://localhost:3000, http://127.0.0.1:3000")

    assert s.cors_origins_list == ["http://localhost:3000", "http://127.0.0.1:3000"]


def test_a_single_origin_is_parsed():
    assert settings_with("http://localhost:3000").cors_origins_list == [
        "http://localhost:3000"
    ]


def test_an_empty_value_means_no_origins():
    assert settings_with("").cors_origins_list == []
    assert settings_with("   ").cors_origins_list == []


def test_malformed_json_names_the_setting():
    s = settings_with('["http://localhost:3000"')

    with pytest.raises(ValueError, match="CORS_ORIGINS"):
        s.cors_origins_list


def build_app_with_origins(monkeypatch, value):
    """Rebuild the app, since the middleware is wired at import time."""
    import importlib

    import app.main
    from app.core import config

    monkeypatch.setattr(config.settings, "cors_origins", value)
    return importlib.reload(app.main).app


def test_a_wildcard_origin_does_not_also_allow_credentials(monkeypatch):
    # Starlette answers a wildcard by echoing the request's Origin. Paired
    # with credentials that let any site call this API with the browser's
    # cookies attached and read the reply.
    from fastapi.testclient import TestClient

    client = TestClient(build_app_with_origins(monkeypatch, '["*"]'))

    response = client.get("/health", headers={"Origin": "https://somewhere.example"})

    assert response.headers.get("access-control-allow-credentials") is None
    assert response.headers.get("access-control-allow-origin") != "https://somewhere.example"


def test_explicit_origins_still_allow_credentials(monkeypatch):
    from fastapi.testclient import TestClient

    client = TestClient(build_app_with_origins(monkeypatch, '["https://app.example"]'))

    response = client.get("/health", headers={"Origin": "https://app.example"})

    assert response.headers["access-control-allow-credentials"] == "true"
    assert response.headers["access-control-allow-origin"] == "https://app.example"


def test_an_unlisted_origin_gets_no_cors_headers(monkeypatch):
    from fastapi.testclient import TestClient

    client = TestClient(build_app_with_origins(monkeypatch, '["https://app.example"]'))

    response = client.get("/health", headers={"Origin": "https://evil.example"})

    assert response.headers.get("access-control-allow-origin") is None
