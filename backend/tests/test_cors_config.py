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
