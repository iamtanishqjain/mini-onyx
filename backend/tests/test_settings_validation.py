import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_a_zero_chunk_size_is_rejected():
    # split_text stops advancing at 0, so ingestion hangs instead of failing.
    with pytest.raises(ValidationError):
        Settings(chunk_size=0)


def test_a_negative_overlap_is_rejected():
    with pytest.raises(ValidationError):
        Settings(chunk_overlap=-1)


def test_an_overlap_as_big_as_the_chunk_is_rejected():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(chunk_size=100, chunk_overlap=100)


def test_a_zero_upload_limit_is_rejected():
    with pytest.raises(ValidationError):
        Settings(max_upload_mb=0)


def test_a_zero_top_k_is_rejected():
    with pytest.raises(ValidationError):
        Settings(rag_top_k=0)


def test_the_defaults_are_valid():
    s = Settings()

    assert s.chunk_size > s.chunk_overlap >= 0
    assert s.max_upload_mb >= 1
