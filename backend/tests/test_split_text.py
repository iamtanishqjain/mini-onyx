from services.rag_service import split_text


def test_no_text_is_dropped_when_a_separator_sits_near_the_window_start():
    # "Hi. " puts a separator at index 2 and nothing after it, so the first
    # window ends at 4. Stepping back by the overlap used to move `start` to
    # -46, and the negative slice skipped everything in between.
    text = "Hi. " + "x" * 1000

    chunks = split_text(text, chunk_size=500, overlap=50)

    # Chunks overlap by design, so the count can exceed 1000 but must never
    # fall short of it. Before the fix only 650 of the 1000 survived.
    assert "".join(chunks).count("x") >= 1000


def test_chunks_cover_the_whole_document():
    text = "Sentence one. Sentence two. " * 60

    chunks = split_text(text, chunk_size=500, overlap=50)

    assert sum(len(c) for c in chunks) >= len(text.strip())


def test_text_without_separators_is_split_by_size():
    text = "y" * 1200

    chunks = split_text(text, chunk_size=500, overlap=50)

    assert len(chunks) > 1
    assert "".join(chunks).count("y") >= 1200


def test_short_text_returns_one_chunk():
    assert split_text("just a short note", chunk_size=500) == ["just a short note"]


def test_empty_text_returns_no_chunks():
    assert split_text("", chunk_size=500) == []
