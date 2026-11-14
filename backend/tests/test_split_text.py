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


def test_a_paragraph_break_near_the_window_start_does_not_win():
    # The search took the first separator type it found anywhere in the
    # window, so the "\n\n" after a short heading beat the sentence breaks
    # hundreds of characters later and left a fragment behind.
    doc = "## Heading\n\n" + "Body sentence with content. " * 25

    chunks = split_text(doc, chunk_size=500, overlap=50)

    assert min(len(c) for c in chunks) > 60, [len(c) for c in chunks]


def test_a_document_of_headings_is_not_shattered():
    doc = "".join("## Heading\n\n" + "Body sentence with content. " * 25 + "\n\n" for _ in range(6))

    chunks = split_text(doc, chunk_size=500, overlap=50)

    # 18 before, against a floor of len(doc)//500 == 8.
    assert len(chunks) <= 13, f"{len(chunks)} chunks for {len(doc)} characters"
    assert not [c for c in chunks if len(c) < 60]


def test_a_sentence_boundary_is_still_preferred_over_a_hard_cut():
    doc = "word " * 60 + "End of sentence. " + "more words here " * 40

    chunks = split_text(doc, chunk_size=500, overlap=50)

    assert chunks[0].endswith("End of sentence.")


def test_text_with_no_usable_separator_is_cut_at_the_window():
    # One separator right at the start and nothing else: a hard cut keeps the
    # chunk usable rather than emitting three characters.
    doc = "Hi\n" + "z" * 1200

    chunks = split_text(doc, chunk_size=500, overlap=50)

    assert max(len(c) for c in chunks) >= 400
