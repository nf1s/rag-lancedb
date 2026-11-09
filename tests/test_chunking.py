from common.chunking import approx_tokens, split_text


def test_empty_text_yields_no_chunks():
    assert split_text("") == []
    assert split_text("   \n\n  ") == []


def test_short_text_is_one_chunk():
    chunks = split_text("Hello world, this is short.", 400, 50)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world, this is short."
    assert chunks[0].heading is None


def test_tracks_markdown_headings():
    text = "# Title\n\nIntro paragraph.\n\n## Section\n\nSection body text."
    chunks = split_text(text, 400, 50)
    assert [chunk.heading for chunk in chunks] == ["Title", "Section"]
    assert "Intro paragraph." in chunks[0].text
    assert "Section body text." in chunks[1].text


def test_respects_token_budget():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
    chunks = split_text(text, 100, 20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert approx_tokens(chunk.text) <= 100


def test_chunks_are_overlapping_and_ordered():
    text = "\n\n".join("sentence number " + str(i) + "." for i in range(60))
    chunks = split_text(text, 40, 15)
    assert len(chunks) > 1
    joined = " ".join(chunk.text for chunk in chunks)
    for i in range(60):
        assert f"sentence number {i}." in joined
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_huge_single_paragraph_is_split():
    text = "word " * 2000
    chunks = split_text(text, 100, 10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert approx_tokens(chunk.text) <= 100
