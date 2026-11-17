import pytest

from researchseek.chunking import ChunkConfig, WhitespaceTokenizer, chunk_text


@pytest.fixture
def tokenizer():
    return WhitespaceTokenizer()


@pytest.mark.parametrize("text", ["", " ", "\n\n\t  \n"])
def test_empty_text(text, tokenizer):
    assert chunk_text(text, tokenizer) == []


@pytest.mark.parametrize("config", [ChunkConfig(2, 0), ChunkConfig(5, 3), ChunkConfig(10, -1),
                                   ChunkConfig(0, 0), ChunkConfig(5.5, 1), ChunkConfig(10, True)])
def test_invalid_configuration(config, tokenizer):
    with pytest.raises(ValueError):
        chunk_text("text", tokenizer, config)


def test_requires_normalized_text(tokenizer):
    with pytest.raises(ValueError, match="LF-normalized"):
        chunk_text("one\r\ntwo", tokenizer)


def test_exact_limit_and_no_redundant_tail(tokenizer):
    text = "one two three four"
    config = ChunkConfig(6, 2)
    assert [chunk.text for chunk in chunk_text(text, tokenizer, config)] == [text]
    chunks = chunk_text(text + " five", tokenizer, config)
    assert [chunk.text.split() for chunk in chunks] == [
        ["one", "two", "three", "four"], ["three", "four", "five"]]


def test_paragraph_preference(tokenizer):
    chunks = chunk_text("one two\n\nthree four five six", tokenizer, ChunkConfig(6, 0))
    assert [chunk.text for chunk in chunks] == ["one two\n\n", "three four five six"]


def test_small_paragraphs_are_combined(tokenizer):
    text = "one\n\ntwo\n\nthree"
    assert [chunk.text for chunk in chunk_text(text, tokenizer, ChunkConfig(6, 0))] == [text]


def test_overlap(tokenizer):
    text = " ".join(str(number) for number in range(11))
    chunks = chunk_text(text, tokenizer, ChunkConfig(7, 2))
    assert [chunk.text.split() for chunk in chunks] == [
        ["0", "1", "2", "3", "4"], ["3", "4", "5", "6", "7"], ["6", "7", "8", "9", "10"]]


def test_long_passage_determinism_positions_and_coverage(tokenizer):
    text = "  " + "\n".join(f"α{number}  β{number}\t" for number in range(80)) + "\n\n"
    config = ChunkConfig(13, 3)
    chunks = chunk_text(text, tokenizer, config)
    assert chunks == chunk_text(text, tokenizer, config)
    assert len(chunks) > 1
    covered = set()
    for ordinal, chunk in enumerate(chunks):
        assert chunk.ordinal == ordinal
        assert chunk.text == text[chunk.start_char:chunk.end_char]
        assert tokenizer.count_tokens(chunk.text) <= config.max_tokens
        assert chunk.start_line == text[:chunk.start_char].count("\n") + 1
        assert chunk.end_line == text[:chunk.end_char - 1].count("\n") + 1
        covered.update(range(chunk.start_char, chunk.end_char))
    assert covered == set(range(len(text)))
    assert len({(chunk.start_char, chunk.end_char) for chunk in chunks}) == len(chunks)


def test_no_overlap_preserves_every_character(tokenizer):
    text = "\n  indented code\t\n\n  next paragraph has many words here\n\n"
    chunks = chunk_text(text, tokenizer, ChunkConfig(6, 0))
    assert "".join(chunk.text for chunk in chunks) == text


def test_line_spans(tokenizer):
    chunks = chunk_text("α β\n\nγ δ", tokenizer, ChunkConfig(4, 0))
    assert [(c.start_char, c.end_char, c.start_line, c.end_line) for c in chunks] == [
        (0, 5, 1, 2), (5, 8, 3, 3)]


def test_effective_model_limit(tokenizer):
    tokenizer.max_tokens = 6
    chunks = chunk_text("one two three four five six", tokenizer, ChunkConfig(20, 0))
    assert len(chunks) == 2
    assert all(tokenizer.count_tokens(chunk.text) <= 6 for chunk in chunks)


def test_candidate_is_recounted_with_special_tokens(tokenizer):
    original_count = tokenizer.count_tokens
    tokenizer.count_tokens = lambda text: original_count(text) + 1
    chunks = chunk_text("one two three four five", tokenizer, ChunkConfig(6, 0))
    assert [chunk.text.split() for chunk in chunks] == [["one", "two", "three"], ["four", "five"]]


def test_unrepresentable_token(tokenizer):
    tokenizer.count_tokens = lambda text: 1000
    with pytest.raises(ValueError, match="exceed"):
        chunk_text("word", tokenizer)


def test_negative_special_token_count(tokenizer):
    tokenizer.special_tokens = -1
    with pytest.raises(ValueError, match="negative"):
        chunk_text("word", tokenizer)


@pytest.mark.parametrize("spans", [[], [(1, 0)], [(0, 100)], [(1, 3), (0, 1)], [(0, 2), (1, 3)]])
def test_invalid_token_offsets(tokenizer, spans):
    tokenizer.token_spans = lambda text: spans
    with pytest.raises(ValueError, match="offsets"):
        chunk_text("abc", tokenizer)
