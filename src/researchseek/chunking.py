"""Deterministic paragraph-aware token windows with source positions."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import re
from typing import Protocol

CHUNKER_VERSION = "paragraph-windows-v1"


class Tokenizer(Protocol):
    identity: str
    max_tokens: int
    special_tokens: int

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """Return ordered non-special token offsets into text."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens, including model special tokens, without truncation."""
        ...


class WhitespaceTokenizer:
    """Offline stub, not an approximation of a transformer tokenizer."""

    identity = "whitespace-v1"
    max_tokens = 256
    special_tokens = 2

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]

    def count_tokens(self, text: str) -> int:
        return len(self.token_spans(text)) + self.special_tokens


@dataclass(frozen=True)
class ChunkConfig:
    max_tokens: int = 200
    overlap: int = 32


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    text: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int


def chunk_text(text: str, tokenizer: Tokenizer, config: ChunkConfig = ChunkConfig()) -> list[Chunk]:
    """Split LF-normalized text into bounded, overlapping source slices."""
    values = (config.max_tokens, config.overlap, tokenizer.max_tokens, tokenizer.special_tokens)
    if any(type(value) is not int for value in values) or tokenizer.special_tokens < 0:
        raise ValueError("Token limits and overlap must be integers; special-token count cannot be negative")
    limit = min(config.max_tokens, tokenizer.max_tokens)
    capacity = limit - tokenizer.special_tokens
    if capacity < 1 or not 0 <= config.overlap < capacity:
        raise ValueError("Token limit must leave content space; overlap must be below that capacity")
    if "\r" in text:
        raise ValueError("Chunking requires LF-normalized text")
    if not text.strip():
        return []
    spans = tokenizer.token_spans(text)
    if not spans or any(not 0 <= start < end <= len(text) for start, end in spans):
        raise ValueError("Tokenizer returned invalid or missing source offsets")
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("Tokenizer offsets must be ordered and non-overlapping")
    starts = [start for start, _ in spans]
    boundaries = sorted({bisect_left(starts, match.end())
                         for match in re.finditer(r"\n[ \t]*\n", text)})
    newlines = [index for index, character in enumerate(text) if character == "\n"]
    chunks = []
    start = 0
    while start < len(spans):
        end = min(start + capacity, len(spans))
        if end < len(spans):
            index = bisect_right(boundaries, end) - 1
            if index >= 0 and boundaries[index] > start + config.overlap:
                end = boundaries[index]
        start_char = 0 if start == 0 else spans[start][0]
        while end > start:
            end_char = len(text) if end == len(spans) else spans[end][0]
            passage = text[start_char:end_char]
            if tokenizer.count_tokens(passage) <= limit:
                break
            end -= 1
        if end == start:
            raise ValueError("A source token and its whitespace exceed the configured token limit")
        chunks.append(Chunk(len(chunks), passage, start_char, end_char,
                            bisect_left(newlines, start_char) + 1,
                            bisect_left(newlines, end_char - 1) + 1))
        if end == len(spans):
            break
        start = max(start + 1, end - config.overlap)
    return chunks
