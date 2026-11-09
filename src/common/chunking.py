"""Markdown-aware, token-budgeted text splitting.

The splitter is deliberately dependency free. Token counts are approximated from
whitespace-separated words (1 word ~ 1.3 tokens), which is accurate enough to keep
chunks below the 512-token limit of bge-small-en-v1.5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKENS_PER_WORD = 1.3
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    heading: str | None = None


def approx_tokens(text: str) -> int:
    words = len(text.split())
    return max(1, round(words * _TOKENS_PER_WORD))


def _words_for_tokens(tokens: int) -> int:
    return max(1, int(tokens / _TOKENS_PER_WORD))


def _sections(text: str):
    """Yield (heading, body) pairs, splitting on ATX markdown headings."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        yield None, text
        return
    if matches[0].start() > 0:
        yield None, text[: matches[0].start()]
    for position, match in enumerate(matches):
        start = match.end()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        yield match.group(2).strip(), text[start:end]


def _units(body: str, max_tokens: int):
    """Break a section into paragraph/sentence/word units no larger than max_tokens."""
    for paragraph in _PARAGRAPH_RE.split(body):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if approx_tokens(paragraph) <= max_tokens:
            yield paragraph
            continue
        for sentence in _SENTENCE_RE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if approx_tokens(sentence) <= max_tokens:
                yield sentence
                continue
            words = sentence.split()
            step = _words_for_tokens(max_tokens)
            for start in range(0, len(words), step):
                yield " ".join(words[start : start + step])


def split_text(
    text: str,
    target_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[TextChunk]:
    """Split markdown or plain text into overlapping chunks."""
    target_tokens = max(50, int(target_tokens))
    overlap_tokens = max(0, min(int(overlap_tokens), target_tokens - 10))

    chunks: list[TextChunk] = []
    index = 0

    for heading, body in _sections(text):
        current: list[str] = []
        current_tokens = 0

        for unit in _units(body, target_tokens):
            unit_tokens = approx_tokens(unit)
            if current and current_tokens + unit_tokens > target_tokens:
                chunks.append(TextChunk(index=index, text="\n\n".join(current), heading=heading))
                index += 1

                # Carry a tail of the previous chunk forward as overlap.
                keep: list[str] = []
                keep_tokens = 0
                for previous in reversed(current):
                    previous_tokens = approx_tokens(previous)
                    if keep_tokens + previous_tokens > overlap_tokens:
                        break
                    keep.insert(0, previous)
                    keep_tokens += previous_tokens

                # Never let the overlap push the next chunk over budget.
                while keep and keep_tokens + unit_tokens > target_tokens:
                    keep_tokens -= approx_tokens(keep.pop(0))
                current, current_tokens = keep, keep_tokens

            current.append(unit)
            current_tokens += unit_tokens

        if current:
            chunks.append(TextChunk(index=index, text="\n\n".join(current), heading=heading))
            index += 1

    return chunks
