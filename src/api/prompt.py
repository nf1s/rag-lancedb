"""Prompt assembly with a token budget."""

from __future__ import annotations

from common.models import SearchHit

_TOKENS_PER_WORD = 1.3

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the user's documents. "
    "Use only the context provided below. Cite sources with bracketed numbers such as [1]. "
    "If the context does not contain the answer, say that you could not find it."
)


def trim_to_tokens(text: str, max_tokens: int) -> str:
    words = text.split()
    limit = max(1, int(max_tokens / _TOKENS_PER_WORD))
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]) + " …"


def build_messages(
    question: str,
    hits: list[SearchHit],
    max_context_tokens: int,
) -> list[dict[str, str]]:
    question_tokens = int(len(question.split()) * _TOKENS_PER_WORD)
    budget = max(200, max_context_tokens - question_tokens - 100)

    blocks: list[str] = []
    for position, hit in enumerate(hits, start=1):
        label = hit.source
        heading = hit.metadata.get("heading")
        if heading:
            label = f"{label} > {heading}"
        blocks.append(f"[{position}] ({label})\n{hit.text}")

    context = trim_to_tokens("\n\n".join(blocks), budget)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
