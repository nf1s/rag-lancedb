"""OpenAI-compatible chat client with SSE streaming and a non-streaming fallback."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from common.config import Settings

logger = logging.getLogger("rag.api.llm")


class LLMError(RuntimeError):
    pass


class _StreamUnsupported(Exception):
    pass


async def stream_chat(settings: Settings, messages: list[dict[str, str]]) -> AsyncIterator[str]:
    """Yield answer tokens from an OpenAI-compatible /chat/completions endpoint."""
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": 0.2,
        "stream": True,
    }
    timeout = httpx.Timeout(settings.llm_timeout_seconds, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async for token in _stream(client, url, headers, payload):
                yield token
            return
        except _StreamUnsupported as exc:
            logger.warning("LLM does not support streaming (%s); retrying without it", exc)

        async for token in _complete(client, url, headers, payload):
            yield token


async def _stream(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> AsyncIterator[str]:
    async with client.stream("POST", url, json=payload, headers=headers) as response:
        if response.status_code >= 400:
            body = (await response.aread()).decode(errors="replace")
            if response.status_code == 400 and "stream" in body.lower():
                raise _StreamUnsupported(body[:300])
            raise LLMError(f"LLM returned {response.status_code}: {body[:300]}")

        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("delta") or {}).get("content")
            if content:
                yield content


async def _complete(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> AsyncIterator[str]:
    body = dict(payload)
    body["stream"] = False
    response = await client.post(url, json=body, headers=headers)
    if response.status_code >= 400:
        raise LLMError(f"LLM returned {response.status_code}: {response.text[:300]}")
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        return
    content = (choices[0].get("message") or {}).get("content")
    if content:
        yield content


async def check_llm(settings: Settings) -> dict[str, Any]:
    """Probe /models. Never raises; some servers do not implement it."""
    url = settings.llm_base_url.rstrip("/") + "/models"
    headers = {}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.llm_health_timeout_seconds, connect=2.0)
        ) as client:
            response = await client.get(url, headers=headers)
        if response.status_code >= 400:
            return {"reachable": False, "error": f"HTTP {response.status_code}"}
        data = response.json()
        models = [item.get("id") for item in data.get("data", []) if isinstance(item, dict)]
        return {"reachable": True, "models": models, "configured": settings.llm_model}
    except Exception as exc:  # noqa: BLE001
        return {"reachable": False, "error": str(exc)}
