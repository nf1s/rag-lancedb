"""HTTP client for the LanceDB store service."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger("rag.api.store")


class StoreError(RuntimeError):
    """The store is unreachable or returned an error."""


class StoreResponseError(StoreError):
    """The store returned a 4xx response that should be forwarded verbatim."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


def _detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return response.text[:300]


class StoreClient:
    def __init__(self, base_url: str, timeout: float = 30.0, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.retries = max(1, retries)
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        token: str | None = None,
    ) -> Any:
        headers = {"X-Rag-Token": token} if token else None
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = await self._client.request(method, path, json=payload, headers=headers)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    await asyncio.sleep(0.25 * (2**attempt))
                continue

            if response.status_code >= 500:
                last_error = StoreError(f"{response.status_code}: {response.text[:300]}")
                if attempt + 1 < self.retries:
                    await asyncio.sleep(0.25 * (2**attempt))
                continue
            if response.status_code >= 400:
                raise StoreResponseError(response.status_code, _detail(response))
            return response.json()

        raise StoreError(f"store unavailable at {self.base_url}: {last_error}") from last_error

    async def health(self) -> dict[str, Any]:
        try:
            response = await self._client.get("/healthz", timeout=2.0)
            if response.content:
                return response.json()
            return {"status": "error", "error": f"HTTP {response.status_code}"}
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return {"status": "error", "error": str(exc)}

    # -- auth --------------------------------------------------------------
    async def login(self, username: str, password: str) -> dict[str, Any]:
        return await self._request(
            "POST", "/auth/login", {"username": username, "password": password}
        )

    async def logout(self, token: str) -> dict[str, Any]:
        return await self._request("POST", "/auth/logout", token=token)

    async def me(self, token: str) -> dict[str, Any]:
        return await self._request("GET", "/auth/me", token=token)

    # -- users -------------------------------------------------------------
    async def list_users(self, token: str) -> list[dict[str, Any]]:
        return await self._request("GET", "/users", token=token)

    async def create_user(
        self, token: str, username: str, password: str, global_role: str
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/users",
            {"username": username, "password": password, "global_role": global_role},
            token=token,
        )

    # -- workspaces --------------------------------------------------------
    async def list_workspaces(self, token: str) -> list[dict[str, Any]]:
        return await self._request("GET", "/workspaces", token=token)

    async def create_workspace(self, token: str, slug: str, name: str) -> dict[str, Any]:
        return await self._request("POST", "/workspaces", {"slug": slug, "name": name}, token=token)

    async def delete_workspace(self, token: str, workspace_id: str) -> dict[str, Any]:
        return await self._request("DELETE", f"/workspaces/{workspace_id}", token=token)

    async def list_members(self, token: str, workspace_id: str) -> list[dict[str, Any]]:
        return await self._request("GET", f"/workspaces/{workspace_id}/members", token=token)

    async def add_member(
        self, token: str, workspace_id: str, username: str, role: str
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/workspaces/{workspace_id}/members",
            {"username": username, "role": role},
            token=token,
        )

    async def remove_member(self, token: str, workspace_id: str, user_id: str) -> dict[str, Any]:
        return await self._request(
            "DELETE", f"/workspaces/{workspace_id}/members/{user_id}", token=token
        )

    # -- workspace data ----------------------------------------------------
    async def stats(self, token: str, workspace_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/ws/{workspace_id}/stats", token=token)

    async def upsert(
        self,
        token: str,
        workspace_id: str,
        source: str,
        rows: list[dict[str, Any]],
        rebuild_fts: bool = False,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/ws/{workspace_id}/upsert",
            {"source": source, "rows": rows, "rebuild_fts": rebuild_fts},
            token=token,
        )

    async def search(
        self,
        *,
        token: str,
        workspace_id: str,
        vector: list[float],
        text: str,
        mode: str,
        top_k: int,
        where: str | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/ws/{workspace_id}/search",
            {
                "vector": vector,
                "text": text,
                "mode": mode,
                "top_k": top_k,
                "where": where,
            },
            token=token,
        )

    async def delete(self, token: str, workspace_id: str, source: str) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/ws/{workspace_id}/delete",
            {"source": source},
            token=token,
        )

    async def reindex(self, token: str, workspace_id: str) -> dict[str, Any]:
        return await self._request("POST", f"/ws/{workspace_id}/reindex", token=token)
