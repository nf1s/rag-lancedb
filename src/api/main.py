"""FastAPI backend: ingestion, retrieval and streamed answers.

Authentication and workspace scoping are enforced by the store. The API forwards
the caller's bearer token as ``X-Rag-Token`` and maps store errors back to the
client unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.embeddings import Embedder
from api.llm import check_llm, stream_chat
from api.prompt import build_messages
from api.store_client import StoreClient, StoreError, StoreResponseError
from common.chunking import TextChunk, split_text
from common.config import Settings, get_settings
from common.models import (
    AddMemberRequest,
    Chunk,
    CreateUserRequest,
    CreateWorkspaceRequest,
    IngestResponse,
    IngestResult,
    LoginRequest,
    LoginResponse,
    MemberInfo,
    MeResponse,
    OkResponse,
    QueryRequest,
    SearchHit,
    SearchQuery,
    SearchResponse,
    StoreStats,
    UserInfo,
    WorkspaceInfo,
)

logger = logging.getLogger("rag.api")

ALLOWED_SUFFIXES = {".md", ".markdown", ".txt"}
LLMStream = Callable[[list[dict[str, str]]], AsyncIterator[str]]


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _chunk_id(workspace_id: str, source: str, chunk: TextChunk) -> str:
    payload = f"{workspace_id}\0{source}\0{chunk.index}\0{chunk.text}".encode()
    return hashlib.sha256(payload).hexdigest()


def _rows_for(
    workspace_id: str,
    source: str,
    chunks: list[TextChunk],
    vectors: list[list[float]],
) -> list[Chunk]:
    rows: list[Chunk] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        rows.append(
            Chunk(
                id=_chunk_id(workspace_id, source, chunk),
                source=source,
                chunk_index=chunk.index,
                text=chunk.text,
                vector=vector,
                metadata={"heading": chunk.heading} if chunk.heading else {},
            )
        )
    return rows


def create_app(
    settings: Settings | None = None,
    embedder: Any = None,
    store: Any = None,
    llm_stream: LLMStream | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    embedder = embedder or Embedder(settings.embedding_model, settings.embedding_dim)
    store = store or StoreClient(
        settings.store_url, settings.store_timeout_seconds, settings.store_retries
    )
    stream_fn: LLMStream = llm_stream or (lambda messages: stream_chat(settings, messages))

    llm_cache: dict[str, Any] = {"checked": 0.0, "payload": {"reachable": None}}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        close = getattr(store, "close", None)
        if close is not None:
            await close()

    app = FastAPI(title="rag api", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.embedder = embedder
    app.state.store = store

    bearer = HTTPBearer(auto_error=False)

    def token_of(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> str:
        if credentials is None or not credentials.credentials:
            raise HTTPException(
                status_code=401,
                detail="missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return credentials.credentials

    @app.exception_handler(StoreResponseError)
    async def _store_response_error(_: Request, exc: StoreResponseError):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(StoreError)
    async def _store_error(_: Request, exc: StoreError):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    async def llm_health() -> dict[str, Any]:
        if time.monotonic() - llm_cache["checked"] < 30:
            return llm_cache["payload"]
        payload = await check_llm(settings)
        llm_cache.update(checked=time.monotonic(), payload=payload)
        return payload

    async def embed_query(question: str) -> list[float]:
        return await asyncio.to_thread(embedder.embed_query, question)

    @app.get("/")
    async def root():
        return {
            "service": "rag-api",
            "docs": "/docs",
            "health": "/healthz",
            "auth": "/auth/login",
        }

    @app.get("/healthz")
    async def healthz():
        store_health = await store.health()
        store_ok = store_health.get("status") == "ok"
        body = {
            "status": "ok" if store_ok else "degraded",
            "store": store_health,
            "llm": await llm_health(),
        }
        return JSONResponse(body, status_code=200 if store_ok else 503)

    # -- auth proxy --------------------------------------------------------
    @app.post("/auth/login", response_model=LoginResponse)
    async def login(request: LoginRequest):
        return await store.login(request.username, request.password)

    @app.post("/auth/logout", response_model=OkResponse)
    async def logout(token: str = Depends(token_of)):
        return await store.logout(token)

    @app.get("/auth/me", response_model=MeResponse)
    async def me(token: str = Depends(token_of)):
        return await store.me(token)

    # -- admin proxy -------------------------------------------------------
    @app.get("/users", response_model=list[UserInfo])
    async def list_users(token: str = Depends(token_of)):
        return await store.list_users(token)

    @app.post("/users", response_model=UserInfo, status_code=201)
    async def create_user(request: CreateUserRequest, token: str = Depends(token_of)):
        return await store.create_user(
            token, request.username, request.password, request.global_role
        )

    @app.get("/workspaces", response_model=list[WorkspaceInfo])
    async def list_workspaces(token: str = Depends(token_of)):
        return await store.list_workspaces(token)

    @app.post("/workspaces", response_model=WorkspaceInfo, status_code=201)
    async def create_workspace(request: CreateWorkspaceRequest, token: str = Depends(token_of)):
        return await store.create_workspace(token, request.slug, request.name)

    @app.delete("/workspaces/{workspace_id}", response_model=OkResponse)
    async def delete_workspace(workspace_id: str, token: str = Depends(token_of)):
        return await store.delete_workspace(token, workspace_id)

    @app.get("/workspaces/{workspace_id}/members", response_model=list[MemberInfo])
    async def list_members(workspace_id: str, token: str = Depends(token_of)):
        return await store.list_members(token, workspace_id)

    @app.post(
        "/workspaces/{workspace_id}/members",
        response_model=MemberInfo,
        status_code=201,
    )
    async def add_member(
        workspace_id: str,
        request: AddMemberRequest,
        token: str = Depends(token_of),
    ):
        return await store.add_member(token, workspace_id, request.username, request.role)

    @app.delete("/workspaces/{workspace_id}/members/{user_id}", response_model=OkResponse)
    async def remove_member(workspace_id: str, user_id: str, token: str = Depends(token_of)):
        return await store.remove_member(token, workspace_id, user_id)

    # -- workspace data ----------------------------------------------------
    @app.get("/ws/{workspace_id}/stats", response_model=StoreStats)
    async def stats(workspace_id: str, token: str = Depends(token_of)):
        return await store.stats(token, workspace_id)

    @app.post("/ws/{workspace_id}/reindex")
    async def reindex(workspace_id: str, token: str = Depends(token_of)):
        return await store.reindex(token, workspace_id)

    @app.post("/ws/{workspace_id}/search", response_model=SearchResponse)
    async def search(
        workspace_id: str,
        request: SearchQuery,
        token: str = Depends(token_of),
    ):
        vector = await embed_query(request.question)
        payload = await store.search(
            token=token,
            workspace_id=workspace_id,
            vector=vector,
            text=request.question,
            mode=request.mode or settings.retrieval_mode,
            top_k=request.top_k or settings.top_k,
        )
        return SearchResponse.model_validate(payload)

    @app.post("/ws/{workspace_id}/ingest", response_model=IngestResponse)
    async def ingest(
        workspace_id: str,
        files: list[UploadFile] = File(...),
        token: str = Depends(token_of),
    ):
        results: list[IngestResult] = []
        prepared: list[tuple[str, list[Chunk]]] = []

        for upload in files:
            name = Path(upload.filename or "upload.txt").name
            suffix = Path(name).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                results.append(
                    IngestResult(
                        source=name,
                        chunks=0,
                        warning=f"unsupported type '{suffix}'; only .md and .txt are indexed",
                    )
                )
                continue

            raw = await upload.read()
            text = raw.decode("utf-8", errors="replace")
            chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
            if not chunks:
                results.append(IngestResult(source=name, chunks=0, warning="no extractable text"))
                continue

            vectors: list[list[float]] = []
            batch_size = max(1, settings.embed_batch_size)
            for start in range(0, len(chunks), batch_size):
                batch = [chunk.text for chunk in chunks[start : start + batch_size]]
                vectors.extend(await asyncio.to_thread(embedder.embed_documents, batch))

            prepared.append((name, _rows_for(workspace_id, name, chunks, vectors)))

        # Rebuild the FTS index once, as part of the last write. Ingest therefore
        # needs only the "write" permission; manual reindexing stays admin-only.
        for position, (name, rows) in enumerate(prepared):
            await store.upsert(
                token,
                workspace_id,
                source=name,
                rows=[row.model_dump() for row in rows],
                rebuild_fts=position == len(prepared) - 1,
            )
            results.append(IngestResult(source=name, chunks=len(rows)))

        return IngestResponse(results=results)

    @app.post("/ws/{workspace_id}/query")
    async def query(
        workspace_id: str,
        request: QueryRequest,
        token: str = Depends(token_of),
    ):
        vector = await embed_query(request.question)
        payload = await store.search(
            token=token,
            workspace_id=workspace_id,
            vector=vector,
            text=request.question,
            mode=request.mode or settings.retrieval_mode,
            top_k=request.top_k or settings.top_k,
        )
        hits = [SearchHit.model_validate(hit) for hit in payload.get("hits", [])]

        async def event_stream() -> AsyncIterator[str]:
            try:
                yield _sse("sources", {"sources": [hit.model_dump() for hit in hits]})
                if not hits:
                    message = "I could not find anything relevant in the indexed documents."
                    yield _sse("token", {"text": message})
                    yield _sse("done", {"answer": message})
                    return

                messages = build_messages(request.question, hits, settings.max_context_tokens)
                collected: list[str] = []
                async for token in stream_fn(messages):
                    collected.append(token)
                    yield _sse("token", {"text": token})
                yield _sse("done", {"answer": "".join(collected)})
            except Exception as exc:  # noqa: BLE001 - report through the stream
                logger.exception("query failed")
                yield _sse("error", {"message": str(exc)})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
