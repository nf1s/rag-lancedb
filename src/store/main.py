"""LanceDB owner service.

This is the only process that opens the LanceDB directory. On a local filesystem
LanceDB has no commit arbitration, so the service runs a single uvicorn worker and
serializes every write behind an asyncio lock.

Each workspace has its own LanceDB table (and therefore its own FTS index). The
SQLite metadata database stores users, workspaces, memberships and sessions, and
is the source of truth for access control. Table names are generated server-side
and never derived from user input.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import lancedb
import pyarrow as pa
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from lancedb.index import FTS
from lancedb.rerankers import RRFReranker

from common.config import Settings, get_settings
from common.models import (
    AddMemberRequest,
    Chunk,
    CreateUserRequest,
    CreateWorkspaceRequest,
    DeleteRequest,
    LoginRequest,
    LoginResponse,
    MemberInfo,
    MeResponse,
    OkResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    StoreHealth,
    StoreStats,
    UpsertRequest,
    UserInfo,
    WorkspaceInfo,
)
from common.permissions import ROLE_RANK, Action, role_allows
from store.metadata import MetadataStore, User, Workspace

logger = logging.getLogger("rag.store")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _where_source(source: str) -> str:
    escaped = source.replace("'", "''")
    return f"source = '{escaped}'"


def _to_hit(row: dict[str, Any]) -> SearchHit:
    score = row.get("_relevance_score")
    if score is None:
        distance = row.get("_distance")
        score = 1.0 - float(distance) if distance is not None else 0.0

    metadata = row.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}

    return SearchHit(
        score=float(score),
        id=str(row.get("id", "")),
        source=str(row.get("source", "")),
        chunk_index=int(row.get("chunk_index", 0)),
        text=str(row.get("text", "")),
        metadata=metadata,
    )


def _user_info(user: User) -> UserInfo:
    return UserInfo(
        id=user.id,
        username=user.username,
        global_role=user.global_role,
        disabled=user.disabled,
        created_at=user.created_at,
    )


def _workspace_info(workspace: Workspace, role: str | None) -> WorkspaceInfo:
    return WorkspaceInfo(
        id=workspace.id,
        slug=workspace.slug,
        name=workspace.name,
        role=role,
        created_at=workspace.created_at,
    )


class LanceStore:
    """Synchronous wrapper around the LanceDB directory and its per-workspace tables."""

    def __init__(self, settings: Settings, metadata: MetadataStore):
        self.settings = settings
        self.metadata = metadata
        self.uri = settings.lancedb_path
        self.meta_path = self.uri / settings.meta_filename
        self.db: Any = None
        self.meta: dict[str, Any] = {}
        self._tables: dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        self.uri.mkdir(parents=True, exist_ok=True)
        self.meta = self._load_meta()
        self._validate_meta()
        self.db = lancedb.connect(str(self.uri))
        self._bootstrap()
        logger.info(
            "store ready at %s workspaces=%s",
            self.uri,
            self.metadata.workspace_count(),
        )

    def _bootstrap(self) -> None:
        admin = self.metadata.get_user_by_username(self.settings.bootstrap_admin_user)
        if admin is not None and admin[0].global_role == "admin":
            owner = admin[0]
        elif self.metadata.user_count() == 0:
            logger.info("creating bootstrap admin '%s'", self.settings.bootstrap_admin_user)
            owner = self.metadata.create_user(
                self.settings.bootstrap_admin_user,
                self.settings.bootstrap_admin_password,
                "admin",
            )
        else:
            admins = [user for user in self.metadata.list_users() if user.global_role == "admin"]
            if not admins:
                raise RuntimeError("no admin user exists and BOOTSTRAP_ADMIN_USER is not present")
            owner = admins[0]

        if self.metadata.workspace_count():
            return

        names = set(self.db.list_tables().tables)
        if self.settings.table_name in names:
            # Legacy single-table deployment: adopt the existing table as-is so
            # that documents survive the upgrade without copying data.
            table_name = self.settings.table_name
            fts_ready = bool(self.meta.get("fts_ready"))
            logger.info("reusing legacy table '%s' as the default workspace", table_name)
        else:
            table_name = self._new_table_name()
            self._create_table(table_name)
            fts_ready = False

        workspace = self.metadata.create_workspace(
            "default",
            "Default",
            table_name,
            owner.id,
            {"fts_ready": fts_ready},
        )
        self.metadata.add_member(workspace.id, owner.id, "owner")
        logger.info("created default workspace %s (table=%s)", workspace.id, table_name)

    def _schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("source", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.settings.embedding_dim)),
                pa.field("metadata", pa.string()),
            ]
        )

    def _new_table_name(self) -> str:
        return f"ws_{uuid.uuid4().hex}"

    def _create_table(self, table_name: str):
        if table_name in set(self.db.list_tables().tables):
            table = self.db.open_table(table_name)
        else:
            logger.info("creating table %s", table_name)
            table = self.db.create_table(table_name, schema=self._schema())
        self._tables[table_name] = table
        return table

    def _table(self, table_name: str):
        table = self._tables.get(table_name)
        if table is None:
            table = self.db.open_table(table_name)
            self._tables[table_name] = table
        return table

    def create_workspace_table(self, table_name: str) -> None:
        self._create_table(table_name)

    def drop_workspace_table(self, table_name: str) -> None:
        self._tables.pop(table_name, None)
        if table_name in set(self.db.list_tables().tables):
            self.db.drop_table(table_name)

    # -- metadata ----------------------------------------------------------
    def _load_meta(self) -> dict[str, Any]:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text())
        return {
            "model": self.settings.embedding_model,
            "dim": self.settings.embedding_dim,
            "fts_ready": False,
            "created_at": _now(),
            "updated_at": None,
        }

    def _validate_meta(self) -> None:
        model = self.meta.get("model")
        dim = self.meta.get("dim")
        if model != self.settings.embedding_model or dim != self.settings.embedding_dim:
            raise RuntimeError(
                f"embedding mismatch: index uses {model}/{dim}, configuration uses "
                f"{self.settings.embedding_model}/{self.settings.embedding_dim}. "
                "Re-ingest the corpus or restore the original model."
            )

    def _save_meta(self) -> None:
        self.meta["updated_at"] = _now()
        temporary = self.meta_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.meta, indent=2))
        temporary.replace(self.meta_path)

    # -- writes ------------------------------------------------------------
    def _batch(self, rows: list[Chunk]) -> pa.Table:
        dim = self.settings.embedding_dim
        for row in rows:
            if len(row.vector) != dim:
                raise ValueError(
                    f"vector for chunk {row.id} has {len(row.vector)} dims, expected {dim}"
                )
        return pa.Table.from_pydict(
            {
                "id": [row.id for row in rows],
                "source": [row.source for row in rows],
                "chunk_index": pa.array([row.chunk_index for row in rows], type=pa.int32()),
                "text": [row.text for row in rows],
                "vector": pa.array([row.vector for row in rows], type=pa.list_(pa.float32(), dim)),
                "metadata": [json.dumps(row.metadata) for row in rows],
            },
            schema=self._schema(),
        )

    def upsert_sync(
        self,
        table_name: str,
        source: str,
        rows: list[Chunk],
        rebuild_fts: bool = True,
    ) -> tuple[int, bool | None]:
        table = self._table(table_name)
        if not rows:
            raise ValueError("no rows to upsert")

        # replace-by-source: removes stale chunks when a document shrinks
        if table.count_rows():
            table.delete(_where_source(source))
        table.add(self._batch(rows))
        self._save_meta()

        if not rebuild_fts:
            return len(rows), None
        return len(rows), self.reindex_table_sync(table_name)

    def delete_sync(self, table_name: str, source: str) -> tuple[int, bool]:
        table = self._table(table_name)
        if table.count_rows() == 0:
            return 0, False
        before = table.count_rows()
        table.delete(_where_source(source))
        removed = before - table.count_rows()
        self._save_meta()
        return removed, self.reindex_table_sync(table_name)

    def reindex_table_sync(self, table_name: str) -> bool:
        table = self._table(table_name)
        if table.count_rows() == 0:
            self._save_meta()
            return False
        try:
            table.create_index("text", config=FTS(), replace=True)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            logger.warning("FTS index rebuild failed for %s: %s", table_name, exc)
            self._save_meta()
            raise
        self._save_meta()
        return True

    # -- reads -------------------------------------------------------------
    def search_sync(
        self,
        table_name: str,
        request: SearchRequest,
        fts_ready: bool,
    ) -> SearchResponse:
        table = self._table(table_name)

        started = time.perf_counter()
        top_k = max(1, int(request.top_k))
        mode = request.mode

        if mode == "hybrid" and not (request.vector and request.text and fts_ready):
            mode = "vector" if request.vector else "fts"
        if mode == "fts":
            if not request.text:
                raise ValueError("fts search requires text")
            if not fts_ready:
                if not request.vector:
                    raise ValueError("full-text index is not ready; reindex first")
                mode = "vector"
        if mode == "vector" and not request.vector:
            raise ValueError("vector search requires a vector")

        def run(active_mode: str):
            if active_mode == "hybrid":
                query = table.search(
                    query_type="hybrid",
                    vector_column_name="vector",
                    fts_columns="text",
                )
                query = query.vector(request.vector).text(request.text).rerank(RRFReranker())
            elif active_mode == "fts":
                query = table.search(request.text, query_type="fts")
            else:
                query = table.search(request.vector, vector_column_name="vector").metric("cosine")
            if request.where:
                query = query.where(request.where)
            return query.limit(top_k).to_list()

        try:
            rows = run(mode)
        except Exception as exc:  # noqa: BLE001 - fall back when the index is unusable
            if mode != "hybrid":
                raise
            logger.warning("hybrid search failed (%s); falling back to vector", exc)
            mode = "vector"
            rows = run(mode)

        return SearchResponse(
            hits=[_to_hit(row) for row in rows],
            mode=mode,
            took_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def stats_sync(self, table_name: str) -> StoreStats:
        table = self._table(table_name)
        rows = table.count_rows()
        sources = 0
        if rows:
            column = table.to_lance().to_table(columns=["source"]).column("source")
            sources = len(set(column.to_pylist()))
        return StoreStats(
            table=table_name,
            rows=rows,
            sources=sources,
            dim=self.settings.embedding_dim,
            model=self.settings.embedding_model,
            fts_ready=False,
            updated_at=self.meta.get("updated_at"),
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    metadata = MetadataStore(settings.auth_db_path)
    store = LanceStore(settings, metadata)
    lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await asyncio.to_thread(metadata.open)
            await asyncio.to_thread(store.open)
        except Exception as exc:  # keep serving so /healthz can report the problem
            logger.exception("store failed to open")
            app.state.startup_error = str(exc)
        yield
        await asyncio.to_thread(metadata.close)

    app = FastAPI(title="rag store", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.state.metadata = metadata
    app.state.startup_error = None

    # -- auth dependencies -------------------------------------------------
    def session_token(x_rag_token: str | None = Header(default=None)) -> str:
        if not x_rag_token:
            raise HTTPException(status_code=401, detail="missing X-Rag-Token header")
        return x_rag_token

    def require_user(token: str = Depends(session_token)) -> User:
        if app.state.startup_error:
            raise HTTPException(status_code=503, detail=app.state.startup_error)
        user = metadata.validate_session(token)
        if user is None:
            raise HTTPException(status_code=401, detail="invalid or expired session")
        return user

    def require_global_admin(user: User = Depends(require_user)) -> User:
        if user.global_role != "admin":
            raise HTTPException(status_code=403, detail="global admin required")
        return user

    def effective_role(workspace: Workspace, user: User) -> str | None:
        if user.global_role == "admin":
            return "owner"
        return metadata.get_membership(workspace.id, user.id)

    def require_access(workspace_id: str, user: User, action: Action) -> Workspace:
        if app.state.startup_error:
            raise HTTPException(status_code=503, detail=app.state.startup_error)
        workspace = metadata.get_workspace(workspace_id)
        if workspace is None:
            raise HTTPException(status_code=404, detail="workspace not found")
        role = effective_role(workspace, user)
        if not role_allows(role, action):
            raise HTTPException(
                status_code=403,
                detail=(f"role '{role or 'none'}' cannot perform '{action}' in this workspace"),
            )
        return workspace

    # -- health ------------------------------------------------------------
    @app.get("/")
    async def root():
        return {"service": "rag-store", "docs": "/docs"}

    @app.get("/healthz")
    async def healthz():
        if app.state.startup_error or store.db is None:
            return JSONResponse(
                status_code=503,
                content=StoreHealth(
                    status="error", error=app.state.startup_error or "store not ready"
                ).model_dump(),
            )
        try:
            workspaces = await asyncio.to_thread(metadata.workspace_count)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                status_code=503, content=StoreHealth(status="error", error=str(exc)).model_dump()
            )
        return StoreHealth(status="ok", workspaces=workspaces)

    # -- auth --------------------------------------------------------------
    def _login_sync(
        username: str, password: str
    ) -> tuple[str, User, list[tuple[Workspace, str]]] | None:
        user = metadata.authenticate(username, password)
        if user is None:
            return None
        token = metadata.create_session(user.id, settings.session_ttl_hours)
        return token, user, metadata.list_workspaces_for(user)

    @app.post("/auth/login", response_model=LoginResponse)
    async def login(request: LoginRequest):
        if app.state.startup_error:
            raise HTTPException(status_code=503, detail=app.state.startup_error)
        result = await asyncio.to_thread(_login_sync, request.username, request.password)
        if result is None:
            raise HTTPException(status_code=401, detail="invalid username or password")
        token, user, workspaces = result
        return LoginResponse(
            token=token,
            user=_user_info(user),
            workspaces=[_workspace_info(workspace, role) for workspace, role in workspaces],
        )

    @app.post("/auth/logout", response_model=OkResponse)
    async def logout(token: str = Depends(session_token)):
        await asyncio.to_thread(metadata.revoke_session, token)
        return OkResponse()

    @app.get("/auth/me", response_model=MeResponse)
    async def me(user: User = Depends(require_user)):
        workspaces = await asyncio.to_thread(metadata.list_workspaces_for, user)
        return MeResponse(
            user=_user_info(user),
            workspaces=[_workspace_info(workspace, role) for workspace, role in workspaces],
        )

    # -- users -------------------------------------------------------------
    @app.get("/users", response_model=list[UserInfo])
    async def list_users(_: User = Depends(require_global_admin)):
        users = await asyncio.to_thread(metadata.list_users)
        return [_user_info(user) for user in users]

    @app.post("/users", response_model=UserInfo, status_code=201)
    async def create_user(request: CreateUserRequest, _: User = Depends(require_global_admin)):
        try:
            created = await asyncio.to_thread(
                metadata.create_user,
                request.username,
                request.password,
                request.global_role,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _user_info(created)

    # -- workspaces --------------------------------------------------------
    @app.get("/workspaces", response_model=list[WorkspaceInfo])
    async def list_workspaces(user: User = Depends(require_user)):
        workspaces = await asyncio.to_thread(metadata.list_workspaces_for, user)
        return [_workspace_info(workspace, role) for workspace, role in workspaces]

    @app.post("/workspaces", response_model=WorkspaceInfo, status_code=201)
    async def create_workspace(
        request: CreateWorkspaceRequest, user: User = Depends(require_global_admin)
    ):
        existing = await asyncio.to_thread(metadata.get_workspace_by_slug, request.slug)
        if existing is not None:
            raise HTTPException(
                status_code=409, detail=f"workspace '{request.slug}' already exists"
            )
        table_name = f"ws_{uuid.uuid4().hex}"
        async with lock:
            try:
                await asyncio.to_thread(store.create_workspace_table, table_name)
                workspace = await asyncio.to_thread(
                    metadata.create_workspace,
                    request.slug,
                    request.name,
                    table_name,
                    user.id,
                    {},
                )
            except ValueError as exc:
                await asyncio.to_thread(store.drop_workspace_table, table_name)
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except Exception:
                await asyncio.to_thread(store.drop_workspace_table, table_name)
                raise
            await asyncio.to_thread(metadata.add_member, workspace.id, user.id, "owner")
        return _workspace_info(workspace, "owner")

    @app.delete("/workspaces/{workspace_id}", response_model=OkResponse)
    async def delete_workspace(workspace_id: str, user: User = Depends(require_user)):
        workspace = require_access(workspace_id, user, Action.DELETE_WORKSPACE)
        async with lock:
            await asyncio.to_thread(store.drop_workspace_table, workspace.table_name)
            await asyncio.to_thread(metadata.delete_workspace, workspace.id)
        return OkResponse(detail=f"workspace '{workspace.slug}' deleted")

    # -- members -----------------------------------------------------------
    @app.get("/workspaces/{workspace_id}/members", response_model=list[MemberInfo])
    async def list_members(workspace_id: str, user: User = Depends(require_user)):
        workspace = require_access(workspace_id, user, Action.MANAGE_MEMBERS)
        members = await asyncio.to_thread(metadata.list_members, workspace.id)
        return [
            MemberInfo(user_id=member.id, username=member.username, role=role)
            for member, role in members
        ]

    @app.post(
        "/workspaces/{workspace_id}/members",
        response_model=MemberInfo,
        status_code=201,
    )
    async def add_member(
        workspace_id: str,
        request: AddMemberRequest,
        user: User = Depends(require_user),
    ):
        workspace = require_access(workspace_id, user, Action.MANAGE_MEMBERS)
        caller_rank = ROLE_RANK.get(effective_role(workspace, user) or "", 0)
        if ROLE_RANK.get(request.role, 0) > caller_rank:
            raise HTTPException(status_code=403, detail="cannot grant a role higher than your own")
        found = await asyncio.to_thread(metadata.get_user_by_username, request.username)
        if found is None:
            raise HTTPException(status_code=404, detail=f"user '{request.username}' not found")
        member = found[0]
        target_role = metadata.get_membership(workspace.id, member.id)
        if ROLE_RANK.get(target_role or "", 0) > caller_rank:
            raise HTTPException(status_code=403, detail="cannot modify a member with a higher role")
        await asyncio.to_thread(metadata.add_member, workspace.id, member.id, request.role)
        return MemberInfo(user_id=member.id, username=member.username, role=request.role)

    @app.delete("/workspaces/{workspace_id}/members/{user_id}", response_model=OkResponse)
    async def remove_member(workspace_id: str, user_id: str, user: User = Depends(require_user)):
        workspace = require_access(workspace_id, user, Action.MANAGE_MEMBERS)
        caller_rank = ROLE_RANK.get(effective_role(workspace, user) or "", 0)
        target_role = metadata.get_membership(workspace.id, user_id)
        if ROLE_RANK.get(target_role or "", 0) > caller_rank:
            raise HTTPException(status_code=403, detail="cannot remove a member with a higher role")
        removed = await asyncio.to_thread(metadata.remove_member, workspace.id, user_id)
        if not removed:
            raise HTTPException(status_code=404, detail="membership not found")
        return OkResponse()

    # -- workspace data ----------------------------------------------------
    @app.get("/ws/{workspace_id}/stats", response_model=StoreStats)
    async def stats(workspace_id: str, user: User = Depends(require_user)):
        workspace = require_access(workspace_id, user, Action.READ)
        result = await asyncio.to_thread(store.stats_sync, workspace.table_name)
        result.workspace_id = workspace.id
        result.slug = workspace.slug
        result.fts_ready = await asyncio.to_thread(metadata.get_fts_ready, workspace.id)
        return result

    @app.post("/ws/{workspace_id}/upsert")
    async def upsert(
        workspace_id: str,
        request: UpsertRequest,
        user: User = Depends(require_user),
    ):
        workspace = require_access(workspace_id, user, Action.WRITE)
        async with lock:
            try:
                rows, ready = await asyncio.to_thread(
                    store.upsert_sync,
                    workspace.table_name,
                    request.source,
                    request.rows,
                    request.rebuild_fts,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if ready is not None:
                await asyncio.to_thread(metadata.set_fts_ready, workspace.id, ready)
        return {"source": request.source, "rows": rows}

    @app.post("/ws/{workspace_id}/search", response_model=SearchResponse)
    async def search(
        workspace_id: str,
        request: SearchRequest,
        user: User = Depends(require_user),
    ):
        workspace = require_access(workspace_id, user, Action.READ)
        fts_ready = await asyncio.to_thread(metadata.get_fts_ready, workspace.id)
        try:
            return await asyncio.to_thread(
                store.search_sync, workspace.table_name, request, fts_ready
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/ws/{workspace_id}/delete")
    async def delete(
        workspace_id: str,
        request: DeleteRequest,
        user: User = Depends(require_user),
    ):
        workspace = require_access(workspace_id, user, Action.WRITE)
        async with lock:
            removed, ready = await asyncio.to_thread(
                store.delete_sync, workspace.table_name, request.source
            )
            await asyncio.to_thread(metadata.set_fts_ready, workspace.id, ready)
        return {"source": request.source, "removed": removed}

    @app.post("/ws/{workspace_id}/reindex")
    async def reindex(workspace_id: str, user: User = Depends(require_user)):
        workspace = require_access(workspace_id, user, Action.REINDEX)
        async with lock:
            ready = await asyncio.to_thread(store.reindex_table_sync, workspace.table_name)
            await asyncio.to_thread(metadata.set_fts_ready, workspace.id, ready)
        return {"fts_ready": ready}

    return app


app = create_app()
