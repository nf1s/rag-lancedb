"""Pydantic schemas for the store API and the backend API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RetrievalMode = Literal["vector", "fts", "hybrid"]
GlobalRole = Literal["admin", "user"]
WorkspaceRole = Literal["viewer", "editor", "admin", "owner"]


# --------------------------------------------------------------------------- store


class Chunk(BaseModel):
    id: str
    source: str
    chunk_index: int
    text: str
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpsertRequest(BaseModel):
    source: str
    rows: list[Chunk]
    rebuild_fts: bool = True


class DeleteRequest(BaseModel):
    source: str


class SearchRequest(BaseModel):
    vector: list[float] | None = None
    text: str | None = None
    mode: RetrievalMode = "vector"
    top_k: int = 5
    where: str | None = None


class SearchHit(BaseModel):
    score: float
    id: str
    source: str
    chunk_index: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    mode: RetrievalMode
    took_ms: float


class StoreStats(BaseModel):
    workspace_id: str | None = None
    slug: str | None = None
    table: str
    rows: int
    sources: int
    dim: int
    model: str
    fts_ready: bool
    updated_at: str | None = None


class StoreHealth(BaseModel):
    status: Literal["ok", "error"]
    workspaces: int = 0
    error: str | None = None


# --------------------------------------------------------------------------- auth


class UserInfo(BaseModel):
    id: str
    username: str
    global_role: GlobalRole
    disabled: bool = False
    created_at: str | None = None


class WorkspaceInfo(BaseModel):
    id: str
    slug: str
    name: str
    role: WorkspaceRole | None = None
    created_at: str | None = None


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class LoginResponse(BaseModel):
    token: str
    user: UserInfo
    workspaces: list[WorkspaceInfo]


class MeResponse(BaseModel):
    user: UserInfo
    workspaces: list[WorkspaceInfo]


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=256)
    global_role: GlobalRole = "user"


class CreateWorkspaceRequest(BaseModel):
    slug: str = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=1, max_length=120)


class MemberInfo(BaseModel):
    user_id: str
    username: str
    role: WorkspaceRole


class AddMemberRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    role: WorkspaceRole


class OkResponse(BaseModel):
    ok: bool = True
    detail: str | None = None


# --------------------------------------------------------------------------- api


class SearchQuery(BaseModel):
    question: str
    top_k: int | None = None
    mode: RetrievalMode | None = None


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    mode: RetrievalMode | None = None


class IngestResult(BaseModel):
    source: str
    chunks: int
    warning: str | None = None


class IngestResponse(BaseModel):
    results: list[IngestResult]
