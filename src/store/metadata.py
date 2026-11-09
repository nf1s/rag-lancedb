"""SQLite metadata: users, workspaces, memberships and sessions.

The store owns this database in the same PVC as LanceDB. At this scale SQLite is
more than enough and it keeps the number of workloads at three.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from common.passwords import hash_password, verify_password
from common.permissions import ASSIGNABLE_ROLES

logger = logging.getLogger("rag.store.metadata")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    global_role TEXT NOT NULL DEFAULT 'user',
    disabled INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    table_name TEXT NOT NULL UNIQUE,
    settings_json TEXT NOT NULL DEFAULT '{}',
    created_by TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memberships (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, workspace_id)
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_memberships_workspace ON memberships(workspace_id);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class User:
    id: str
    username: str
    global_role: str
    disabled: bool
    created_at: str


@dataclass(frozen=True)
class Workspace:
    id: str
    slug: str
    name: str
    table_name: str
    created_at: str
    settings: dict[str, Any] = field(default_factory=dict)


class MetadataStore:
    """Small, thread-safe wrapper around the SQLite metadata database.

    The connection is opened in :meth:`open` so that importing the module (and
    therefore constructing the FastAPI app) has no filesystem side effects.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        conn.commit()
        with self._lock:
            self._conn = conn
        logger.info("metadata ready at %s", self.path)

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("metadata store is not open")
        return self._conn

    # -- users -------------------------------------------------------------
    def user_count(self) -> int:
        with self._lock:
            return int(self._connection().execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def create_user(self, username: str, password: str, global_role: str = "user") -> User:
        if global_role not in {"admin", "user"}:
            raise ValueError(f"invalid global role: {global_role}")
        user = User(
            id=uuid.uuid4().hex,
            username=username,
            global_role=global_role,
            disabled=False,
            created_at=_now(),
        )
        password_hash = hash_password(password)
        with self._lock, self._connection() as conn:
            try:
                conn.execute(
                    "INSERT INTO users (id, username, password_hash, global_role, disabled, "
                    "created_at) VALUES (?, ?, ?, ?, 0, ?)",
                    (
                        user.id,
                        user.username,
                        password_hash,
                        user.global_role,
                        user.created_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"username '{username}' already exists") from exc
        return user

    def get_user(self, user_id: str) -> User | None:
        with self._lock:
            row = (
                self._connection()
                .execute("SELECT * FROM users WHERE id = ?", (user_id,))
                .fetchone()
            )
        return _row_to_user(row) if row else None

    def get_user_by_username(self, username: str) -> tuple[User, str] | None:
        with self._lock:
            row = (
                self._connection()
                .execute("SELECT * FROM users WHERE username = ?", (username,))
                .fetchone()
            )
        if row is None:
            return None
        return _row_to_user(row), str(row["password_hash"])

    def list_users(self) -> list[User]:
        with self._lock:
            rows = self._connection().execute("SELECT * FROM users ORDER BY username").fetchall()
        return [_row_to_user(row) for row in rows]

    def set_user_disabled(self, user_id: str, disabled: bool) -> bool:
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "UPDATE users SET disabled = ? WHERE id = ?", (int(disabled), user_id)
            )
            if disabled:
                conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        return cursor.rowcount > 0

    def set_user_password(self, user_id: str, password: str) -> bool:
        password_hash = hash_password(password)
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id)
            )
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        return cursor.rowcount > 0

    def authenticate(self, username: str, password: str) -> User | None:
        found = self.get_user_by_username(username)
        if found is None:
            return None
        user, password_hash = found
        if user.disabled or not verify_password(password, password_hash):
            return None
        return user

    # -- sessions ----------------------------------------------------------
    def create_session(self, user_id: str, ttl_hours: float) -> str:
        token = secrets.token_urlsafe(32)
        created_at = datetime.now(UTC)
        expires_at = created_at + timedelta(hours=max(0.1, ttl_hours))
        with self._lock, self._connection() as conn:
            conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (_now(),))
            conn.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, expires_at, revoked) "
                "VALUES (?, ?, ?, ?, 0)",
                (_token_hash(token), user_id, created_at.isoformat(), expires_at.isoformat()),
            )
        return token

    def validate_session(self, token: str) -> User | None:
        with self._lock:
            row = (
                self._connection()
                .execute(
                    "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id "
                    "WHERE s.token_hash = ? AND s.revoked = 0 AND s.expires_at > ? "
                    "AND u.disabled = 0",
                    (_token_hash(token), _now()),
                )
                .fetchone()
            )
        return _row_to_user(row) if row else None

    def revoke_session(self, token: str) -> bool:
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "UPDATE sessions SET revoked = 1 WHERE token_hash = ?", (_token_hash(token),)
            )
        return cursor.rowcount > 0

    # -- workspaces --------------------------------------------------------
    def workspace_count(self) -> int:
        with self._lock:
            return int(self._connection().execute("SELECT COUNT(*) FROM workspaces").fetchone()[0])

    def create_workspace(
        self,
        slug: str,
        name: str,
        table_name: str,
        created_by: str | None,
        settings: dict[str, Any] | None = None,
    ) -> Workspace:
        workspace = Workspace(
            id=uuid.uuid4().hex,
            slug=slug,
            name=name,
            table_name=table_name,
            created_at=_now(),
            settings=dict(settings or {}),
        )
        with self._lock, self._connection() as conn:
            try:
                conn.execute(
                    "INSERT INTO workspaces (id, slug, name, table_name, settings_json, "
                    "created_by, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        workspace.id,
                        workspace.slug,
                        workspace.name,
                        workspace.table_name,
                        json.dumps(workspace.settings, sort_keys=True),
                        created_by,
                        workspace.created_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"workspace '{slug}' already exists") from exc
        return workspace

    def get_workspace(self, workspace_id: str) -> Workspace | None:
        with self._lock:
            row = (
                self._connection()
                .execute("SELECT * FROM workspaces WHERE id = ?", (workspace_id,))
                .fetchone()
            )
        return _row_to_workspace(row) if row else None

    def get_workspace_by_slug(self, slug: str) -> Workspace | None:
        with self._lock:
            row = (
                self._connection()
                .execute("SELECT * FROM workspaces WHERE slug = ?", (slug,))
                .fetchone()
            )
        return _row_to_workspace(row) if row else None

    def list_workspaces(self) -> list[Workspace]:
        with self._lock:
            rows = self._connection().execute("SELECT * FROM workspaces ORDER BY slug").fetchall()
        return [_row_to_workspace(row) for row in rows]

    def list_workspaces_for(self, user: User) -> list[tuple[Workspace, str]]:
        """Return ``(workspace, effective role)`` pairs visible to ``user``."""
        with self._lock:
            if user.global_role == "admin":
                rows = (
                    self._connection().execute("SELECT * FROM workspaces ORDER BY slug").fetchall()
                )
                return [(_row_to_workspace(row), "owner") for row in rows]
            rows = (
                self._connection()
                .execute(
                    "SELECT w.*, m.role AS membership_role FROM workspaces w "
                    "JOIN memberships m ON m.workspace_id = w.id "
                    "WHERE m.user_id = ? ORDER BY w.slug",
                    (user.id,),
                )
                .fetchall()
            )
        return [(_row_to_workspace(row), str(row["membership_role"])) for row in rows]

    def delete_workspace(self, workspace_id: str) -> bool:
        with self._lock, self._connection() as conn:
            cursor = conn.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))
        return cursor.rowcount > 0

    def set_fts_ready(self, workspace_id: str, ready: bool) -> None:
        with self._lock:
            row = (
                self._connection()
                .execute("SELECT settings_json FROM workspaces WHERE id = ?", (workspace_id,))
                .fetchone()
            )
            if row is None:
                return
            settings = _load_json(row["settings_json"])
            settings["fts_ready"] = bool(ready)
            with self._connection() as conn:
                conn.execute(
                    "UPDATE workspaces SET settings_json = ? WHERE id = ?",
                    (json.dumps(settings, sort_keys=True), workspace_id),
                )

    def get_fts_ready(self, workspace_id: str) -> bool:
        with self._lock:
            row = (
                self._connection()
                .execute("SELECT settings_json FROM workspaces WHERE id = ?", (workspace_id,))
                .fetchone()
            )
        return bool(_load_json(row["settings_json"]).get("fts_ready")) if row else False

    # -- memberships -------------------------------------------------------
    def add_member(self, workspace_id: str, user_id: str, role: str) -> None:
        if role not in ASSIGNABLE_ROLES:
            raise ValueError(f"invalid workspace role: {role}")
        with self._lock, self._connection() as conn:
            conn.execute(
                "INSERT INTO memberships (user_id, workspace_id, role, created_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(user_id, workspace_id) "
                "DO UPDATE SET role = excluded.role",
                (user_id, workspace_id, role, _now()),
            )

    def remove_member(self, workspace_id: str, user_id: str) -> bool:
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM memberships WHERE workspace_id = ? AND user_id = ?",
                (workspace_id, user_id),
            )
        return cursor.rowcount > 0

    def get_membership(self, workspace_id: str, user_id: str) -> str | None:
        with self._lock:
            row = (
                self._connection()
                .execute(
                    "SELECT role FROM memberships WHERE workspace_id = ? AND user_id = ?",
                    (workspace_id, user_id),
                )
                .fetchone()
            )
        return str(row["role"]) if row else None

    def list_members(self, workspace_id: str) -> list[tuple[User, str]]:
        with self._lock:
            rows = (
                self._connection()
                .execute(
                    "SELECT u.*, m.role AS membership_role FROM memberships m "
                    "JOIN users u ON u.id = m.user_id WHERE m.workspace_id = ? "
                    "ORDER BY u.username",
                    (workspace_id,),
                )
                .fetchall()
            )
        return [(_row_to_user(row), str(row["membership_role"])) for row in rows]


def _row_to_user(row: sqlite3.Row) -> User:
    return User(
        id=str(row["id"]),
        username=str(row["username"]),
        global_role=str(row["global_role"]),
        disabled=bool(row["disabled"]),
        created_at=str(row["created_at"]),
    )


def _row_to_workspace(row: sqlite3.Row) -> Workspace:
    return Workspace(
        id=str(row["id"]),
        slug=str(row["slug"]),
        name=str(row["name"]),
        table_name=str(row["table_name"]),
        created_at=str(row["created_at"]),
        settings=_load_json(row["settings_json"]),
    )


def _load_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}
