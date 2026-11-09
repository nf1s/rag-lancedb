import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.store_client import StoreResponseError
from common.config import Settings

DIM = 4
TOKEN = "tok"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


class FakeEmbedder:
    model_name = "fake-embedder"
    dim = DIM

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]


class FakeStore:
    def __init__(self):
        # workspace id -> source name -> rows
        self.sources: dict[str, dict[str, list[dict]]] = {}
        self.fts_rebuilds = 0
        self.deleted_workspaces: list[str] = []
        self.deny = False
        self.workspaces = [
            {
                "id": "ws-1",
                "slug": "default",
                "name": "Default",
                "role": "owner",
                "created_at": None,
            }
        ]
        self.user = {
            "id": "u1",
            "username": "admin",
            "global_role": "admin",
            "disabled": False,
            "created_at": None,
        }

    def _check(self, token: str) -> None:
        if token != TOKEN:
            raise StoreResponseError(401, "invalid or expired session")

    def _rows(self, workspace_id: str) -> list[dict]:
        return [row for rows in self.sources.get(workspace_id, {}).values() for row in rows]

    async def health(self):
        return {"status": "ok", "workspaces": len(self.workspaces)}

    # -- auth --------------------------------------------------------------
    async def login(self, username, password):
        if username != "admin" or password != "secret":
            raise StoreResponseError(401, "invalid username or password")
        return {"token": TOKEN, "user": self.user, "workspaces": self.workspaces}

    async def me(self, token):
        self._check(token)
        return {"user": self.user, "workspaces": self.workspaces}

    async def logout(self, token):
        self._check(token)
        return {"ok": True}

    # -- admin -------------------------------------------------------------
    async def list_users(self, token):
        self._check(token)
        return [self.user]

    async def create_user(self, token, username, password, global_role):
        self._check(token)
        return {
            "id": "u2",
            "username": username,
            "global_role": global_role,
            "disabled": False,
            "created_at": None,
        }

    async def list_workspaces(self, token):
        self._check(token)
        return self.workspaces

    async def create_workspace(self, token, slug, name):
        self._check(token)
        workspace = {
            "id": f"ws-{len(self.workspaces) + 1}",
            "slug": slug,
            "name": name,
            "role": "owner",
            "created_at": None,
        }
        self.workspaces.append(workspace)
        return workspace

    async def delete_workspace(self, token, workspace_id):
        self._check(token)
        self.deleted_workspaces.append(workspace_id)
        return {"ok": True}

    async def list_members(self, token, workspace_id):
        self._check(token)
        return [{"user_id": "u1", "username": "admin", "role": "owner"}]

    async def add_member(self, token, workspace_id, username, role):
        self._check(token)
        return {"user_id": "u2", "username": username, "role": role}

    async def remove_member(self, token, workspace_id, user_id):
        self._check(token)
        return {"ok": True}

    # -- data --------------------------------------------------------------
    async def stats(self, token, workspace_id):
        self._check(token)
        rows = self._rows(workspace_id)
        return {
            "workspace_id": workspace_id,
            "slug": "default",
            "table": "ws-test",
            "rows": len(rows),
            "sources": len(self.sources.get(workspace_id, {})),
            "dim": DIM,
            "model": "fake-embedder",
            "fts_ready": True,
            "updated_at": None,
        }

    async def upsert(self, token, workspace_id, source, rows, rebuild_fts=False):
        self._check(token)
        self.sources.setdefault(workspace_id, {})[source] = rows
        if rebuild_fts:
            self.fts_rebuilds += 1
        return {"source": source, "rows": len(rows)}

    async def reindex(self, token, workspace_id):
        self._check(token)
        self.fts_rebuilds += 1
        return {"fts_ready": True}

    async def search(self, *, token, workspace_id, vector, text, mode, top_k, where=None):
        self._check(token)
        if self.deny:
            raise StoreResponseError(403, "role 'viewer' cannot perform 'read'")
        hits = [
            {
                "score": 0.9,
                "id": row["id"],
                "source": row["source"],
                "chunk_index": row["chunk_index"],
                "text": row["text"],
                "metadata": row["metadata"],
            }
            for row in self._rows(workspace_id)
        ]
        return {"hits": hits[:top_k], "mode": mode, "took_ms": 0.5}


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def received_prompts():
    return []


@pytest.fixture
def client(store, received_prompts):
    settings = Settings(
        log_level="WARNING",
        embedding_dim=DIM,
        llm_base_url="http://127.0.0.1:1/v1",
        llm_health_timeout_seconds=0.2,
        chunk_size=100,
        chunk_overlap=10,
    )

    async def fake_llm(messages):
        received_prompts.append(messages)
        for token in ["Hello", " world", "!"]:
            yield token

    app = create_app(
        settings=settings,
        embedder=FakeEmbedder(),
        store=store,
        llm_stream=fake_llm,
    )
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_store_and_llm(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["store"]["status"] == "ok"
    assert body["llm"]["reachable"] is False


def test_login_me_and_logout(client):
    response = client.post("/auth/login", json={"username": "admin", "password": "secret"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["token"] == TOKEN
    assert body["user"]["username"] == "admin"
    assert body["workspaces"][0]["slug"] == "default"

    me = client.get("/auth/me", headers=HEADERS)
    assert me.status_code == 200
    assert me.json()["user"]["global_role"] == "admin"

    assert client.post("/auth/logout", headers=HEADERS).status_code == 200


def test_login_failure_is_forwarded(client):
    response = client.post("/auth/login", json={"username": "admin", "password": "nope"})
    assert response.status_code == 401
    assert "invalid" in response.json()["detail"]


def test_requires_bearer_token(client):
    assert client.get("/auth/me").status_code == 401
    assert client.get("/ws/ws-1/stats").status_code == 401
    assert client.post("/ws/ws-1/search", json={"question": "hi"}).status_code == 401


def test_invalid_token_is_forwarded(client):
    response = client.get("/auth/me", headers={"Authorization": "Bearer expired"})
    assert response.status_code == 401


def test_ingest_indexes_markdown(client, store):
    response = client.post(
        "/ws/ws-1/ingest",
        headers=HEADERS,
        files={"files": ("notes.md", b"# Title\n\nHello world. " * 30, "text/markdown")},
    )
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert results[0]["source"] == "notes.md"
    assert results[0]["chunks"] > 0
    assert "notes.md" in store.sources["ws-1"]
    assert store.fts_rebuilds == 1


def test_ingest_multiple_files_rebuilds_fts_once(client, store):
    response = client.post(
        "/ws/ws-1/ingest",
        headers=HEADERS,
        files=[
            ("files", ("a.md", b"# A\n\nBody a.", "text/markdown")),
            ("files", ("b.md", b"# B\n\nBody b.", "text/markdown")),
        ],
    )
    assert response.status_code == 200, response.text
    assert set(store.sources["ws-1"]) == {"a.md", "b.md"}
    assert store.fts_rebuilds == 1


def test_ingest_rejects_unsupported_type(client, store):
    response = client.post(
        "/ws/ws-1/ingest",
        headers=HEADERS,
        files={"files": ("paper.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["chunks"] == 0
    assert "unsupported" in result["warning"]
    assert store.sources == {}


def test_ingest_warns_on_empty_document(client):
    response = client.post(
        "/ws/ws-1/ingest",
        headers=HEADERS,
        files={"files": ("empty.txt", b"   \n\n  ", "text/plain")},
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["chunks"] == 0
    assert result["warning"] == "no extractable text"


def test_search_uses_embedder_and_workspace(client, store):
    client.post(
        "/ws/ws-1/ingest",
        headers=HEADERS,
        files={"files": ("a.md", b"# Doc\n\nPuppies in the garden.", "text/markdown")},
    )
    response = client.post(
        "/ws/ws-1/search", headers=HEADERS, json={"question": "where are the puppies?"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["hits"][0]["source"] == "a.md"
    assert "a.md" in store.sources["ws-1"]


def test_reindex_proxies_store(client, store):
    response = client.post("/ws/ws-1/reindex", headers=HEADERS)
    assert response.status_code == 200, response.text
    assert response.json()["fts_ready"] is True
    assert store.fts_rebuilds == 1


def test_store_permission_error_is_forwarded(client, store):
    store.deny = True
    response = client.post("/ws/ws-1/search", headers=HEADERS, json={"question": "anything"})
    assert response.status_code == 403
    assert "cannot perform" in response.json()["detail"]


def test_query_streams_sources_tokens_and_done(client, received_prompts):
    client.post(
        "/ws/ws-1/ingest",
        headers=HEADERS,
        files={"files": ("a.md", b"# Doc\n\nPuppies in the garden.", "text/markdown")},
    )
    response = client.post(
        "/ws/ws-1/query", headers=HEADERS, json={"question": "where are the puppies?"}
    )
    assert response.status_code == 200
    body = response.text
    assert "event: sources" in body
    assert "event: token" in body
    assert "event: done" in body
    assert "Hello" in body
    assert received_prompts
    assert "[1]" in received_prompts[0][1]["content"]


def test_query_without_hits_does_not_call_llm(client, received_prompts):
    response = client.post("/ws/ws-1/query", headers=HEADERS, json={"question": "anything?"})
    assert response.status_code == 200
    body = response.text
    assert "could not find" in body
    assert received_prompts == []


def test_stats_proxies_store(client):
    client.post(
        "/ws/ws-1/ingest",
        headers=HEADERS,
        files={"files": ("a.md", b"# Doc\n\nBody.", "text/markdown")},
    )
    stats = client.get("/ws/ws-1/stats", headers=HEADERS).json()
    assert stats["rows"] > 0
    assert stats["sources"] == 1
    assert stats["workspace_id"] == "ws-1"


def test_admin_proxies(client, store):
    created = client.post(
        "/workspaces",
        headers=HEADERS,
        json={"slug": "second", "name": "Second"},
    )
    assert created.status_code == 201, created.text
    assert created.json()["slug"] == "second"

    user = client.post(
        "/users",
        headers=HEADERS,
        json={"username": "bob", "password": "password123", "global_role": "user"},
    )
    assert user.status_code == 201, user.text
    assert user.json()["username"] == "bob"

    members = client.get("/workspaces/ws-1/members", headers=HEADERS)
    assert members.status_code == 200
    assert members.json()[0]["role"] == "owner"

    added = client.post(
        "/workspaces/ws-1/members",
        headers=HEADERS,
        json={"username": "bob", "role": "viewer"},
    )
    assert added.status_code == 201
    assert added.json()["role"] == "viewer"

    removed = client.delete("/workspaces/ws-1/members/u2", headers=HEADERS)
    assert removed.status_code == 200

    deleted = client.delete("/workspaces/ws-2", headers=HEADERS)
    assert deleted.status_code == 200
    assert store.deleted_workspaces == ["ws-2"]
