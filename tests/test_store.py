import json

import lancedb
import pyarrow as pa
import pytest
from fastapi.testclient import TestClient

from common.config import Settings
from store.main import create_app

DIM = 4
PASSWORD = "secret123"


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        lancedb_uri=str(tmp_path / "db"),
        embedding_dim=DIM,
        table_name="documents",
        bootstrap_admin_user="admin",
        bootstrap_admin_password=PASSWORD,
        session_ttl_hours=1,
        log_level="WARNING",
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def login(client: TestClient, username: str = "admin", password: str = PASSWORD) -> str:
    response = client.post("/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["token"]


@pytest.fixture
def token(client):
    return login(client)


@pytest.fixture
def workspace(client, token):
    me = client.get("/auth/me", headers=auth(token)).json()
    assert me["workspaces"]
    return me["workspaces"][0]["id"]


def auth(token: str) -> dict[str, str]:
    return {"X-Rag-Token": token}


def make_row(index: int, source: str = "a.md", text: str | None = None) -> dict:
    return {
        "id": f"{source}-{index}",
        "source": source,
        "chunk_index": index,
        "text": text or f"chunk {index} about puppies and gardens",
        "vector": [1.0, 0.0, 0.0, 0.0],
        "metadata": {"heading": "Intro"},
    }


def test_health_and_stats(client, token, workspace):
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["workspaces"] == 1

    stats = client.get(f"/ws/{workspace}/stats", headers=auth(token)).json()
    assert stats["rows"] == 0
    assert stats["dim"] == DIM
    assert stats["fts_ready"] is False
    assert stats["slug"] == "default"


def test_requires_token(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/auth/me").status_code == 401
    assert client.post("/ws/anything/search", json={}).status_code == 401


def test_login_rejects_bad_password(client):
    response = client.post("/auth/login", json={"username": "admin", "password": "wrong-password"})
    assert response.status_code == 401


def test_upsert_search_and_stats(client, token, workspace):
    headers = auth(token)
    response = client.post(
        f"/ws/{workspace}/upsert",
        headers=headers,
        json={"source": "a.md", "rows": [make_row(0), make_row(1)], "rebuild_fts": True},
    )
    assert response.status_code == 200, response.text
    assert response.json()["rows"] == 2

    stats = client.get(f"/ws/{workspace}/stats", headers=headers).json()
    assert stats["rows"] == 2
    assert stats["sources"] == 1
    assert stats["fts_ready"] is True

    search = client.post(
        f"/ws/{workspace}/search",
        headers=headers,
        json={
            "vector": [1.0, 0.0, 0.0, 0.0],
            "text": "puppies",
            "mode": "vector",
            "top_k": 5,
        },
    )
    assert search.status_code == 200, search.text
    body = search.json()
    assert body["mode"] == "vector"
    assert len(body["hits"]) == 2
    assert body["hits"][0]["source"] == "a.md"
    assert body["hits"][0]["score"] == pytest.approx(1.0)


def test_hybrid_search_returns_hits(client, token, workspace):
    headers = auth(token)
    client.post(
        f"/ws/{workspace}/upsert",
        headers=headers,
        json={"source": "a.md", "rows": [make_row(0, text="puppies in the garden")]},
    )
    response = client.post(
        f"/ws/{workspace}/search",
        headers=headers,
        json={
            "vector": [1.0, 0.0, 0.0, 0.0],
            "text": "puppies garden",
            "mode": "hybrid",
            "top_k": 5,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["mode"] == "hybrid"
    assert response.json()["hits"]


def test_hybrid_falls_back_to_vector_without_fts(client, token, workspace):
    headers = auth(token)
    client.post(
        f"/ws/{workspace}/upsert",
        headers=headers,
        json={"source": "a.md", "rows": [make_row(0)], "rebuild_fts": False},
    )
    response = client.post(
        f"/ws/{workspace}/search",
        headers=headers,
        json={
            "vector": [1.0, 0.0, 0.0, 0.0],
            "text": "puppies",
            "mode": "hybrid",
            "top_k": 5,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["mode"] == "vector"


def test_replace_by_source_removes_stale_chunks(client, token, workspace):
    headers = auth(token)
    client.post(
        f"/ws/{workspace}/upsert",
        headers=headers,
        json={"source": "a.md", "rows": [make_row(i) for i in range(3)]},
    )
    client.post(
        f"/ws/{workspace}/upsert",
        headers=headers,
        json={"source": "a.md", "rows": [make_row(0)]},
    )
    assert client.get(f"/ws/{workspace}/stats", headers=headers).json()["rows"] == 1

    client.post(
        f"/ws/{workspace}/upsert",
        headers=headers,
        json={"source": "b.md", "rows": [make_row(0, source="b.md")]},
    )
    stats = client.get(f"/ws/{workspace}/stats", headers=headers).json()
    assert stats["rows"] == 2
    assert stats["sources"] == 2


def test_delete_source(client, token, workspace):
    headers = auth(token)
    client.post(
        f"/ws/{workspace}/upsert",
        headers=headers,
        json={"source": "a.md", "rows": [make_row(0)]},
    )
    response = client.post(f"/ws/{workspace}/delete", headers=headers, json={"source": "a.md"})
    assert response.status_code == 200
    assert response.json()["removed"] == 1
    assert client.get(f"/ws/{workspace}/stats", headers=headers).json()["rows"] == 0


def test_rejects_wrong_vector_dimension(client, token, workspace):
    row = make_row(0)
    row["vector"] = [1.0, 0.0]
    response = client.post(
        f"/ws/{workspace}/upsert",
        headers=auth(token),
        json={"source": "a.md", "rows": [row]},
    )
    assert response.status_code == 400
    assert "expected" in response.json()["detail"]


def test_rejects_unknown_source_delete(client, token, workspace):
    response = client.post(
        f"/ws/{workspace}/delete", headers=auth(token), json={"source": "missing.md"}
    )
    assert response.status_code == 200
    assert response.json()["removed"] == 0


def test_workspace_roles_and_isolation(client, token, workspace):
    headers = auth(token)
    created = client.post(
        "/users",
        headers=headers,
        json={"username": "bob", "password": "password123"},
    )
    assert created.status_code == 201, created.text

    created = client.post("/workspaces", headers=headers, json={"slug": "second", "name": "Second"})
    assert created.status_code == 201, created.text
    second = created.json()["id"]

    bob = login(client, "bob", "password123")
    bob_headers = auth(bob)

    # Not a member yet.
    assert client.get(f"/ws/{second}/stats", headers=bob_headers).status_code == 403
    # Global endpoints are admin-only.
    assert (
        client.post(
            "/workspaces", headers=bob_headers, json={"slug": "nope", "name": "Nope"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/users",
            headers=bob_headers,
            json={"username": "eve", "password": "password123"},
        ).status_code
        == 403
    )

    added = client.post(
        f"/workspaces/{second}/members",
        headers=headers,
        json={"username": "bob", "role": "viewer"},
    )
    assert added.status_code == 201, added.text
    assert client.get(f"/ws/{second}/stats", headers=bob_headers).status_code == 200

    # Viewers can read but not write or manage members.
    ingest = client.post(
        f"/ws/{second}/upsert",
        headers=bob_headers,
        json={"source": "b.md", "rows": [make_row(0, source="b.md")]},
    )
    assert ingest.status_code == 403
    assert client.get(f"/workspaces/{second}/members", headers=bob_headers).status_code == 403
    # ... and cannot read the workspace they are not a member of.
    assert client.get(f"/ws/{workspace}/stats", headers=bob_headers).status_code == 403

    # Editors can ingest but still cannot reindex.
    client.post(
        f"/workspaces/{second}/members",
        headers=headers,
        json={"username": "bob", "role": "editor"},
    )
    ingest = client.post(
        f"/ws/{second}/upsert",
        headers=bob_headers,
        json={
            "source": "b.md",
            "rows": [make_row(0, source="b.md")],
            "rebuild_fts": True,
        },
    )
    assert ingest.status_code == 200, ingest.text
    assert client.post(f"/ws/{second}/reindex", headers=bob_headers).status_code == 403

    # A workspace admin cannot promote themselves or anyone above their own rank,
    # and cannot remove the owner.
    client.post(
        f"/workspaces/{second}/members",
        headers=headers,
        json={"username": "bob", "role": "admin"},
    )
    promote = client.post(
        f"/workspaces/{second}/members",
        headers=bob_headers,
        json={"username": "bob", "role": "owner"},
    )
    assert promote.status_code == 403
    owner_id = client.get("/auth/me", headers=headers).json()["user"]["id"]
    remove_owner = client.delete(f"/workspaces/{second}/members/{owner_id}", headers=bob_headers)
    assert remove_owner.status_code == 403
    assert client.post(f"/ws/{second}/reindex", headers=bob_headers).status_code == 200

    # The global admin can see and write both workspaces.
    assert client.get(f"/ws/{workspace}/stats", headers=headers).status_code == 200
    assert client.get(f"/ws/{second}/stats", headers=headers).status_code == 200


def test_same_source_name_is_isolated_between_workspaces(client, token, workspace):
    headers = auth(token)
    second = client.post(
        "/workspaces", headers=headers, json={"slug": "other", "name": "Other"}
    ).json()["id"]

    for ws in (workspace, second):
        response = client.post(
            f"/ws/{ws}/upsert",
            headers=headers,
            json={"source": "same.md", "rows": [make_row(0, source="same.md")]},
        )
        assert response.status_code == 200, response.text

    assert client.get(f"/ws/{workspace}/stats", headers=headers).json()["rows"] == 1
    assert client.get(f"/ws/{second}/stats", headers=headers).json()["rows"] == 1

    client.post(f"/ws/{second}/delete", headers=headers, json={"source": "same.md"})
    assert client.get(f"/ws/{workspace}/stats", headers=headers).json()["rows"] == 1
    assert client.get(f"/ws/{second}/stats", headers=headers).json()["rows"] == 0


def test_owner_can_delete_workspace(client, token):
    headers = auth(token)
    created = client.post(
        "/workspaces", headers=headers, json={"slug": "temporary", "name": "Temporary"}
    )
    workspace_id = created.json()["id"]

    response = client.delete(f"/workspaces/{workspace_id}", headers=headers)
    assert response.status_code == 200, response.text
    assert client.get(f"/ws/{workspace_id}/stats", headers=headers).status_code == 404


def test_duplicate_workspace_slug_conflicts(client, token):
    response = client.post(
        "/workspaces", headers=auth(token), json={"slug": "default", "name": "Dup"}
    )
    assert response.status_code == 409


def test_detects_embedding_model_mismatch(tmp_path):
    uri = str(tmp_path / "db")
    settings = Settings(
        lancedb_uri=uri,
        embedding_dim=DIM,
        bootstrap_admin_password=PASSWORD,
        log_level="WARNING",
    )
    with TestClient(create_app(settings)) as first:
        token = login(first)
        workspace = first.get("/auth/me", headers=auth(token)).json()["workspaces"][0]["id"]
        first.post(
            f"/ws/{workspace}/upsert",
            headers=auth(token),
            json={"source": "a.md", "rows": [make_row(0)]},
        )

    with TestClient(
        create_app(
            Settings(
                lancedb_uri=uri,
                embedding_dim=8,
                bootstrap_admin_password=PASSWORD,
                log_level="WARNING",
            )
        )
    ) as second:
        health = second.get("/healthz")
        assert health.status_code == 503
        assert "embedding mismatch" in health.json()["error"]


def test_legacy_documents_table_becomes_default_workspace(tmp_path):
    uri = tmp_path / "db"
    uri.mkdir()
    db = lancedb.connect(str(uri))
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), DIM)),
            pa.field("metadata", pa.string()),
        ]
    )
    table = db.create_table("documents", schema=schema)
    table.add(
        [
            {
                "id": "legacy-1",
                "source": "old.md",
                "chunk_index": 0,
                "text": "legacy content",
                "vector": [1.0, 0.0, 0.0, 0.0],
                "metadata": "{}",
            }
        ]
    )
    del table
    del db

    settings = Settings(
        lancedb_uri=str(uri),
        embedding_dim=DIM,
        bootstrap_admin_password=PASSWORD,
        log_level="WARNING",
    )
    (uri / "meta.json").write_text(
        json.dumps(
            {
                "model": settings.embedding_model,
                "dim": DIM,
                "fts_ready": False,
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": None,
            }
        )
    )

    with TestClient(create_app(settings)) as client:
        token = login(client)
        me = client.get("/auth/me", headers=auth(token)).json()
        assert len(me["workspaces"]) == 1
        workspace = me["workspaces"][0]
        assert workspace["slug"] == "default"
        stats = client.get(f"/ws/{workspace['id']}/stats", headers=auth(token)).json()
        assert stats["rows"] == 1
        assert stats["sources"] == 1
        assert stats["fts_ready"] is False
