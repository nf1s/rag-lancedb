---
title: "How to use LanceDB for RAG: hybrid search, one table per workspace, and the single-writer catch"
published: true
description: "A field report from building a local RAG stack on k3s: table design, replace-by-source ingest, native FTS + RRF hybrid search, per-workspace tables, and the embedded-database constraint that shapes the whole architecture."
tags: rag, lancedb, python, kubernetes
cover_image: https://raw.githubusercontent.com/nf1s/rag-lancedb/main/assets/cover-wide.png
---

LanceDB is not a server. It is a library you import, and it reads and writes a directory on disk. That single fact is why it is pleasant to use for RAG, and why running it inside Kubernetes forces an unusual architecture.

I spent the last stretch building a local RAG stack on it: Markdown/text in, hybrid retrieval, streamed answers with citations, all on a Colima k3s cluster with three containers and no cloud services. Then I added workspaces and permissions. This is a field report on what LanceDB actually gives you for RAG, with the parts that bit me written down. Everything here runs on one Mac; the numbers come from a smoke test, not a benchmark.

## Why LanceDB fits RAG

A RAG index needs three things: store vectors, search them by similarity, and search the same text by keywords. Most setups reach for a vector database plus Elasticsearch, or a vector DB with a keyword add-on that is easy to forget. LanceDB gives you all three from one library:

- **Vector search** over a fixed-size `float32` column, with cosine distance.
- **Native full-text search** via an FTS index on a text column.
- **Hybrid search** that merges both ranked lists with reciprocal rank fusion (`RRFReranker`).

It is Arrow-native, so the table is just typed columns. There is no query language to learn beyond `table.search(...)`, no daemon to keep alive, and no network hop between the embedding code and the index. The trade-off is that the database lives in your process, which we will get to.

## The shape of a RAG table

Here is the whole schema. Shortened, but not by much:

```python
def schema(dim: int = 384) -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("source", pa.string()),
        pa.field("chunk_index", pa.int32()),
        pa.field("text", pa.string()),          # FTS-indexed
        pa.field("vector", pa.list_(pa.float32(), dim)),
        pa.field("metadata", pa.string()),      # JSON: heading, etc.
    ])
```

`dim=384` comes from `BAAI/bge-small-en-v1.5`, embedded with `fastembed` on CPU. `source` is the filename, which is also the unit of replacement. `text` is the chunk the FTS index sees. `metadata` is a JSON string because it is cheap and I do not filter on it.

Two details matter more than they look. First, the vector column is a **fixed-size list**, so a wrong-dimensional embedding is rejected at write time instead of silently corrupting a search later. Second, the chunk `id` is a hash of `workspace_id + source + chunk_index + text`, so re-ingesting the same document is idempotent.

## Ingest: replace-by-source

Chunking is deliberately boring: a markdown-heading-aware recursive splitter, 400 tokens with 50 overlap. `bge-small` truncates at 512 tokens, so 800-token chunks would quietly lose content. Each batch is embedded, then written:

```python
if table.count_rows():
    table.delete(where_source(source))   # "source = 'notes.md'"
table.add(batch(rows))
table.create_index("text", config=FTS(), replace=True)
```

Deleting by source before adding is the whole ingest transaction. If a file shrinks, stale chunks disappear; if you upload the same file twice, you still get one copy. All of it runs behind a single write lock, because the store service is the only process allowed to touch that directory.

Note the FTS rebuild. `create_fts_index()` is deprecated in current LanceDB and `use_tantivy=True` raises; `create_index("text", config=FTS(), replace=True)` is the working form. Skip it until at least one row exists, or it fails on an empty table.

## Hybrid search is the reason to use it

This is the part that convinced me. One query, both signals:

```python
query = table.search(
    query_type="hybrid",
    vector_column_name="vector",
    fts_columns="text",
).vector(query_vector).text(question).rerank(RRFReranker())
hits = query.limit(5).to_list()
```

Vector-only search misses exact terms like error codes and product names; keyword-only search misses paraphrases. Hybrid with RRF handles both without me writing a score-normalisation function. Results come back with `_relevance_score` from the reranker (or `_distance` for plain vector search); normalising that into a single `{score, source, text, metadata}` shape keeps the rest of the app simple.

I track an `fts_ready` flag per workspace. If the index is missing or unusable, hybrid degrades to vector-only rather than returning a 500. At this scale — tens of thousands of chunks at most — there is no ANN index at all; a flat scan is fine. IVF_PQ is a problem for later.

## The catch: it is embedded

Because LanceDB is a library, the process that writes the directory **is** the database. On a local filesystem there is no commit arbitration between concurrent writers, so exactly two things are safe: one writer, or corruption. That constraint generates the architecture:

```plaintext
  browser ──▶ ui (Streamlit) ──▶ api (FastAPI) ──▶ store (FastAPI)
                                      │                 │
                                      │                 └─ PVC /data
                                      │                    ├─ lancedb/  tables + FTS
                                      │                    └─ auth.db   SQLite
                                      └─▶ OpenAI-compatible LLM on the host
```

The **store** is a small FastAPI service I wrote whose only job is to own that directory. One replica, one uvicorn worker, one `asyncio.Lock` around every write. The Helm chart refuses to render if `store.replicas != 1`. The **api** chunks, embeds, builds prompts and streams SSE; it never imports LanceDB. The **ui** calls the api server-side, so there is no browser CORS problem.

A `meta.json` beside the directory records the embedding model and dimension. On startup the store compares it to the config and fails fast with a "re-ingest required" message. Changing `EMBEDDING_MODEL` without re-ingesting silently poisons a vector index; this is a cheap guard against that.

## One table per workspace

The second round of work added users, login and workspaces. The obvious design is one big table with a `workspace_id` column and a `where` filter on every query. I did not do that. Each workspace gets its own table, named `ws_<uuid>`:

```python
table_name = f"ws_{uuid.uuid4().hex}"   # generated, never from user input
table = db.create_table(table_name, schema=schema())
```

Isolation becomes structural: a search literally cannot see another workspace's rows, so there is no filter to forget and no ACL bug to audit. Each table has its own FTS index and its own `fts_ready` flag. The cost is that creating a workspace creates a table and deleting one drops it — both cheap at this scale.

Metadata lives in SQLite (`/data/auth.db`) on the same PVC: users, workspaces, memberships, sessions. No new workload, which keeps the deployment at three containers. Sessions are opaque random tokens stored as SHA-256 hashes with a 12-hour TTL; passwords use the standard library's `scrypt`, so there was no dependency to add. A bootstrap admin comes from a Helm Secret on first start.

Permissions are workspace-level, enforced in the store:

```python
REQUIRED_ROLE = {
    "read": "viewer", "write": "editor", "reindex": "admin",
    "manage_members": "admin", "delete_workspace": "owner",
}

def require_access(workspace_id, user, action):
    role = membership(workspace_id, user) or "none"
    if RANK.get(role, 0) < RANK[REQUIRED_ROLE[action]]:
        raise HTTPException(403, f"role '{role}' cannot {action}")
```

The api just forwards the caller's `Authorization: Bearer` token as an internal header and maps `401/403/404` back unchanged. Two subtleties were worth the extra code:

- Ingest rebuilds the FTS index as part of the final write, so an `editor` can ingest without needing the `admin`-only reindex permission.
- A workspace `admin` cannot grant `owner`, demote an owner, or remove a higher-ranked member. Otherwise the role table is an escalation ladder.

The migration was the nicest part: an existing single-table deployment had 17 chunks across 2 sources. On upgrade with an empty metadata database, that `documents` table was adopted as the **default** workspace — no copy, no re-embedding, FTS state carried over. New workspaces get fresh `ws_<uuid>` tables.

## What the API looks like

```bash
TOKEN=$(curl -s -X POST localhost:8080/auth/login \
  -H 'content-type: application/json' \
  -d '{"username":"admin","password":"changeme"}' | jq -r .token)

curl -s -X POST localhost:8080/ws/$WS/search \
  -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"question":"why is the store a separate service?"}'
```

The query endpoint streams `sources`, `token`, `done` and `error` events; the prompt instructs the model to answer only from context and cite `[1]`, `[2]`, and the UI renders those as expandable source cards. The LLM is an OpenAI-compatible server on the host — Ollama, LM Studio, anything that binds `0.0.0.0` so the pod can reach it.

## Local infrastructure war stories

A few things that cost real time and are easy to miss in a polished tutorial:

- The registry listens on **5001**, not 5000, because macOS AirPlay Receiver owns 5000. Skaffold's `default-repo` and `insecure-registries` are persisted during bootstrap.
- `build.platforms: ["linux/arm64"]` overrides a global `DOCKER_DEFAULT_PLATFORM=linux/amd64`; otherwise the k3s nodes get images they cannot run.
- The store runs as uid 1000 with `fsGroup: 1000`; `make verify` proves a non-root pod can actually write the PVC before any app code is deployed.
- `skaffold dev` has no store file-sync, so a hot-reload child process never opens the database twice.
- The bootstrap password is `changeme`, dev-only, used only while the user table is empty.

## Try it and what I verified

The full source is on GitHub: [nf1s/rag-lancedb](https://github.com/nf1s/rag-lancedb).

```bash
make up                 # Colima + k3s + registry + Skaffold config
LLM_PORT=11434 LLM_MODEL=qwen3.8:27b-mlx make config
make run                # build, push, Helm deploy
make smoke              # port-forward + end-to-end check
make test               # 65 tests, no cluster needed
```

`make smoke` logs in, creates a temporary workspace and a viewer user, ingests a document, asserts the retrieved chunk came back, checks the viewer gets `403` on a workspace they are not a member of, then streams an answer. The UI gained a login page, a workspace switcher and an admin panel; the default workspace still reports 17 chunks, 2 sources and a ready FTS index after the migration.

## Limits, honestly

- One embedding model and dimension are shared across all workspaces; per-workspace settings are a reserved column, not a feature yet.
- There is no password-reset UI and no user-delete endpoint. Sessions expire, but an admin cannot yet rotate a password from the UI.
- SQLite and LanceDB both want a single writer. That is fine for one store pod and a local demo; it is the first thing to replace for a real multi-node deployment.
- One table per workspace is lovely at a dozen workspaces and questionable at ten thousand.
- `bge-small-en-v1.5` is English-only and 512-token-max; the 400-token chunk size is a workaround, not a solution.
- No ANN index, no OCR, no PDF or HTML ingestion. Unsupported uploads return a clear "0 chunks" warning.

LanceDB's sweet spot is a RAG index you can reason about completely: a typed table, a hybrid query, a directory you can back up. The price is that *someone* has to own that directory. Once I accepted that and wrote the small store service around it, the rest — streaming, citations, even workspaces and roles — was ordinary application code.

If you are building RAG locally, start with the schema and one hybrid query. The architecture will tell you where it wants to go from there.
