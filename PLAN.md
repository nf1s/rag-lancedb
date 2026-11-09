# RAG pipeline on LanceDB — local Kubernetes with Colima + Skaffold

## 1. Goal

A document Q&A pipeline that ingests `.md`/`.txt` files, chunks and embeds them, stores
vectors in LanceDB, and answers questions with source citations through a Streamlit chat UI.
Everything runs locally: Colima provides the Kubernetes cluster, Skaffold builds images and
deploys the Helm chart, and the LLM is an OpenAI-compatible server running on the Mac host.

No cloud services, no API keys required (a dummy key is used for local LLM servers that
insist on a non-empty value).

---

## 2. Environment

Verified on this machine:

| Tool | Version / state |
|---|---|
| macOS | Apple Silicon, 11 CPU, 36 GB RAM, 530 GB free |
| `colima` | 0.10.3, currently **not running** |
| `docker`, `kubectl`, `helm`, `uv` | installed (kubectl 1.32 client, helm 3.17.2) |
| `skaffold` | v2.10.1 (asdf shim); latest config schema is **`skaffold/v4beta9`** |
| Python (host) | 3.11.7 (pyenv) |
| Repo | `/Users/ahmed/projects/rag-lancedb`, empty, not a git repo |

Container target: `python:3.12-slim`, linux/arm64. Use `uv python install 3.12` so local
tests and the container run the same interpreter.

**arm64 wheel availability checked** (no source builds needed):
`lancedb 0.41.0`, `pylance 14.0.2`, `pyarrow 26`, `onnxruntime 1.31.1` all publish
`manylinux_2_28_aarch64` wheels; `python:3.12-slim` (glibc 2.36) satisfies that.
`fastembed 0.9.1` accepts `onnxruntime >=1.17` on Python 3.12 (only 1.20.0 / 1.24.0 /
1.24.1 are excluded).

---

## 3. Decision log

| Topic | Decision |
|---|---|
| Ingest formats | **Markdown and plain text only.** No PDF, no HTML, no URL ingestion |
| Embeddings | fastembed `BAAI/bge-small-en-v1.5`, 384 dim, English, ONNX CPU, in the API pod |
| LLM | OpenAI-compatible server on the Mac host (LM Studio `:1234` default; Ollama `:11434`; vLLM `:8000`). No cloud endpoint |
| Indexing | Vector + **Lance-native FTS** hybrid search with RRF reranking. `tantivy` is **not** used (removed in LanceDB ≥0.25) |
| UI | Streamlit, its own Deployment |
| Topology | Three workloads: `store` (StatefulSet), `api` (Deployment), `ui` (Deployment) |
| Packaging | Helm chart + Skaffold build/deploy |
| Local access | Skaffold `portForward` (primary), Ingress template disabled by default |

### Why a separate "LanceDB deployment" is a service we write

`lancedb.connect(uri)` is always embedded — there is no open-source server binary.
Only LanceDB Cloud/Enterprise (`db://` + API key) is a server. So the `store` pod is a thin
FastAPI service that owns the LanceDB directory and is the **only process** holding the
LanceDB client. The API and UI never import `lancedb`.

This is also a correctness requirement: on a local filesystem there is **no commit
arbitration** (conditional-put / DynamoDB commit handlers only exist for object stores).
Two processes writing the same directory is unsafe. Hence: 1 replica, 1 uvicorn worker,
all writes serialized behind an `asyncio.Lock`.

Benefits: API and UI are stateless and independently scalable; the API can be restarted
without touching data or the writer lock; only one pod needs the PVC.

---

## 4. Architecture

```
  browser
    │  localhost:8501 (skaffold port-forward)
    ▼
┌────────────────────┐  login + Bearer  ┌──────────────────────────┐
│ ui  (Deployment)   │─────────────────▶│ api (Deployment)          │
│ Streamlit          │  /ws/{id}/...    │ FastAPI                   │
│ - login            │                  │ - chunk (400 tok)         │──▶ LLM on Mac host
│ - workspace switch │                  │ - embed (fastembed 384d)  │   http://<host-ip>:1234/v1
│ - ingest / chat    │                  │ - bearer forwarding       │
│ - admin panel      │                  │ - prompt + SSE streaming  │
└────────────────────┘                  └────────────┬─────────────┘
                                                     │ HTTP + X-Rag-Token
                                                     ▼
                              ┌─────────────────────────────────────┐
                              │ store (StatefulSet, 1)              │
                              │ FastAPI, 1 worker                   │
                              │ - users/workspaces/sessions (SQLite)│
                              │ - RBAC enforcement                  │
                              │ - one LanceDB table + FTS per ws    │
                              │ - write lock                        │
                              │ - PVC /data (lancedb + auth.db)     │
                              └─────────────────────────────────────┘

  localhost:8080 (skaffold port-forward) → api /docs, /healthz, /auth/login
```

The UI calls the API at `http://rag-api:8080` from the Streamlit Python process (server-side
HTTP), so there is no browser CORS concern.

---

## 5. Data model and retrieval

### Tables (one per workspace)

Workspaces are isolated structurally: each owns a LanceDB table named `ws_<uuid>` with its
own FTS index. The name is generated server-side and recorded in SQLite; it is never derived
from user input. On upgrade, the legacy `documents` table is adopted as the `default`
workspace without copying data. No query needs a `workspace_id` filter.

| Column | Type | Notes |
|---|---|---|
| `id` | string | sha256 of `workspace_id + source + chunk_index + text` |
| `source` | string | original filename / relative path |
| `chunk_index` | int | position in the source |
| `text` | string | FTS-indexed column |
| `vector` | `fixed_size_list<float32>[384]` | normalized bge embedding |
| `metadata` | JSON | heading, mtime, char offsets |

### Chunking and embedding

- Recursive splitter, markdown-heading aware: **400 tokens, 50 overlap**. `bge-small-en-v1.5`
  truncates at 512 tokens, so 800-token chunks would silently lose ~35% of the content.
- `fastembed` `query_embed()` for queries, `embed()` / `passage_embed()` for documents;
  no manual prefix required for bge v1.5.
- Embedding model name + dim + version recorded in `meta.json` beside the DB directory in
  the store volume. Startup fails with a clear "re-ingest required" message on mismatch.
- Vector dim is sent with every write and validated by the store (400 on mismatch).
- Cosine metric is set explicitly on search (`.metric("cosine")`), not left at LanceDB's
  default L2.

### Ingest semantics (important)

`POST /upsert` is a **replace-by-source transaction**, not `merge_insert` on a hash:

1. acquire write lock
2. `delete where source = X`
3. `add(new_rows)`
4. release lock, then rebuild the FTS index if requested

This removes stale chunks when a file shrinks and is idempotent under retries (the API's
store client retries on network errors).

### FTS / hybrid search

- `table.create_index("text", config=FTS())`, with `replace=True` after ingest batches.
  `create_fts_index()` is deprecated and `use_tantivy=True` now raises.
- Skip index creation until there is at least one row; fall back to vector-only search if
  the index is absent (e.g. empty corpus).
- `query_type="hybrid"` + `RRFReranker()`. `/search` returns a normalized
  `{score, source, chunk_index, text, metadata}` shape regardless of whether results came
  from vector (`_distance`) or reranked hybrid (`_relevance_score`).
- No ANN/vector index at demo scale (flat scan is fine); add IVF_PQ past ~100k chunks.
- `top_k = 5` → ~2,000 context tokens. `maxContextTokens` (default 3000) trims the prompt
  so a larger top-k cannot overflow a local 4k-context model.
- `read_consistency_interval` is unnecessary: the store is the only process reading or
  writing the table.

---

## 6. Service contracts

### `store` (cluster-internal only, port 8081)

| Endpoint | Body / result |
|---|---|
| `GET /healthz` | unauthenticated readiness; workspace count |
| `POST /auth/login`, `/auth/logout`, `GET /auth/me` | opaque session tokens, hashed at rest |
| `GET/POST /users` | list/create users (global admin) |
| `GET/POST /workspaces` | list visible workspaces / create one (global admin) |
| `DELETE /workspaces/{id}` | owner or global admin; drops the table |
| `GET/POST/DELETE /workspaces/{id}/members` | workspace admin or global admin |
| `GET /ws/{id}/stats` | read role |
| `POST /ws/{id}/upsert` | editor role; replace-by-source + add, under write lock |
| `POST /ws/{id}/search` | viewer role; normalized hits |
| `POST /ws/{id}/delete` | editor role; remove all chunks for a source |
| `POST /ws/{id}/reindex` | workspace admin; rebuild the FTS index |

All LanceDB calls run through `asyncio.to_thread` (or `connect_async`) so the sync client
never blocks the event loop. Every workspace-scoped endpoint resolves the caller from the
`X-Rag-Token` header and checks the shared RBAC matrix before touching the table.

### `api` (port 8080)

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | 503 until store is reachable; also reports LLM reachability |
| `POST /auth/login`, `/auth/logout`, `GET /auth/me` | proxy to the store |
| `GET/POST /users`, `/workspaces`, members | admin proxy, forwards the bearer token |
| `GET /ws/{id}/stats` | proxy of store stats |
| `POST /ws/{id}/ingest` | multipart `.md`/`.txt` → chunk → embed → upsert → reindex |
| `POST /ws/{id}/search` | embed the question and proxy the search |
| `POST /ws/{id}/query` | SSE stream: `event: token`, `event: sources`, `event: done`, `event: error` |
| `GET /docs` | Swagger UI |

Missing or invalid bearer tokens return `401`; store `403`/`404` responses are forwarded
unchanged, so the UI can show the real reason. Tokens are never stored in environment
variables or URLs.

`POST /ws/{id}/query` falls back to non-streaming completion if the host server lacks streaming.
System prompt instructs the model to answer only from context and cite sources.

### `ui` (port 8501)

Streamlit app with a login page, a sidebar workspace switcher, and tabs for **Ingest**,
**Chat**, and **Admin** (visible to global admins and workspace admins/owners). The admin
tab creates users/workspaces, manages members, and deletes workspaces with confirmation.
Every call forwards `Authorization: Bearer <token>`; the server-side Streamlit process makes
the API calls, so no browser CORS change is needed. Run with
`--server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false
--server.runOnSave=true`; readiness on `/_stcore/health`.

---

## 7. Repository layout

```
PLAN.md
pyproject.toml  uv.lock  .python-version  .gitignore
skaffold.yaml
Dockerfile                       # targets: store, api, ui
Makefile
src/
  common/{config,models,chunking}.py    # shared pydantic schemas + splitter
  store/main.py                        # LanceDB owner
  api/{main,embeddings,llm,store_client}.py
ui/app.py
deploy/
  helm/rag/
    Chart.yaml  values.yaml
    templates/
      _helpers.tpl
      store-statefulset.yaml  store-service.yaml
      api-deployment.yaml     api-service.yaml
      ui-deployment.yaml      ui-service.yaml
      configmap.yaml          ingress.yaml   # ingress disabled
  local.values.yaml                        # generated, gitignored
scripts/
  bootstrap.sh  gen-local-values.sh  ctr-import.sh
  smoke-test.sh  verify-first.sh
tests/
data/                                    # sample .md files for the smoke test
```

### Images (three targets from one multi-stage Dockerfile)

| Target | Dependencies | Approx. |
|---|---|---|
| `store` | `lancedb`, `pyarrow`, `pylance`, `fastapi`, `uvicorn`, `orjson` | ~450 MB |
| `api` | `fastembed`, `onnxruntime`, `httpx`, `fastapi`, `uvicorn` (no lancedb/pyarrow) | ~450 MB incl. baked model |
| `ui` | `streamlit`, `httpx` | ~150 MB |

The bge-small model is downloaded at image build time into `/models` via
`FASTEMBED_CACHE_PATH`, avoiding cold-start downloads in the cluster. All images run
non-root with `securityContext.fsGroup`.

---

## 8. Kubernetes resources

Namespace `rag`.

| Workload | Kind | Replicas | Volume | Requests → Limits |
|---|---|---|---|---|
| `store` | **StatefulSet** | 1 (enforced by Helm `fail` if >1) | `volumeClaimTemplates`, 5Gi, `local-path` | 200m/512Mi → 1Gi |
| `api` | Deployment | 1 (scalable) | none | 500m/1Gi → 2Gi (ONNX) |
| `ui` | Deployment | 1 (scalable) | none | 100m/256Mi → 512Mi |

Why a StatefulSet for the store: stable identity, `volumeClaimTemplates`, and a rolling
update that removes the old pod before creating the new one — no two pods ever mount the
RWO volume simultaneously.

Other resources: `ConfigMap` (non-secret config), ClusterIP services for all three,
readiness/liveness probes, optional `wait-for-store` init container on the API. No Secret is
needed by default; optional `llm.existingSecret` for servers that require a key.

**PVC lifecycle:** `volumeClaimTemplates` PVCs are not owned by Helm and survive
`skaffold delete` / `helm uninstall` (data is preserved — desirable, but hidden). Provide
`make clean-data` with an explicit confirmation and document
`kubectl delete pvc -n rag data-rag-store-0`. Verify non-root writes to `local-path` host
dirs at implementation time; add `fsGroup` and/or a chown init container if needed.

Ingress template ships with `ingress.enabled: false`; Skaffold `portForward` is the primary
access path.

---

## 9. Skaffold configuration

```yaml
apiVersion: skaffold/v4beta9
kind: Config
metadata:
  name: rag-lancedb

build:
  tagPolicy: { dateTime: {} }          # no git dependency
  local:
    push: true
    useDockerCLI: true
  artifacts:
    - image: rag-store
      context: .
      docker: { dockerfile: Dockerfile, target: store }
      # no file sync: store changes rebuild the image, so the single-writer
      # process is never replaced by a hot-reload child
    - image: rag-api
      context: .
      docker: { dockerfile: Dockerfile, target: api }
      sync:
        manual:
          - { src: "src/api/**",    dest: /app/src/api }
          - { src: "src/common/**", dest: /app/src/common }
    - image: rag-ui
      context: .
      docker: { dockerfile: Dockerfile, target: ui }
      sync:
        manual:
          - { src: "ui/**", dest: /app/ui }

deploy:
  kubeContext: colima
  helm:
    releases:
      - name: rag
        chartPath: deploy/helm/rag
        namespace: rag
        createNamespace: true
        valuesFiles:
          - deploy/helm/rag/values.yaml
          - deploy/local.values.yaml
        setValueTemplates:
          store.image.repository: "{{.IMAGE_REPO_rag_store}}"
          store.image.tag:        "{{.IMAGE_TAG_rag_store}}"
          api.image.repository:   "{{.IMAGE_REPO_rag_api}}"
          api.image.tag:          "{{.IMAGE_TAG_rag_api}}"
          ui.image.repository:    "{{.IMAGE_REPO_rag_ui}}"
          ui.image.tag:           "{{.IMAGE_TAG_rag_ui}}"
        wait: true

portForward:
  - { resourceType: service, resourceName: rag-api, namespace: rag, port: 8080, localPort: 8080 }
  - { resourceType: service, resourceName: rag-ui,  namespace: rag, port: 8501, localPort: 8501 }
```

Key points:

- **Local registry** published on `127.0.0.1:5001` (container port 5000) running as a
  Docker container inside the Colima VM. `127.0.0.1` (not `localhost`) avoids containerd
  resolving `::1` where docker-proxy only listens on IPv4. It works from both the Mac (via
  Colima port forwarding) and the VM.
  - **Port 5000 is unusable on macOS**: the AirPlay Receiver (`ControlCenter`, server
    `AirTunes`) already binds it, so host-side requests reach AirTunes instead of the
    registry. Skaffold's post-push digest lookup then fails with `403 Forbidden`. Publishing
    on 5001 fixes this without touching system settings.
- `skaffold config set default-repo 127.0.0.1:5001` and `insecure-registries` are persisted
  for the `colima` kube-context during bootstrap, **and** `SKAFFOLD_DEFAULT_REPO` is
  exported in Makefile targets. Without this, a bare `skaffold dev` would try to push to
  Docker Hub, and digest checks would attempt TLS against the plain-HTTP registry.
- `imagePullPolicy: Always` in chart values — cheap on a local registry, prevents stale
  same-tag images.
- `setValueTemplates` is the supported way to inject artifact repo/tag into a Helm chart.
- **Verified against the installed binary:** Skaffold v2.10.1's latest schema is
  `skaffold/v4beta9` (`skaffold fix` upgrades a `v4beta1` config to `v4beta9`).
  `skaffold/v4beta13` is rejected with `unknown skaffold config API version`.
  A config using every field above (`tagPolicy.dateTime`, `local.push/useDockerCLI`,
  `docker.target`, `sync.manual`, `deploy.kubeContext`, `helm.releases.valuesFiles`,
  `setValueTemplates`, `wait`, `portForward`) parses cleanly at `v4beta9`.
- Still run `skaffold diagnose` and `skaffold config list` as pre-flight assertions; if
  Skaffold is upgraded later, `skaffold fix` will migrate the schema.

---

## 10. Bootstrap and dev workflow

`scripts/bootstrap.sh` — idempotent:

```sh
colima status >/dev/null || colima start --kubernetes --cpu 6 --memory 12 --disk 60
docker context use colima
docker inspect registry >/dev/null || \
  docker run -d --restart unless-stopped -p 5001:5000 \
    -v rag-registry:/var/lib/registry --name registry registry:2
# only restart k3s if the file changed:
printf 'mirrors:\n  "127.0.0.1:5001":\n    endpoint: ["http://127.0.0.1:5001"]\n' \
  | colima ssh -- sudo tee /etc/rancher/k3s/registries.yaml
colima ssh -- sudo systemctl restart k3s
skaffold config set default-repo 127.0.0.1:5001
skaffold config set insecure-registries 127.0.0.1:5001
./scripts/gen-local-values.sh
skaffold diagnose
```

`scripts/gen-local-values.sh` resolves the Mac's address as seen from the VM and writes the
gitignored `deploy/local.values.yaml`:

```sh
HOST_IP=$(colima ssh -- getent hosts host.lima.internal | awk '{print $1}')
# LLM_PORT (default 1234) and LLM_MODEL from env
# -> llm.baseUrl: http://<HOST_IP>:<LLM_PORT>/v1
```

Regenerate this on **every** `make dev` / `make run`: the host IP can change across
`colima stop` / `colima start`. Pre-flight fails with a clear message if the file is missing.

| Make target | Action |
|---|---|
| `make up` | bootstrap, generate values |
| `make config` | regenerate `deploy/local.values.yaml` only |
| `make dev` | `skaffold dev` (build + deploy + sync + port-forward, watches changes) |
| `make run` | `skaffold run` (one-shot) |
| `make delete` | `skaffold delete` |
| `make clean-data` | delete the store PVC, with confirmation |
| `make test` | `uv run pytest` |
| `make verify` | prove registry pulls and non-root PVC writes |
| `make smoke` | port-forward the API and run the end-to-end smoke test |
| `make down` | `colima stop` |

`skaffold dev` works standalone after bootstrap because `default-repo` is persisted.
API and UI changes are synced into running pods (uvicorn `--reload`, Streamlit
`--server.runOnSave=true`), so dependency changes are the main trigger for a rebuild.
The store is deliberately **not** synced: Skaffold rebuilds and rolls the single-writer
StatefulSet instead, so the database is never opened by a hot-reload child process.

---

## 11. Configuration values

`deploy/helm/rag/values.yaml` (defaults):

```yaml
embedding:
  model: "BAAI/bge-small-en-v1.5"
  dim: 384
retrieval:
  topK: 5
  mode: hybrid            # vector | hybrid
  chunkSize: 400
  chunkOverlap: 50
  maxContextTokens: 3000
llm:
  baseUrl: "http://localhost:1234/v1"   # overridden by deploy/local.values.yaml in-cluster
  model: "qwen2.5-7b-instruct"          # example; set LLM_MODEL for your host server
  apiKey: "lm-studio"                   # dummy; most local servers want a non-empty value
  timeoutSeconds: 120
auth:
  bootstrapAdmin:
    username: admin
    password: changeme        # used only before the first user exists
  existingSecret: ""          # Secret with BOOTSTRAP_ADMIN_USER / BOOTSTRAP_ADMIN_PASSWORD
  sessionTtlHours: 12
store:
  replicas: 1
  workers: 1
  dataPath: /data/lancedb
  storage: 5Gi
api:  { replicas: 1, devReload: false }
ui:   { replicas: 1, devReload: false }
imagePullPolicy: Always
ingress:
  enabled: false
ollama:                 # optional fallback if host networking proves troublesome
  enabled: false
```

`LLM_BASE_URL` / `LLM_MODEL` are also available as plain environment variables for running
on the host without Kubernetes.

Startup behavior: API probes `GET {llm.baseUrl}/models` and logs available models (warn, do
not fail — some servers do not implement it). `/healthz` returns 503 until the store answers.

---

## 12. Host LLM requirements

- The server must bind `0.0.0.0`, not `127.0.0.1`:
  LM Studio → *Serve on Local Network*; Ollama → `OLLAMA_HOST=0.0.0.0`;
  vLLM → `--host 0.0.0.0`.
- macOS firewall may prompt for an inbound exception from the Colima VM.
- Pod → host traffic relies on flannel masquerading; the smoke test verifies connectivity
  with a `curl` to `/v1/models` from inside the API pod via `kubectl exec`.
- Optional chart toggle `ollama.enabled=true` runs the LLM in-cluster instead, using the
  same OpenAI-compatible client (`http://ollama:11434/v1`).

---

## 13. Testing and verification

- **Unit**: chunking (markdown headings, overlap, size bound), store add → search → stats
  against a temp directory, replace-by-source removes stale chunks, hybrid fallback when FTS
  is absent, dim mismatch rejection, scrypt roundtrip, and the full role/action matrix.
- **Auth and isolation**: login success/failure, missing/invalid tokens, user and workspace
  creation, member management, viewer/editor/admin/owner enforcement, same source name in
  two workspaces, deleted workspaces, duplicate slugs, bootstrap admin, legacy `documents`
  adoption, and per-workspace FTS state.
- **API**: FastAPI `TestClient` with a stub LLM and an in-process fake store; asserts bearer
  forwarding, workspace paths, SSE events, and forwarded 401/403 responses.
- **Deterministic smoke test** (`scripts/smoke-test.sh`): health → login → create a temporary
  workspace → create a viewer → ingest `data/sample.md` → `POST /search` and assert the
  ingested chunk comes back → assert the viewer can read its workspace but gets `403` on the
  default workspace → `POST /query` and assert 200 + at least one token event (generation
  quality is *not* asserted, since it depends on whatever model is on the host).
- **`scripts/verify-first.sh`** — two things to prove before writing app code:
  1. containerd can pull from `127.0.0.1:5001` (throwaway `busybox` Deployment)
  2. a non-root pod can write to the `local-path` PVC
- Pre-flight assertions: `skaffold diagnose`, `skaffold config list`,
  `deploy/local.values.yaml` exists, `curl $LLM_BASE_URL/models` from the API pod.

---

## 14. Known limits and mitigations

| Limit | Mitigation |
|---|---|
| LanceDB is embedded; only one writer is safe on local FS | Single store pod, 1 worker, write lock, Helm `fail` if `replicas != 1`; storage URI kept configurable so MinIO/S3 or Enterprise is a later drop-in |
| `bge-small-en-v1.5` is English-only, 512-token max | 400-token chunks; model/dim recorded in `meta.json`; changing the model fails fast with a re-ingest message |
| FTS index must be rebuilt after writes | `POST /reindex` after each batch; noted as a cost to revisit past ~100k chunks |
| Extra HTTP hop API → store | ~1–3 ms locally, few KB of JSON; acceptable. The store shrinks to a shim if a real remote DB replaces it |
| `skaffold delete` leaves the PVC | Intentional (data preserved); `make clean-data` + docs |
| Host LLM availability | Health check reports it; optional in-cluster Ollama toggle |
| No OCR, no HTML, no PDF | Explicitly out of scope; API returns a clear "0 chunks" warning for empty text |
| Local registry restart ordering | `--restart unless-stopped`; bootstrap verifies registry reachability before deploy |

---

## 15. Workspaces and permissions

- **Model**: users + per-workspace roles. Roles are `viewer` (read), `editor` (+ ingest and
  delete sources), `admin` (+ reindex and manage members), and `owner` (+ delete the
  workspace). A global `admin` flag can create users/workspaces and access everything.
  Permissions are workspace-level; there are no per-document ACLs.
- **Isolation**: one LanceDB table per workspace, named `ws_<uuid>` and recorded in SQLite.
  Table names never come from user input, so a request cannot name a table. Search cannot
  cross tables, which makes isolation structural rather than filter-based.
- **Enforcement**: the store is the policy enforcement point. It validates an opaque bearer
  token on every workspace-scoped call, resolves the workspace, and checks the matrix in
  `src/common/permissions.py`. The API proxies the token as `X-Rag-Token` and maps store
  errors; it does not decide access on its own.
- **Sessions and passwords**: tokens are `secrets.token_urlsafe(32)`, stored as SHA-256
  hashes with a 12h TTL and a revocation flag. Passwords use the standard library's scrypt
  (no new runtime dependency), with a self-describing `scrypt$N$r$p$salt$digest` format.
- **Metadata**: `/data/auth.db` (SQLite, WAL) on the same PVC: users, workspaces,
  memberships, sessions. `workspaces.settings_json` holds `fts_ready` and is the extension
  point for future per-workspace embedding/chunk settings.
- **Migration**: on the first start with an empty metadata database, a bootstrap admin is
  created and the existing `documents` table becomes the `default` workspace. No data is
  copied, and its FTS state is carried over.
- **Bootstrap**: `auth.bootstrapAdmin.{username,password}` in Helm values, or
  `auth.existingSecret` containing `BOOTSTRAP_ADMIN_USER` and `BOOTSTRAP_ADMIN_PASSWORD`.
  Bootstrap values are used only while the user table is empty; they do not reset a
  password on restart.
- **Limits**: one embedding model/dim is shared by all workspaces; global admins can read
  every workspace; deleting a workspace drops its table; there is no password-reset UI yet
  (use a future admin endpoint). Streamlit keeps the session token in server-side session
  state, not in a cookie or URL.

---

## 16. Implementation order

1. **Foundation** — `uv` scaffold (Python 3.12), `common` schemas + chunking, `store`
   service with native FTS, `meta.json`, tests. Run locally with `uv run store`.
2. **API + UI** — fastembed wrapper, LLM client with SSE, `store_client` with timeouts and
   retries, `POST /ingest`, `POST /query`, Streamlit app. Test with a stub LLM.
3. **Containers** — Dockerfile `store`/`api`/`ui` targets; verify the trio on a Docker
   network with **one named volume on the store only** (the API/UI need no volume).
4. **Colima + Skaffold + Helm** — bootstrap script, `verify-first.sh`, chart, generated
   values, `skaffold dev`, port-forwards, smoke test.
5. **Optional toggles** — in-cluster Ollama, Ingress, cross-encoder reranking, `skaffold
   verify` custom test, backup Job for the PVC.

Each milestone is independently runnable; milestone 4 is the first end-to-end demo.

### Fallbacks

- If the local-registry approach fails, `scripts/ctr-import.sh` implements a Skaffold
  `custom` builder: `docker build -t "$IMAGE"` then
  `docker save "$IMAGE" | colima ssh -- sudo k3s ctr images import -`, with `push: false`.
  If stdin forwarding through `colima ssh` is unreliable, save the tar under `$HOME`
  (mounted into the VM) and import from that path.
- If hybrid FTS proves fussy, `retrieval.mode: vector` is a one-value switch.
