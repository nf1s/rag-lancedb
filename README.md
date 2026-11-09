# RAG on LanceDB — local Kubernetes with Colima + Skaffold

A retrieval-augmented generation pipeline that indexes Markdown and text files into
LanceDB and answers questions with citations. Three workloads run in a local k3s cluster
provided by Colima: a LanceDB store, a FastAPI backend, and a Streamlit UI.

```
ui (Deployment) ──▶ api (Deployment) ──▶ store (StatefulSet, PVC)
                                         └─ LanceDB + native FTS
                              api ──▶ OpenAI-compatible LLM on your Mac
```

## Prerequisites

- macOS Apple Silicon, Colima, Docker CLI, kubectl, Helm, Skaffold v2.10+, `uv`
- An OpenAI-compatible LLM server on the host (LM Studio, Ollama, vLLM, …) listening on
  `0.0.0.0`, not `127.0.0.1`:
  - LM Studio: enable *Serve on Local Network* (port 1234)
  - Ollama: `OLLAMA_HOST=0.0.0.0 ollama serve` (port 11434)
  - vLLM: `--host 0.0.0.0` (port 8000)

## Quickstart

```bash
# 1. Start Colima + k3s, the in-VM registry, k3s mirror config, Skaffold default repo
make up

# 2. Tell the chart where your model lives (defaults: port 1234, qwen2.5-7b-instruct)
LLM_PORT=1234 LLM_MODEL=qwen2.5-7b-instruct make config

# 3. Optional but recommended: prove registry pulls and non-root PVC writes
make verify

# 4. Build, push, deploy, port-forward, and watch for changes
make dev
```

Then open:

- UI: <http://localhost:8501>
- API docs: <http://localhost:8080/docs>

Sign in with the bootstrap account created on first start: `admin` / `changeme`.
Change it in `deploy/helm/rag/values.yaml` (`auth.bootstrapAdmin`) or point
`auth.existingSecret` at a Secret containing `BOOTSTRAP_ADMIN_USER` and
`BOOTSTRAP_ADMIN_PASSWORD`. The bootstrap account is only created when the SQLite
user table is empty.

Upload `data/sample.md` in the UI, then ask *"why is the store a separate service?"*.

One-shot deploy without the dev loop:

```bash
make run            # skaffold run
make delete         # skaffold delete (PVC is preserved)
make clean-data     # delete the LanceDB PVC, with confirmation
make smoke          # run the smoke test (defined as: scripts/smoke-test.sh)
```

`make smoke` logs in as `admin`/`changeme`; override with `ADMIN_USER` and
`ADMIN_PASSWORD` if you changed the bootstrap credentials.

## Workspaces and permissions

Each workspace owns a separate LanceDB table and full-text index. The store is the policy
enforcement point: every workspace-scoped request carries a session token, and the store
checks membership and role before touching the table. Isolation is structural — a search
never sees rows from another workspace.

| Role | Allowed actions |
| --- | --- |
| `viewer` | search, read stats and sources |
| `editor` | + ingest and delete documents |
| `admin` | + reindex and manage members |
| `owner` | + delete the workspace |

Global admins can create users and workspaces and can access every workspace. The UI adds
a login page, a workspace switcher, and an **Admin** tab for global admins and workspace
admins/owners.

The API uses bearer tokens:

```bash
TOKEN=$(curl -s -X POST http://localhost:8080/auth/login \
  -H 'content-type: application/json' \
  -d '{"username":"admin","password":"changeme"}' | jq -r .token)

curl -s http://localhost:8080/workspaces -H "Authorization: Bearer ${TOKEN}"

curl -s -X POST http://localhost:8080/ws/<workspace-id>/search \
  -H "Authorization: Bearer ${TOKEN}" -H 'content-type: application/json' \
  -d '{"question":"why is the store a separate service?"}'
```

Metadata (users, workspaces, memberships, sessions) lives in `/data/auth.db` on the same
PVC as LanceDB. A `default` workspace is created on first start; an existing single-table
deployment is adopted as the default workspace without copying documents.

## Configuration

Chart defaults live in `deploy/helm/rag/values.yaml`; machine-specific values are generated
into `deploy/local.values.yaml` by `make config` (gitignored). The generator resolves the
Mac's address as seen from the Colima VM and points `llm.baseUrl` at it.

Environment variables understood by the services include `EMBEDDING_MODEL`, `EMBEDDING_DIM`,
`RETRIEVAL_MODE`, `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MAX_CONTEXT_TOKENS`, `STORE_URL`,
`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `SESSION_TTL_HOURS`, `BOOTSTRAP_ADMIN_USER`,
and `BOOTSTRAP_ADMIN_PASSWORD`.

## Tests

```bash
make test    # unit/integration tests, no network or cluster required
make lint
./scripts/smoke-test.sh   # against a running deployment
```

## Notes

- LanceDB is embedded and, on a local filesystem, allows a single writer. The store runs
  one replica and one worker; the chart fails to render if `store.replicas != 1`.
- The store also owns `/data/auth.db` (SQLite). It is backed up and restored with the same
  PVC; deleting the PVC removes users, workspaces and documents together.
- Images are built with the local Docker daemon and pushed to `127.0.0.1:5001`, a registry
  running inside the Colima VM. Skaffold's default repo is persisted so bare `skaffold dev`
  also works after bootstrap.
- `scripts/ctr-import.sh` is a fallback custom builder that imports images straight into
  the k3s containerd store if the registry path is unavailable.

## Troubleshooting

**Skaffold logs `image [...] is not used`.** Cosmetic. The chart receives the repository and
tag through `setValueTemplates`, while Skaffold checks for the digest-qualified reference.
`kubectl -n rag get pods -o jsonpath='{..image}'` confirms the pods run the built tag.

**Registry push works but Skaffold's digest check returns 403.** On macOS the AirPlay
Receiver owns port 5000; the registry is published on 5001 for that reason. If you changed
`REGISTRY_PORT`, update `skaffold config set default-repo`/`insecure-registries` too.

**Health shows `llm.reachable: false`.** The backend runs in the VM, so the server must bind
`0.0.0.0`. Verify from the VM:
`colima ssh -- curl http://192.168.5.2:11434/v1/models` (adjust port).

**The store pod restarts after a config change.** Expected: `EMBEDDING_MODEL` and
`EMBEDDING_DIM` are checked against `meta.json` on startup. Re-ingest the corpus after
changing either, or the store reports an embedding mismatch.

See `PLAN.md` for the full design and rationale.
