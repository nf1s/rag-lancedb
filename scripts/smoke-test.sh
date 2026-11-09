#!/usr/bin/env bash
# End-to-end smoke test against the port-forwarded API.
set -euo pipefail
cd "$(dirname "$0")/.."

API_URL="${API_URL:-http://localhost:8080}"
ADMIN_USER="${ADMIN_USER:-admin}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-changeme}"
SLUG="smoke-$(date +%s)"
VIEWER_USER="smoke-viewer"
VIEWER_PASSWORD="smoke-password-123"

json_field() {
  python3 -c "import json,sys; print(json.load(sys.stdin)$1)"
}

echo "==> health"
health="$(curl -fsS "${API_URL}/healthz")"
echo "${health}" | head -c 300
echo

echo "==> login as ${ADMIN_USER}"
login="$(curl -fsS -X POST "${API_URL}/auth/login" \
  -H 'content-type: application/json' \
  -d "{\"username\":\"${ADMIN_USER}\",\"password\":\"${ADMIN_PASSWORD}\"}")"
TOKEN="$(echo "${login}" | json_field "['token']")"
AUTH="Authorization: Bearer ${TOKEN}"
default_ws="$(echo "${login}" | python3 -c \
  'import json,sys; ws=json.load(sys.stdin)["workspaces"]; print(next(w["id"] for w in ws if w["slug"] == "default"))')"
echo "logged in; default workspace ${default_ws}"

echo "==> create workspace ${SLUG}"
workspace="$(curl -fsS -X POST "${API_URL}/workspaces" \
  -H "${AUTH}" -H 'content-type: application/json' \
  -d "{\"slug\":\"${SLUG}\",\"name\":\"Smoke ${SLUG}\"}")"
WS="$(echo "${workspace}" | json_field "['id']")"
echo "workspace ${WS}"

cleanup() {
  curl -sS -X DELETE "${API_URL}/workspaces/${WS}" -H "${AUTH}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> create viewer ${VIEWER_USER}"
code="$(curl -sS -o /dev/null -w '%{http_code}' -X POST "${API_URL}/users" \
  -H "${AUTH}" -H 'content-type: application/json' \
  -d "{\"username\":\"${VIEWER_USER}\",\"password\":\"${VIEWER_PASSWORD}\"}")"
if [[ "${code}" != "201" && "${code}" != "409" ]]; then
  echo "user creation failed: ${code}" >&2
  exit 1
fi
curl -fsS -X POST "${API_URL}/workspaces/${WS}/members" \
  -H "${AUTH}" -H 'content-type: application/json' \
  -d "{\"username\":\"${VIEWER_USER}\",\"role\":\"viewer\"}" >/dev/null

viewer_login="$(curl -fsS -X POST "${API_URL}/auth/login" \
  -H 'content-type: application/json' \
  -d "{\"username\":\"${VIEWER_USER}\",\"password\":\"${VIEWER_PASSWORD}\"}")"
VIEWER_TOKEN="$(echo "${viewer_login}" | json_field "['token']")"
VIEWER_AUTH="Authorization: Bearer ${VIEWER_TOKEN}"

echo "==> ingest data/sample.md into ${SLUG}"
curl -fsS -X POST "${API_URL}/ws/${WS}/ingest" -H "${AUTH}" \
  -F "files=@data/sample.md;type=text/markdown"
echo

echo "==> retrieval (deterministic)"
response="$(curl -fsS -X POST "${API_URL}/ws/${WS}/search" \
  -H "${AUTH}" -H 'content-type: application/json' \
  -d '{"question":"why is the store a separate service?","top_k":3}')"
echo "${response}" | head -c 400
echo
echo "${response}" | grep -q 'sample.md'
echo "retrieval ok"

echo "==> workspace isolation"
code="$(curl -sS -o /dev/null -w '%{http_code}' \
  -X GET "${API_URL}/ws/${WS}/stats" -H "${VIEWER_AUTH}")"
if [[ "${code}" != "200" ]]; then
  echo "viewer could not read its workspace: ${code}" >&2
  exit 1
fi
echo "viewer can read the smoke workspace"

code="$(curl -sS -o /dev/null -w '%{http_code}' \
  -X GET "${API_URL}/ws/${default_ws}/stats" -H "${VIEWER_AUTH}")"
if [[ "${code}" != "403" ]]; then
  echo "isolation failed: expected 403 for the default workspace, got ${code}" >&2
  exit 1
fi
echo "viewer cannot read the default workspace"

echo "==> generation (needs the host LLM)"
query_output="$(curl -fsSN -X POST "${API_URL}/ws/${WS}/query" \
  -H "${AUTH}" -H 'content-type: application/json' \
  -d '{"question":"why is the store a separate service?","top_k":3}' || true)"
echo "${query_output}" | head -c 400
echo
if echo "${query_output}" | grep -q 'event: token'; then
  echo "generation ok"
else
  echo "WARNING: no token events; check that the LLM is reachable at the configured base URL" >&2
fi

echo "==> all checks passed"
