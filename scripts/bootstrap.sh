#!/usr/bin/env bash
# Idempotent local cluster bootstrap: Colima + k3s + local registry + Skaffold defaults.
set -euo pipefail
cd "$(dirname "$0")/.."

REGISTRY_PORT="${REGISTRY_PORT:-5001}"
REGISTRY="127.0.0.1:${REGISTRY_PORT}"

if ! colima status >/dev/null 2>&1; then
  echo "==> starting colima (kubernetes)"
  # No --disk flag: Colima will not shrink an existing instance's disk.
  colima start --kubernetes --cpu 6 --memory 12
else
  echo "==> colima already running"
fi

docker context use colima >/dev/null
kubectl config use-context colima >/dev/null

echo "==> local registry"
if ! docker inspect registry >/dev/null 2>&1; then
  docker run -d --restart unless-stopped -p "${REGISTRY_PORT}:5000" \
    -v rag-registry:/var/lib/registry --name registry registry:2
else
  docker start registry >/dev/null 2>&1 || true
fi

REGISTRIES_CONFIG="mirrors:
  \"127.0.0.1:${REGISTRY_PORT}\":
    endpoint:
      - \"http://127.0.0.1:${REGISTRY_PORT}\""

current="$(colima ssh -- sudo cat /etc/rancher/k3s/registries.yaml 2>/dev/null || true)"
if [[ "${current}" != "${REGISTRIES_CONFIG}" ]]; then
  echo "==> configuring k3s for the insecure local registry"
  printf '%s\n' "${REGISTRIES_CONFIG}" \
    | colima ssh -- sudo tee /etc/rancher/k3s/registries.yaml >/dev/null
  colima ssh -- sudo systemctl restart k3s
  kubectl wait --for=condition=Ready node --all --timeout=240s
else
  echo "==> k3s registry config unchanged"
fi

echo "==> skaffold defaults"
skaffold config set default-repo "${REGISTRY}"
skaffold config set insecure-registries "${REGISTRY}"

./scripts/gen-local-values.sh

echo "==> pre-flight"
skaffold diagnose --yaml-only >/dev/null
skaffold config list || true

echo "==> ready: run 'make dev' or 'make run'"
