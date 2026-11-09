#!/usr/bin/env bash
# Fallback Skaffold custom builder for Colima: build with Docker, then import the image
# straight into the k3s containerd store.
#
# Usage in skaffold.yaml:
#   build:
#     artifacts:
#       - image: rag-api
#         custom:
#           buildCommand: ./scripts/ctr-import.sh api
#
# Skaffold exports IMAGE for the artifact; PUSH_IMAGE must be false.
set -euo pipefail
cd "$(dirname "$0")/.."

target="${1:?usage: ctr-import.sh <store|api|ui>}"
: "${IMAGE:?IMAGE is set by Skaffold}"

docker build --target "${target}" -t "${IMAGE}" .
docker save "${IMAGE}" | colima ssh -- sudo k3s ctr images import -
