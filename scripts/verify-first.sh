#!/usr/bin/env bash
# Prove the two assumptions that the deployment depends on:
#   1. k3s containerd can pull images from the in-VM registry at 127.0.0.1:5001
#   2. a non-root pod can write to a local-path persistent volume
set -euo pipefail
cd "$(dirname "$0")/.."

export DOCKER_DEFAULT_PLATFORM="${DOCKER_DEFAULT_PLATFORM:-linux/arm64}"
NAMESPACE="${NAMESPACE:-rag}"
BUILDER_IMAGE="busybox:1.36"

kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1 || kubectl create namespace "${NAMESPACE}"

echo "==> 1/2 registry pull"
docker pull -q "${BUILDER_IMAGE}"
docker tag "${BUILDER_IMAGE}" 127.0.0.1:5001/busybox:verify
docker push -q 127.0.0.1:5001/busybox:verify

kubectl -n "${NAMESPACE}" delete pod registry-verify --ignore-not-found >/dev/null
kubectl -n "${NAMESPACE}" run registry-verify \
  --image=127.0.0.1:5001/busybox:verify \
  --restart=Never -- sleep 30 >/dev/null

if kubectl -n "${NAMESPACE}" wait --for=condition=Ready pod/registry-verify --timeout=120s; then
  echo "registry pull ok"
else
  kubectl -n "${NAMESPACE}" describe pod registry-verify || true
  kubectl -n "${NAMESPACE}" delete pod registry-verify --ignore-not-found >/dev/null
  exit 1
fi
kubectl -n "${NAMESPACE}" delete pod registry-verify --ignore-not-found >/dev/null

echo "==> 2/2 non-root PVC write"
cat <<'EOF' | kubectl -n "${NAMESPACE}" apply -f - >/dev/null
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: verify-pvc
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 1Gi
---
apiVersion: v1
kind: Pod
metadata:
  name: pvc-verify
spec:
  restartPolicy: Never
  securityContext:
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
  containers:
    - name: writer
      image: busybox:1.36
      command: ["sh", "-c", "echo ok > /data/verify && ls -l /data"]
      volumeMounts:
        - name: data
          mountPath: /data
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: verify-pvc
EOF

phase=""
for _ in $(seq 1 60); do
  phase="$(kubectl -n "${NAMESPACE}" get pod pvc-verify -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  [[ "${phase}" == "Succeeded" ]] && break
  [[ "${phase}" == "Failed" ]] && break
  sleep 2
done
kubectl -n "${NAMESPACE}" logs pvc-verify || true

kubectl -n "${NAMESPACE}" delete pod pvc-verify --ignore-not-found >/dev/null
kubectl -n "${NAMESPACE}" delete pvc verify-pvc --ignore-not-found >/dev/null

if [[ "${phase}" != "Succeeded" ]]; then
  echo "non-root PVC write FAILED (phase=${phase})" >&2
  exit 1
fi
echo "pvc write ok"
