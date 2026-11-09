SHELL := /bin/bash
export DOCKER_DEFAULT_PLATFORM := linux/arm64
REGISTRY ?= 127.0.0.1:5001
LLM_PORT ?= 1234
LLM_MODEL ?= qwen2.5-7b-instruct

.PHONY: up config dev run delete clean-data test lint verify smoke down logs

up:
	./scripts/bootstrap.sh

config:
	LLM_PORT=$(LLM_PORT) LLM_MODEL=$(LLM_MODEL) ./scripts/gen-local-values.sh

dev: config
	SKAFFOLD_DEFAULT_REPO=$(REGISTRY) skaffold dev

run: config
	SKAFFOLD_DEFAULT_REPO=$(REGISTRY) skaffold run

delete:
	skaffold delete

clean-data:
	@read -p "Delete the LanceDB PVC data-rag-store-0? [y/N] " answer; \
	if [[ "$$answer" == "y" || "$$answer" == "Y" ]]; then \
		kubectl -n rag delete pvc data-rag-store-0 --ignore-not-found; \
	else \
		echo "aborted"; \
	fi

test:
	uv run pytest

lint:
	uv run ruff check src ui tests

verify:
	./scripts/verify-first.sh

smoke:
	@kubectl -n rag port-forward svc/rag-api 8080:8080 >/tmp/rag-port-forward.log 2>&1 & \
	pf=$$!; sleep 2; ./scripts/smoke-test.sh; status=$$?; kill $$pf; exit $$status

down:
	colima stop

logs:
	kubectl -n rag logs -f statefulset/rag-store
