"""Embedding wrapper around fastembed (ONNX, CPU)."""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger("rag.api.embeddings")


class Embedder:
    """Lazily loads the fastembed model so importing the API stays cheap."""

    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self.dim = dim
        self._model: Any = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    logger.info("loading embedding model %s", self.model_name)
                    self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def _check(self, vector: list[float]) -> list[float]:
        if len(vector) != self.dim:
            raise RuntimeError(
                f"model {self.model_name} returned {len(vector)} dims, expected {self.dim}. "
                "Update EMBEDDING_DIM together with EMBEDDING_MODEL and re-ingest."
            )
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._ensure().embed(texts)
        return [self._check(_to_list(vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = next(iter(self._ensure().query_embed(text)))
        return self._check(_to_list(vector))


def _to_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        return [float(value) for value in vector.tolist()]
    return [float(value) for value in vector]
