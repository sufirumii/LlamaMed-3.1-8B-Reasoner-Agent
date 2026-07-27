"""Thin wrapper around sentence-transformers so the rest of the codebase
doesn't need to know which embedding model is in use.

Default is all-MiniLM-L6-v2: small (~80MB), CPU-friendly, and good enough
for retrieval over a personal/clinical-reference PDF collection. Swap via
rag.embedding_model in config.yaml if you want a stronger model.
"""

from __future__ import annotations

from typing import List

import numpy as np


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return vecs.astype("float32")
