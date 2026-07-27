"""FAISS vector store: a flat inner-product index plus a JSON sidecar
mapping vector row -> {text, source, page}.

Kept deliberately simple (no server, two files on disk) since this is
meant to run on a single local machine: index.faiss + metadata.json in
rag.index_dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        import faiss

        self._faiss = faiss
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []

    def add(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        assert vectors.shape[0] == len(metadata)
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vec: np.ndarray, top_k: int = 4) -> List[Tuple[Dict, float]]:
        if self.index.ntotal == 0:
            return []
        scores, idxs = self.index.search(query_vec.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def save(self, index_dir: str) -> None:
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "metadata": self.metadata}, f)

    @classmethod
    def load(cls, index_dir: str) -> "VectorStore":
        import faiss

        path = Path(index_dir)
        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            payload = json.load(f)
        store = cls(dim=payload["dim"])
        store.index = faiss.read_index(str(path / "index.faiss"))
        store.metadata = payload["metadata"]
        return store

    @staticmethod
    def exists(index_dir: str) -> bool:
        path = Path(index_dir)
        return (path / "index.faiss").exists() and (path / "metadata.json").exists()
