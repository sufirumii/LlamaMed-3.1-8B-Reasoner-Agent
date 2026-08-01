"""Two kinds of memory, kept deliberately separate from document RAG:

  SessionMemory   -- the running transcript of one chat session, persisted
                      to a JSON file on disk so `chat` can resume where you
                      left off (--session <id>) or you can inspect it with
                      the /history slash command. This is short-term,
                      turn-by-turn state.

  LongTermMemory  -- a small FAISS index (same VectorStore used for
                      documents, pointed at a different directory) of past
                      (query, answer) pairs across *all* sessions. The
                      agent can search it via the `recall_memory` tool, so
                      it can say "you asked about X last week" or reuse a
                      prior calculation instead of redoing the reasoning
                      from scratch -- this is what gives it accuracy that
                      improves the more you use it.

Neither store ever holds identifiable patient information by design (see
guardrails.py, rule 5) -- it holds the user's own questions and the
agent's own answers, not third-party clinical data.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .rag.embeddings import Embedder
from .rag.store import VectorStore


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


class SessionMemory:
    """Persists one chat session's transcript as a JSON file."""

    def __init__(self, sessions_dir: str, session_id: Optional[str] = None):
        self.dir = Path(sessions_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.turns: List[Turn] = []
        if session_id:
            self._load_if_exists()

    @property
    def path(self) -> Path:
        return self.dir / f"{self.session_id}.json"

    def _load_if_exists(self) -> None:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.turns = [Turn(**t) for t in raw.get("turns", [])]

    def add(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))
        self.save()

    def save(self) -> None:
        payload = {
            "session_id": self.session_id,
            "turns": [{"role": t.role, "content": t.content, "timestamp": t.timestamp} for t in self.turns],
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def render_history(self) -> str:
        if not self.turns:
            return "(no turns yet this session)"
        lines = []
        for t in self.turns:
            stamp = time.strftime("%H:%M:%S", time.localtime(t.timestamp))
            lines.append(f"[{stamp}] {t.role}: {t.content}")
        return "\n".join(lines)

    @classmethod
    def list_sessions(cls, sessions_dir: str) -> List[str]:
        path = Path(sessions_dir)
        if not path.exists():
            return []
        return sorted(p.stem for p in path.glob("*.json"))


class LongTermMemory:
    """Semantic recall across all past (query, answer) turns.

    Backed by the same VectorStore/Embedder used for document RAG, kept in
    its own directory so it's never mixed up with attached-PDF content.
    """

    def __init__(self, memory_dir: str, embedding_model: str):
        self.memory_dir = memory_dir
        self.embedding_model = embedding_model
        self._embedder: Optional[Embedder] = None

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.embedding_model)
        return self._embedder

    def remember(self, query: str, answer: str, session_id: str) -> None:
        embedder = self._get_embedder()
        if VectorStore.exists(self.memory_dir):
            store = VectorStore.load(self.memory_dir)
        else:
            store = VectorStore(dim=embedder.dim)

        text = f"Q: {query}\nA: {answer}"
        vec = embedder.embed([text])
        store.add(
            vec,
            [
                {
                    "text": text,
                    "source": f"session:{session_id}",
                    "page": 0,
                    "timestamp": time.time(),
                }
            ],
        )
        store.save(self.memory_dir)

    def recall(self, query: str, top_k: int = 3) -> List[str]:
        if not VectorStore.exists(self.memory_dir):
            return []
        embedder = self._get_embedder()
        store = VectorStore.load(self.memory_dir)
        if store.index.ntotal == 0:
            return []
        vec = embedder.embed([query])[0]
        hits = store.search(vec, top_k=top_k)
        return [meta["text"] for meta, _score in hits]
