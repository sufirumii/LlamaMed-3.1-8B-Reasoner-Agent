import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from llamamed_agent.memory import LongTermMemory, SessionMemory


class FakeEmbedder:
    dim = 4

    def embed(self, texts):
        return np.array([[1.0, 0.0, 0.0, 0.0]] * len(texts), dtype="float32")


class FakeVectorStore:
    """Minimal in-memory stand-in for the FAISS-backed VectorStore, so
    LongTermMemory can be tested without faiss or sentence-transformers
    installed."""

    _saved = {}  # class-level "disk" keyed by memory_dir

    def __init__(self, dim):
        self.dim = dim
        self.metadata = []
        self.index = type("Idx", (), {"ntotal": 0})()

    def add(self, vectors, metadata):
        self.metadata.extend(metadata)
        self.index.ntotal = len(self.metadata)

    def search(self, query_vec, top_k=4):
        return [(m, 1.0) for m in self.metadata[:top_k]]

    def save(self, memory_dir):
        FakeVectorStore._saved[memory_dir] = self

    @classmethod
    def load(cls, memory_dir):
        return cls._saved[memory_dir]

    @classmethod
    def exists(cls, memory_dir):
        return memory_dir in cls._saved


def test_session_memory_persists_and_reloads():
    with tempfile.TemporaryDirectory() as tmp:
        session = SessionMemory(tmp)
        session.add("user", "What's the CKD-EPI eGFR formula?")
        session.add("assistant", "It's a race-free equation published in 2021.")
        session_id = session.session_id

        reloaded = SessionMemory(tmp, session_id=session_id)
        assert len(reloaded.turns) == 2
        assert reloaded.turns[0].role == "user"
        assert "eGFR" in reloaded.turns[0].content


def test_session_memory_lists_saved_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        s1 = SessionMemory(tmp)
        s1.add("user", "hello")
        s2 = SessionMemory(tmp)
        s2.add("user", "hi again")

        sessions = SessionMemory.list_sessions(tmp)
        assert s1.session_id in sessions
        assert s2.session_id in sessions


def test_session_memory_empty_history_message():
    with tempfile.TemporaryDirectory() as tmp:
        session = SessionMemory(tmp)
        assert "no turns yet" in session.render_history()


def test_long_term_memory_remember_and_recall():
    FakeVectorStore._saved.clear()
    with patch("llamamed_agent.memory.VectorStore", FakeVectorStore), \
         patch("llamamed_agent.memory.Embedder", return_value=FakeEmbedder()):
        ltm = LongTermMemory(memory_dir="fake-dir", embedding_model="fake-model")
        ltm.remember("What's the MAP formula?", "MAP = DBP + 1/3(SBP - DBP)", session_id="s1")

        hits = ltm.recall("MAP formula", top_k=3)
        assert len(hits) == 1
        assert "MAP" in hits[0]


def test_long_term_memory_recall_empty_when_nothing_stored():
    FakeVectorStore._saved.clear()
    with patch("llamamed_agent.memory.VectorStore", FakeVectorStore), \
         patch("llamamed_agent.memory.Embedder", return_value=FakeEmbedder()):
        ltm = LongTermMemory(memory_dir="empty-dir", embedding_model="fake-model")
        assert ltm.recall("anything") == []
