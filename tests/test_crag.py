import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from llamamed_agent.backends.base import LLMBackend
from llamamed_agent.rag.crag import retrieve_with_correction


class FakeEmbedder:
    dim = 4

    def embed(self, texts):
        return np.array([[1.0, 0.0, 0.0, 0.0]] * len(texts), dtype="float32")


class FakeStore:
    """A minimal stand-in for VectorStore with a scripted search result."""

    class _Index:
        def __init__(self, ntotal):
            self.ntotal = ntotal

    def __init__(self, scripted_results):
        self._scripted = scripted_results
        self.index = self._Index(ntotal=len(scripted_results))

    def search(self, query_vec, top_k=4):
        return self._scripted[:top_k]


class FakeBackend(LLMBackend):
    """Returns a canned relevance verdict regardless of prompt content."""

    def __init__(self, grade_verdict="relevant"):
        self.grade_verdict = grade_verdict
        self.calls = 0

    def generate(self, prompt, stop=None, max_tokens=512, temperature=0.6, top_p=0.95):
        self.calls += 1
        return self.grade_verdict


def test_strong_local_match_skips_web_search():
    store = FakeStore([({"text": "good match", "source": "a.pdf", "page": 1}, 0.9)])
    backend = FakeBackend()
    with patch("llamamed_agent.rag.crag.web_search") as mock_web:
        results = retrieve_with_correction(
            "query", store, FakeEmbedder(), backend,
            relevance_threshold=0.35, web_fallback_enabled=True,
        )
    mock_web.assert_not_called()
    assert len(results) == 1
    assert results[0].source == "local"


def test_weak_local_match_triggers_web_fallback():
    store = FakeStore([({"text": "weak match", "source": "a.pdf", "page": 1}, 0.05)])
    backend = FakeBackend()
    with patch("llamamed_agent.rag.crag.web_search") as mock_web:
        mock_web.return_value = [
            {"title": "Result", "url": "http://example.com", "snippet": "web snippet"}
        ]
        results = retrieve_with_correction(
            "query", store, FakeEmbedder(), backend,
            relevance_threshold=0.35, web_fallback_enabled=True,
        )
    mock_web.assert_called_once()
    sources = {r.source for r in results}
    assert "web" in sources
    assert "local" in sources  # weak local result is kept, not discarded


def test_no_local_store_goes_straight_to_web():
    backend = FakeBackend()
    with patch("llamamed_agent.rag.crag.web_search") as mock_web:
        mock_web.return_value = [
            {"title": "Result", "url": "http://example.com", "snippet": "web snippet"}
        ]
        results = retrieve_with_correction(
            "query", None, FakeEmbedder(), backend,
            relevance_threshold=0.35, web_fallback_enabled=True,
        )
    mock_web.assert_called_once()
    assert len(results) == 1
    assert results[0].source == "web"


def test_web_fallback_disabled_returns_local_only_even_if_weak():
    store = FakeStore([({"text": "weak match", "source": "a.pdf", "page": 1}, 0.05)])
    backend = FakeBackend()
    with patch("llamamed_agent.rag.crag.web_search") as mock_web:
        results = retrieve_with_correction(
            "query", store, FakeEmbedder(), backend,
            relevance_threshold=0.35, web_fallback_enabled=False,
        )
    mock_web.assert_not_called()
    assert len(results) == 1
    assert results[0].source == "local"


def test_borderline_score_defers_to_llm_grading():
    # best_score is within the borderline band above the threshold
    store = FakeStore([({"text": "borderline match", "source": "a.pdf", "page": 1}, 0.40)])
    backend = FakeBackend(grade_verdict="irrelevant")
    with patch("llamamed_agent.rag.crag.web_search") as mock_web:
        mock_web.return_value = []
        retrieve_with_correction(
            "query", store, FakeEmbedder(), backend,
            relevance_threshold=0.35, web_fallback_enabled=True,
        )
    mock_web.assert_called_once()  # LLM said "irrelevant" -> should trigger web fallback
