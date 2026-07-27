"""Corrective RAG (Yan et al., 2024): grade the retrieved local context,
and fall back to a free web search when it isn't good enough, rather than
answering from weak or irrelevant context. This is the one retrieval
algorithm this agent uses -- deliberately not stacked with anything else.

This is a pragmatic, fully-local version of the idea. The original paper
trains a small dedicated relevance evaluator; here, a cheap similarity-
score threshold does most of the filtering (no extra model call for the
common cases), and the main LLM is only asked to grade relevance in the
borderline zone near that threshold -- so a normal query with either a
clearly strong or clearly weak local match never pays for an extra
generation call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..backends.base import LLMBackend, build_llama3_prompt
from .embeddings import Embedder
from .store import VectorStore
from .web_search import web_search

RELEVANCE_CHECK_SYSTEM = (
    "You judge whether a passage is relevant enough to help answer a question. "
    "Respond with exactly one word: relevant, ambiguous, or irrelevant."
)

# Borderline band around the threshold in which we bother asking the LLM
# to double check, instead of trusting the raw cosine score alone.
_BORDERLINE_BAND = 0.15


@dataclass
class RetrievalResult:
    text: str
    source: str  # "local" or "web"
    citation: str  # e.g. "report.pdf, p.3" or a URL
    score: Optional[float] = None


def _grade_with_llm(backend: LLMBackend, query: str, passage: str) -> str:
    prompt = build_llama3_prompt(
        RELEVANCE_CHECK_SYSTEM,
        f"Question: {query}\n\nPassage:\n{passage}",
    )
    verdict = backend.generate(
        prompt, stop=["<|eot_id|>"], max_tokens=5, temperature=0.0
    ).strip().lower()
    if "irrelevant" in verdict:
        return "irrelevant"
    if "ambiguous" in verdict:
        return "ambiguous"
    return "relevant"


def retrieve_with_correction(
    query: str,
    store: Optional[VectorStore],
    embedder: Embedder,
    backend: LLMBackend,
    top_k: int = 4,
    relevance_threshold: float = 0.35,
    web_fallback_enabled: bool = True,
    max_web_results: int = 3,
) -> List[RetrievalResult]:
    """Runs local retrieval, grades it, and supplements with (or falls back
    to) free web search when the local context is weak."""

    local_results: List[RetrievalResult] = []

    if store is not None and store.index.ntotal > 0:
        query_vec = embedder.embed([query])[0]
        for meta, score in store.search(query_vec, top_k=top_k):
            local_results.append(
                RetrievalResult(
                    text=meta["text"],
                    source="local",
                    citation=f"{meta['source']}, p.{meta['page']}",
                    score=score,
                )
            )

    best_score = max((r.score for r in local_results), default=0.0)
    needs_web = best_score < relevance_threshold

    if local_results and 0 <= (best_score - relevance_threshold) < _BORDERLINE_BAND:
        verdict = _grade_with_llm(backend, query, local_results[0].text)
        needs_web = verdict == "irrelevant"

    if not web_fallback_enabled or not needs_web:
        return local_results

    web_hits = web_search(query, max_results=max_web_results)
    web_results = [
        RetrievalResult(
            text=f"{hit['title']}: {hit['snippet']}",
            source="web",
            citation=hit["url"],
        )
        for hit in web_hits
        if hit.get("snippet")
    ]

    # Keep any genuinely useful local passages and supplement with web
    # results, rather than discarding local evidence just because it
    # wasn't a perfect match.
    return local_results + web_results
