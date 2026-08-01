from __future__ import annotations

from .base import Tool
from ..backends.base import LLMBackend
from ..rag.crag import retrieve_with_correction
from ..rag.embeddings import Embedder
from ..rag.store import VectorStore


class SearchDocumentsTool(Tool):
    name = "search_documents"
    description = (
        "Searches the attached documents for context relevant to a query (Corrective RAG). "
        "If the attached documents don't have a good enough answer, this automatically falls "
        "back to a free web search and returns that instead, clearly labeled by source."
    )
    parameters = {
        "query": {"type": "string", "description": "What to search for"},
        "top_k": {"type": "integer", "description": "How many local results to consider (default 4)"},
    }

    def __init__(
        self,
        backend: LLMBackend,
        index_dir: str,
        embedding_model: str,
        default_top_k: int = 4,
        relevance_threshold: float = 0.35,
        web_fallback_enabled: bool = True,
        max_web_results: int = 3,
    ):
        self.backend = backend
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self.default_top_k = default_top_k
        self.relevance_threshold = relevance_threshold
        self.web_fallback_enabled = web_fallback_enabled
        self.max_web_results = max_web_results
        self._embedder = None  # lazy: only load the embedding model once it's actually needed

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.embedding_model)
        return self._embedder

    def run(self, query: str, top_k: int = None) -> str:
        store = VectorStore.load(self.index_dir) if VectorStore.exists(self.index_dir) else None
        embedder = self._get_embedder()
        k = int(top_k) if top_k else self.default_top_k

        results = retrieve_with_correction(
            query=query,
            store=store,
            embedder=embedder,
            backend=self.backend,
            top_k=k,
            relevance_threshold=self.relevance_threshold,
            web_fallback_enabled=self.web_fallback_enabled,
            max_web_results=self.max_web_results,
        )

        if not results:
            return (
                "No relevant passages found in the attached documents, and web search "
                "returned nothing (or is unavailable). Try attaching a relevant PDF, or "
                "rephrasing the query."
            )

        tags = {"local": "[Attached PDF]", "pubmed": "[PubMed]", "web": "[Web]"}
        blocks = []
        for r in results:
            tag = tags.get(r.source, "[Web]")
            score_note = f", score={r.score:.2f}" if r.score is not None else ""
            blocks.append(f"{tag} {r.citation}{score_note}\n{r.text}")
        return "\n\n---\n\n".join(blocks)
