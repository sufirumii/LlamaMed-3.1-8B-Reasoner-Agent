from __future__ import annotations

from .base import Tool
from ..rag.pubmed_search import pubmed_search


class SearchPubMedTool(Tool):
    name = "search_pubmed"
    description = (
        "Searches PubMed directly (free, no API key) for peer-reviewed abstracts on a "
        "biomedical topic. Use this instead of search_documents when the question is about "
        "published literature/evidence in general, rather than about an attached document."
    )
    parameters = {
        "query": {"type": "string", "description": "Biomedical search terms"},
        "max_results": {"type": "integer", "description": "How many abstracts to return (default 3)"},
    }

    def run(self, query: str, max_results: int = 3) -> str:
        hits = pubmed_search(query, max_results=int(max_results) if max_results else 3)
        if not hits:
            return "No PubMed results found (or PubMed is unreachable right now)."
        blocks = [f"[PubMed] {h['url']}\n{h['title']}\n{h['snippet']}" for h in hits]
        return "\n\n---\n\n".join(blocks)
