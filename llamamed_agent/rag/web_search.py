"""Free web search fallback for Corrective RAG.

Uses DuckDuckGo's search results via the `ddgs` package -- no API key, no
account, no email required, which is the whole point: this only exists as
a safety net for when the attached PDFs don't have the answer, and it has
to stay zero-cost and zero-signup to fit that role.

Caveat, stated plainly: DuckDuckGo does not offer an official free search
API, so this is an unofficial/best-effort method (the kind widely used in
open-source local-agent projects). If it ever breaks because DuckDuckGo
changes something, the rest of the agent keeps working -- it just loses
the web-fallback safety net until the dependency is updated.
"""

from __future__ import annotations

from typing import Dict, List


def web_search(query: str, max_results: int = 3) -> List[Dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # older package name
        except ImportError:
            return []

    results: List[Dict] = []
    try:
        with DDGS() as ddgs:
            for hit in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": hit.get("title", ""),
                        "url": hit.get("href") or hit.get("link", ""),
                        "snippet": hit.get("body", ""),
                    }
                )
    except Exception:
        # Network hiccup, layout change, rate limit, etc. -- fail soft.
        return []
    return results
