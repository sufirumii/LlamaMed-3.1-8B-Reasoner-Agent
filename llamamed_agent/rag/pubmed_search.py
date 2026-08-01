"""Free PubMed search via NCBI's E-utilities -- no API key required.

This is a second, medicine-specific alternative to the generic DuckDuckGo
web fallback in web_search.py. For clinical/biomedical queries it's a
better-quality source than general web search (peer-reviewed abstracts vs.
arbitrary pages), so Corrective RAG tries it first and only falls through
to DuckDuckGo if PubMed comes back empty (e.g. for a non-biomedical
query).

No API key is required, but NCBI asks unregistered callers to stay under
3 requests/second and to identify the tool (see `tool`/`email` params
below) -- both handled here so this stays a good citizen of a free public
service. An optional NCBI API key (env: NCBI_API_KEY) raises that limit to
10 req/s if you have one; it's entirely optional.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Dict, List

import requests

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_TIMEOUT = 8


def _common_params() -> Dict:
    params = {"tool": "llamamed-agent", "email": "llamamed-agent@localhost"}
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


def pubmed_search(query: str, max_results: int = 3) -> List[Dict]:
    """Returns a list of {title, url, snippet} dicts, or [] on any failure
    (network hiccup, rate limit, no results) -- fails soft, same contract
    as web_search.web_search, so callers can treat both sources the same.
    """
    try:
        pmids = _esearch(query, max_results)
        if not pmids:
            return []
        return _efetch_summaries(pmids)
    except Exception:
        return []


def _esearch(query: str, max_results: int) -> List[str]:
    params = {
        **_common_params(),
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": "relevance",
        "retmode": "json",
    }
    resp = requests.get(_ESEARCH_URL, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def _efetch_summaries(pmids: List[str]) -> List[Dict]:
    params = {
        **_common_params(),
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    resp = requests.get(_EFETCH_URL, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return _parse_pubmed_xml(resp.text)


def _parse_pubmed_xml(xml_text: str) -> List[Dict]:
    results: List[Dict] = []
    root = ET.fromstring(xml_text)
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""
        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else "(untitled)"

        abstract_parts = [
            "".join(node.itertext()) for node in article.findall(".//AbstractText")
        ]
        snippet = " ".join(abstract_parts).strip()
        if not snippet:
            continue  # skip entries with no usable abstract text

        results.append(
            {
                "title": title,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "snippet": snippet[:1000],
            }
        )
    return results
