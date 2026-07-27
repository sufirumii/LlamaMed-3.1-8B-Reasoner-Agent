"""Extracts text from clinical PDFs (page by page, so chunks keep a page
citation) and ingests it into the FAISS store.

Extraction strategy: try pdfplumber first (better layout/table handling,
common in lab reports and discharge summaries); if a page yields nothing
useful, fall back to pypdf for that page. This mirrors the kind of mixed
document quality you actually get with scanned/exported clinical PDFs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from .chunking import chunk_text
from .embeddings import Embedder
from .store import VectorStore


def extract_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """Returns a list of (page_number, text) tuples, 1-indexed."""
    pages: List[Tuple[int, str]] = []

    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append((i, text))
    except Exception:
        pages = []

    if not pages or all(not t.strip() for _, t in pages):
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        pages = [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]

    return pages


def ingest_pdf(
    pdf_path: str,
    store: VectorStore,
    embedder: Embedder,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> int:
    """Extracts, chunks, embeds, and adds one PDF to the store. Returns the
    number of chunks added."""
    source_name = Path(pdf_path).name
    pages = extract_pages(pdf_path)

    records: List[Dict] = []
    for page_num, text in pages:
        if not text.strip():
            continue
        for chunk in chunk_text(text, chunk_size, chunk_overlap):
            records.append({"text": chunk, "source": source_name, "page": page_num})

    if not records:
        return 0

    vectors = embedder.embed([r["text"] for r in records])
    store.add(vectors, records)
    return len(records)


def ingest_page_range_text(pdf_path: str, start_page: int, end_page: int) -> str:
    """Returns verbatim extracted text for a page range (1-indexed,
    inclusive), for cases where exact wording matters more than semantic
    retrieval -- e.g. pulling an exact lab value table rather than a
    paraphrased chunk."""
    pages = extract_pages(pdf_path)
    selected = [text for page_num, text in pages if start_page <= page_num <= end_page]
    return "\n\n".join(selected).strip()


def ingest_path(
    path: str,
    index_dir: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> Dict[str, int]:
    """Ingests a single PDF or every PDF in a directory. Loads an existing
    index at index_dir if present, otherwise creates a new one. Saves the
    result back to index_dir. Returns {filename: chunks_added}."""
    embedder = Embedder(embedding_model)

    if VectorStore.exists(index_dir):
        store = VectorStore.load(index_dir)
    else:
        store = VectorStore(dim=embedder.dim)

    target = Path(path)
    pdf_files = [target] if target.is_file() else sorted(target.glob("*.pdf"))

    results: Dict[str, int] = {}
    for pdf_file in pdf_files:
        n = ingest_pdf(str(pdf_file), store, embedder, chunk_size, chunk_overlap)
        results[pdf_file.name] = n

    store.save(index_dir)
    return results
