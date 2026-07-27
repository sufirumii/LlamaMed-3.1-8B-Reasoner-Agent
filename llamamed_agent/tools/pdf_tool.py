from __future__ import annotations

from pathlib import Path

from .base import Tool
from ..rag.embeddings import Embedder
from ..rag.ingest import ingest_page_range_text, ingest_pdf
from ..rag.store import VectorStore


class IngestPDFTool(Tool):
    name = "ingest_pdf"
    description = (
        "Reads a clinical PDF (lab report, discharge summary, guideline, paper, etc.) "
        "from a local file path, splits it into chunks, and adds it to the searchable "
        "document index so search_documents can retrieve from it. Use this the first "
        "time a specific PDF is mentioned, before trying to search it."
    )
    parameters = {"pdf_path": {"type": "string", "description": "Local path to the PDF file"}}

    def __init__(self, index_dir: str, embedding_model: str, chunk_size: int, chunk_overlap: int):
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedder = None  # lazy: only load the embedding model if this tool is used

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.embedding_model)
        return self._embedder

    def run(self, pdf_path: str) -> str:
        if not Path(pdf_path).exists():
            return f"Error: no file found at '{pdf_path}'."

        embedder = self._get_embedder()
        if VectorStore.exists(self.index_dir):
            store = VectorStore.load(self.index_dir)
        else:
            store = VectorStore(dim=embedder.dim)

        n_chunks = ingest_pdf(pdf_path, store, embedder, self.chunk_size, self.chunk_overlap)
        store.save(self.index_dir)

        if n_chunks == 0:
            return (
                f"Indexed 0 chunks from '{pdf_path}' -- no extractable text was found. "
                "It may be a scanned/image-only PDF that needs OCR first."
            )
        return f"Indexed {n_chunks} chunks from '{pdf_path}'. You can now use search_documents on it."


class ReadPDFPagesTool(Tool):
    name = "read_pdf_pages"
    description = (
        "Returns the exact, verbatim extracted text of a page range from a local PDF, "
        "without any paraphrasing or chunking. Use this when exact wording matters, e.g. "
        "quoting a specific lab value table, rather than search_documents (which returns "
        "semantically relevant chunks that may come from anywhere in the document)."
    )
    parameters = {
        "pdf_path": {"type": "string", "description": "Local path to the PDF file"},
        "start_page": {"type": "integer", "description": "First page to read, 1-indexed"},
        "end_page": {"type": "integer", "description": "Last page to read, inclusive"},
    }

    def run(self, pdf_path: str, start_page: int, end_page: int) -> str:
        if not Path(pdf_path).exists():
            return f"Error: no file found at '{pdf_path}'."
        text = ingest_page_range_text(pdf_path, int(start_page), int(end_page))
        return text if text else f"No extractable text found on pages {start_page}-{end_page}."
