from __future__ import annotations

from typing import Optional

from ..backends.base import LLMBackend
from ..config import Config
from ..memory import LongTermMemory
from .base import Tool, ToolRegistry
from .calculator_tool import ClinicalCalculatorTool
from .memory_tool import RecallMemoryTool
from .pdf_tool import IngestPDFTool, ReadPDFPagesTool
from .pubmed_tool import SearchPubMedTool
from .rag_tool import SearchDocumentsTool


def build_default_registry(
    cfg: Config, backend: LLMBackend, memory: Optional[LongTermMemory] = None
) -> ToolRegistry:
    tools = [
        SearchDocumentsTool(
            backend=backend,
            index_dir=cfg.rag.index_dir,
            embedding_model=cfg.rag.embedding_model,
            default_top_k=cfg.rag.top_k,
            relevance_threshold=cfg.rag.relevance_threshold,
            web_fallback_enabled=cfg.rag.web_fallback_enabled,
            max_web_results=cfg.rag.max_web_results,
        ),
        IngestPDFTool(
            index_dir=cfg.rag.index_dir,
            embedding_model=cfg.rag.embedding_model,
            chunk_size=cfg.rag.chunk_size,
            chunk_overlap=cfg.rag.chunk_overlap,
        ),
        ReadPDFPagesTool(),
        ClinicalCalculatorTool(),
        SearchPubMedTool(),
    ]
    if memory is not None:
        tools.append(RecallMemoryTool(memory))
    return ToolRegistry(tools)


__all__ = ["Tool", "ToolRegistry", "build_default_registry"]
