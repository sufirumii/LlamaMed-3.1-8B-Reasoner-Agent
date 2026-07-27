"""Simple paragraph-aware chunker.

Splits on blank lines first (paragraphs), then greedily packs paragraphs
into chunks up to `chunk_size` characters, with `chunk_overlap` characters
of trailing context carried into the next chunk. This keeps clinical
tables and short paragraphs (e.g. a single lab value line) intact more
often than a naive fixed-width slice would.
"""

from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            # Paragraph itself is too long (e.g. a dense table dump);
            # fall back to a fixed-width slice with overlap.
            for i in range(0, len(para), chunk_size - chunk_overlap):
                piece = para[i : i + chunk_size]
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(piece)
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            overlap_tail = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{overlap_tail}\n\n{para}".strip()

    if current:
        chunks.append(current)

    return chunks
