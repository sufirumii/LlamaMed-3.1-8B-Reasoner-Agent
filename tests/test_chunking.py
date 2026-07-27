import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llamamed_agent.rag.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_one_chunk():
    chunks = chunk_text("A single short paragraph.", chunk_size=800, chunk_overlap=100)
    assert len(chunks) == 1


def test_long_text_splits_into_multiple_chunks():
    paragraph = "Clinical finding. " * 100  # long enough to force a split
    text = "\n\n".join([paragraph] * 3)
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 + 50 for c in chunks)  # allow for overlap slack
