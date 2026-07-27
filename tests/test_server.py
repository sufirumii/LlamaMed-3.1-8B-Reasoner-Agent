"""Integration tests for server.py using FastAPI's TestClient.

Uses fake backend/embedder (no model weights or network needed) to verify
the actual HTTP contract the ui/ JS relies on: session CRUD, chat with a
tool call round-trip, PDF attach -> ingest -> retrieve, per-session
document isolation, and static file serving.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llamamed_agent.backends.base import LLMBackend
from llamamed_agent.config import Config


class FakeEmbedder:
    """Deterministic, network-free stand-in for sentence-transformers."""

    dim = 8
    _KEYWORDS = ["kidney", "creatinine", "disease", "stage", "chronic", "assessment", "patient", "report"]

    def __init__(self, *a, **kw):
        pass

    def embed(self, texts):
        vecs = []
        for t in texts:
            v = np.zeros(self.dim, dtype="float32")
            tl = t.lower()
            for i, kw in enumerate(self._KEYWORDS):
                if kw in tl:
                    v[i] = 1.0
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm > 0 else v)
        return np.array(vecs, dtype="float32")


class ScriptedBackend(LLMBackend):
    """Deterministic backend: calls search_documents once, then answers."""

    def generate(self, prompt, stop=None, max_tokens=512, temperature=0.6, top_p=0.95):
        if "Observation:" in prompt:
            return (
                "Thought: The document confirms it.\n"
                "Final Answer: The lab report indicates chronic kidney disease, stage 3."
            )
        if max_tokens <= 5:
            return "relevant"
        return (
            'Thought: I should check the attached document.\n'
            'Action: search_documents\n'
            'Action Input: {"query": "kidney disease assessment"}\n'
        )


@pytest.fixture
def client(monkeypatch):
    import llamamed_agent.rag.ingest as ingest_mod
    import llamamed_agent.tools.rag_tool as rag_tool_mod
    import llamamed_agent.server as server_mod

    monkeypatch.setattr(ingest_mod, "Embedder", FakeEmbedder)
    monkeypatch.setattr(rag_tool_mod, "Embedder", FakeEmbedder)
    monkeypatch.setattr(server_mod, "get_backend", lambda model_cfg: ScriptedBackend())

    from fastapi.testclient import TestClient

    tmpdir = tempfile.mkdtemp()
    cfg = Config()
    cfg.rag.index_dir = str(Path(tmpdir) / "index")
    cfg.rag.pdf_dir = str(Path(tmpdir) / "pdfs")
    cfg.rag.relevance_threshold = -1.0  # deterministic: always treat local match as sufficient
    cfg.rag.web_fallback_enabled = False

    app = server_mod.create_app(cfg)
    return TestClient(app)


def _make_test_pdf(path: Path) -> None:
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path))
    c.drawString(72, 720, "Patient lab report.")
    c.drawString(72, 700, "Serum creatinine: 1.4 mg/dL. Patient is a 62 year old woman.")
    c.drawString(72, 680, "Assessment: findings consistent with chronic kidney disease, stage 3.")
    c.save()


def test_session_crud(client):
    assert client.get("/api/sessions").json() == {"sessions": []}

    session_id = client.post("/api/sessions").json()["id"]
    assert session_id

    sessions = client.get("/api/sessions").json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["title"] == "New chat"

    client.delete(f"/api/sessions/{session_id}")
    assert client.get("/api/sessions").json() == {"sessions": []}


def test_chat_without_attachment(client):
    session_id = client.post("/api/sessions").json()["id"]
    r = client.post("/api/chat", json={"session_id": session_id, "message": "Hello"})
    assert r.status_code == 200
    data = r.json()
    assert "reply" in data and "trace" in data

    messages = client.get(f"/api/sessions/{session_id}/messages").json()["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_attach_and_retrieve_end_to_end(client, tmp_path):
    pdf_path = tmp_path / "lab_report.pdf"
    _make_test_pdf(pdf_path)

    session_id = client.post("/api/sessions").json()["id"]

    with open(pdf_path, "rb") as f:
        r = client.post(
            "/api/attach",
            data={"session_id": session_id},
            files={"file": ("lab_report.pdf", f, "application/pdf")},
        )
    assert r.json()["status"] == "ok"
    assert r.json()["chunks_indexed"] >= 1

    r = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "What does the lab report say about kidney disease?"},
    )
    data = r.json()
    assert "chronic kidney disease" in data["reply"].lower()
    assert data["trace"][0]["action"] == "search_documents"
    assert "[Attached PDF]" in data["trace"][0]["observation"]
    assert "lab_report.pdf" in data["trace"][0]["observation"]


def test_sessions_do_not_share_documents(client, tmp_path):
    pdf_path = tmp_path / "lab_report.pdf"
    _make_test_pdf(pdf_path)

    session_a = client.post("/api/sessions").json()["id"]
    with open(pdf_path, "rb") as f:
        client.post(
            "/api/attach",
            data={"session_id": session_a},
            files={"file": ("lab_report.pdf", f, "application/pdf")},
        )

    session_b = client.post("/api/sessions").json()["id"]
    r = client.post("/api/chat", json={"session_id": session_b, "message": "What does the lab report say?"})
    observation = r.json()["trace"][0]["observation"]
    assert "lab_report.pdf" not in observation


def test_static_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"LlamaMed" in r.content

    assert client.get("/style.css").status_code == 200
    assert client.get("/app.js").status_code == 200
