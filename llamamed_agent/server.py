"""Local web server for llamamed_agent/ui/.

Thin adapter only: every endpoint here just translates HTTP <-> the same
Agent, tools, and Corrective RAG retrieval used by the CLI. No reasoning
or retrieval logic is duplicated in this file.

Sessions (chats) are simple JSON files under data/sessions/. Each session
also gets its own document index under data/index/<session_id>/, so PDFs
attached in one chat don't leak into another chat's retrieval -- this
matches what the UI shows (per-chat attachment chips).
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent.core import Agent
from .backends import get_backend
from .config import Config
from .rag.ingest import ingest_path
from .tools import build_default_registry

UI_DIR = Path(__file__).resolve().parent / "ui"


class ChatRequest(BaseModel):
    session_id: str
    message: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sessions_dir(cfg: Config) -> Path:
    path = Path(cfg.rag.index_dir).parent / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session_path(cfg: Config, session_id: str) -> Path:
    return _sessions_dir(cfg) / f"{session_id}.json"


def _session_index_dir(cfg: Config, session_id: str) -> str:
    return str(Path(cfg.rag.index_dir) / session_id)


def _new_session_dict(session_id: str) -> dict:
    now = _now()
    return {"id": session_id, "title": "New chat", "created_at": now, "updated_at": now, "messages": []}


def _load_session(cfg: Config, session_id: str) -> Optional[dict]:
    path = _session_path(cfg, session_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_session(cfg: Config, session: dict) -> None:
    with open(_session_path(cfg, session["id"]), "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2)


def create_app(cfg: Optional[Config] = None) -> FastAPI:
    cfg = cfg or Config.load()
    backend = get_backend(cfg.model)
    tools = build_default_registry(cfg, backend)
    agent = Agent(backend, tools, cfg)
    search_tool = tools.get("search_documents")

    app = FastAPI(title="LlamaMed-3.1-8B-Reasoner-Agent")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/api/sessions")
    def list_sessions():
        sessions = []
        for path in _sessions_dir(cfg).glob("*.json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append(
                {"id": data["id"], "title": data.get("title", "New chat"), "updated_at": data.get("updated_at", "")}
            )
        sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        return {"sessions": sessions}

    @app.post("/api/sessions")
    def create_session():
        session_id = uuid.uuid4().hex[:12]
        _save_session(cfg, _new_session_dict(session_id))
        return {"id": session_id}

    @app.get("/api/sessions/{session_id}/messages")
    def get_messages(session_id: str):
        session = _load_session(cfg, session_id)
        return {"messages": session.get("messages", []) if session else []}

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str):
        path = _session_path(cfg, session_id)
        if path.exists():
            path.unlink()
        shutil.rmtree(_session_index_dir(cfg, session_id), ignore_errors=True)
        shutil.rmtree(Path(cfg.rag.pdf_dir) / "sessions" / session_id, ignore_errors=True)
        return {"deleted": True}

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        session = _load_session(cfg, req.session_id) or _new_session_dict(req.session_id)

        # Scope retrieval to this session's own attached documents.
        search_tool.index_dir = _session_index_dir(cfg, req.session_id)

        trace = []

        def on_step(step, observation):
            trace.append(
                {
                    "thought": step.thought,
                    "action": step.action,
                    "action_input": step.action_input,
                    "observation": observation,
                }
            )

        result = agent.run(req.message, on_step=on_step)

        session["messages"].append({"role": "user", "text": req.message, "trace": None})
        session["messages"].append({"role": "assistant", "text": result.final_answer, "trace": trace})
        session["updated_at"] = _now()
        if session["title"] == "New chat":
            session["title"] = req.message[:40] + ("..." if len(req.message) > 40 else "")
        _save_session(cfg, session)

        return {"reply": result.final_answer, "trace": trace}

    @app.post("/api/attach")
    def attach(session_id: str = Form(...), file: UploadFile = File(...)):
        if _load_session(cfg, session_id) is None:
            _save_session(cfg, _new_session_dict(session_id))

        upload_dir = Path(cfg.rag.pdf_dir) / "sessions" / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / file.filename
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)

        try:
            results = ingest_path(
                str(dest),
                index_dir=_session_index_dir(cfg, session_id),
                embedding_model=cfg.rag.embedding_model,
                chunk_size=cfg.rag.chunk_size,
                chunk_overlap=cfg.rag.chunk_overlap,
            )
            n_chunks = results.get(file.filename, 0)
            if n_chunks == 0:
                return {
                    "filename": file.filename,
                    "chunks_indexed": 0,
                    "status": "error",
                    "error": "No extractable text found (scanned/image-only PDF?)",
                }
            return {"filename": file.filename, "chunks_indexed": n_chunks, "status": "ok", "error": None}
        except Exception as e:  # noqa: BLE001 - surfaced to the UI, not raised
            return {"filename": file.filename, "chunks_indexed": 0, "status": "error", "error": str(e)}

    # Serve the static UI at "/" -- must be mounted last so it doesn't shadow /api/* routes.
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

    return app
