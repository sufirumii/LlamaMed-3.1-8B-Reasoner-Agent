"""Central configuration for the agent.

Configuration is resolved in this order (later wins):
  1. Built-in defaults below
  2. config.yaml in the project root, if present
  3. Environment variables (LLAMAMED_*)
  4. CLI flags (applied by cli.py)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelConfig:
    # "gguf" (llama.cpp, default) or "transformers" (HF + optional bitsandbytes)
    backend: str = "gguf"

    # --- GGUF backend ---
    # Local path to a .gguf file. If it doesn't exist and gguf_hf_repo /
    # gguf_hf_file are set, it will be downloaded there on first use.
    gguf_path: str = "models/LlamaMed-3.1-8B-Reasoner.Q4_K_M.gguf"
    gguf_hf_repo: Optional[str] = None  # e.g. "Rumiii/LlamaMed-3.1-8B-Reasoner-GGUF"
    gguf_hf_file: Optional[str] = None  # e.g. "LlamaMed-3.1-8B-Reasoner.Q4_K_M.gguf"
    n_ctx: int = 4096
    n_gpu_layers: int = -1  # -1 = offload all layers if a GPU is available
    n_threads: Optional[int] = None  # None = let llama.cpp decide

    # --- transformers backend ---
    hf_repo: str = "Rumiii/LlamaMed-3.1-8B-Reasoner"
    load_in_4bit: bool = False  # set True on small/consumer GPUs

    # --- generation ---
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 1024


@dataclass
class RagConfig:
    index_dir: str = "data/index"
    pdf_dir: str = "data/pdfs"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    # --- Corrective RAG (the only retrieval algorithm used) ---
    web_fallback_enabled: bool = True
    relevance_threshold: float = 0.35  # cosine similarity below this = "not good enough"
    max_web_results: int = 3


@dataclass
class AgentConfig:
    max_iterations: int = 6
    verbose: bool = True  # print Thought/Action/Observation trace to the console


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        cfg = cls()
        yaml_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            _merge_section(cfg.model, raw.get("model", {}))
            _merge_section(cfg.rag, raw.get("rag", {}))
            _merge_section(cfg.agent, raw.get("agent", {}))
        _apply_env_overrides(cfg)
        return cfg


def _merge_section(section_obj, values: dict) -> None:
    valid = {f.name for f in fields(section_obj)}
    for key, value in values.items():
        if key in valid:
            setattr(section_obj, key, value)


def _apply_env_overrides(cfg: Config) -> None:
    env_map = {
        "LLAMAMED_BACKEND": ("model", "backend", str),
        "LLAMAMED_GGUF_PATH": ("model", "gguf_path", str),
        "LLAMAMED_HF_REPO": ("model", "hf_repo", str),
        "LLAMAMED_TEMPERATURE": ("model", "temperature", float),
        "LLAMAMED_INDEX_DIR": ("rag", "index_dir", str),
        "LLAMAMED_TOP_K": ("rag", "top_k", int),
    }
    for env_key, (section, attr, cast) in env_map.items():
        if env_key in os.environ:
            setattr(getattr(cfg, section), attr, cast(os.environ[env_key]))
