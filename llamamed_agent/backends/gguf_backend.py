"""Backend that runs a local GGUF checkpoint via llama-cpp-python.

Works on CPU-only machines and on GPUs (CUDA/Metal) that llama.cpp
supports, with no bitsandbytes/CUDA-toolkit requirement. This is the
default backend so that cloning the repo and running the agent does not
require a specific GPU setup.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from .base import LLMBackend, truncate_at_stop
from ..config import ModelConfig


class GGUFBackend(LLMBackend):
    def __init__(self, cfg: ModelConfig):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for the gguf backend. "
                "Install with: pip install llama-cpp-python"
            ) from e

        model_path = self._resolve_model_path(cfg)

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=cfg.n_ctx,
            n_gpu_layers=cfg.n_gpu_layers,
            n_threads=cfg.n_threads,
            verbose=False,
        )

    @staticmethod
    def _resolve_model_path(cfg: ModelConfig) -> Path:
        path = Path(cfg.gguf_path)
        if path.exists():
            return path

        if cfg.gguf_hf_repo and cfg.gguf_hf_file:
            from huggingface_hub import hf_hub_download

            path.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=cfg.gguf_hf_repo,
                filename=cfg.gguf_hf_file,
                local_dir=str(path.parent),
            )
            return Path(downloaded)

        raise FileNotFoundError(
            f"No GGUF file found at '{path}' and no gguf_hf_repo/gguf_hf_file "
            "configured to download one. Either place a .gguf file at that "
            "path (see scripts/convert_to_gguf.sh) or set model.gguf_hf_repo "
            "and model.gguf_hf_file in config.yaml."
        )

    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> str:
        stop = stop or []
        result = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            echo=False,
        )
        text = result["choices"][0]["text"]
        return truncate_at_stop(text, stop)

    def close(self) -> None:
        # llama-cpp-python frees resources on GC; nothing explicit needed.
        self._llm = None
