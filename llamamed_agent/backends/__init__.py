from __future__ import annotations

from ..config import ModelConfig
from .base import LLMBackend, build_llama3_prompt, truncate_at_stop


def get_backend(cfg: ModelConfig) -> LLMBackend:
    if cfg.backend == "gguf":
        from .gguf_backend import GGUFBackend

        return GGUFBackend(cfg)
    if cfg.backend == "transformers":
        from .transformers_backend import TransformersBackend

        return TransformersBackend(cfg)
    raise ValueError(f"Unknown backend '{cfg.backend}'. Use 'gguf' or 'transformers'.")


__all__ = ["get_backend", "LLMBackend", "build_llama3_prompt", "truncate_at_stop"]
