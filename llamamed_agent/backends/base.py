"""Backend interface.

Both backends (GGUF/llama.cpp and Transformers) implement the same
`generate(prompt, stop, max_tokens, temperature, top_p) -> str` signature,
operating on a single raw prompt string rather than a chat-template
message list.

This is a deliberate design choice: the agent's ReAct loop needs to
generate a *partial* assistant turn (a scratchpad that already contains
prior Thought/Action/Observation blocks) and continue it, then stop at an
exact string like "Observation:". Chat-template APIs are built around
whole turns, not partial continuations, so both backends instead build
the raw Llama-3 special-token prompt directly. This keeps behavior
identical across backends and avoids surprises from template drift.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


def build_llama3_prompt(system: str, user: str, scratchpad: str = "") -> str:
    """Builds a raw Llama-3 Instruct prompt, ending mid-assistant-turn.

    The returned string ends right after the assistant header, with the
    running scratchpad appended, so the model's continuation naturally
    picks up from there.
    """
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{scratchpad}"
    )
    return prompt


def truncate_at_stop(text: str, stop: List[str]) -> str:
    """Cuts `text` at the earliest occurrence of any stop string."""
    cut_at = len(text)
    for s in stop:
        idx = text.find(s)
        if idx != -1:
            cut_at = min(cut_at, idx)
    return text[:cut_at]


class LLMBackend(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> str:
        """Returns the generated continuation only (not the prompt)."""
        raise NotImplementedError

    def close(self) -> None:
        pass
