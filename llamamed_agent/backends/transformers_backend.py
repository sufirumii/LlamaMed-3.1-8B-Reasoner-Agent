"""Backend that runs the model directly via transformers.

Matches how the model was trained/uploaded (Rumiii/LlamaMed-3.1-8B-Reasoner).
Requires a CUDA GPU for reasonable speed; set model.load_in_4bit = true in
config.yaml on smaller/consumer GPUs.
"""

from __future__ import annotations

from typing import List, Optional

from .base import LLMBackend, truncate_at_stop
from ..config import ModelConfig


class TransformersBackend(LLMBackend):
    def __init__(self, cfg: ModelConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.hf_repo)

        model_kwargs = {"device_map": "auto"}
        if cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(cfg.hf_repo, **model_kwargs)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> str:
        stop = stop or []
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )
        with self._torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return truncate_at_stop(text, stop)
