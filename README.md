# LlamaMed-3.1-8B-Reasoner

<img width="1200" height="400" alt="LlamaMed-3.1-8B-Reasoner" src="https://github.com/user-attachments/assets/91085f09-a50c-43be-819c-932b6b8e9799" />

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/Rumiii/LlamaMed-3.1-8B-Reasoner)

A Llama-3.1-8B fine-tune that works through medical multiple-choice questions step by step, reasoning through each option before answering.

## Overview

LlamaMed-3.1-8B-Reasoner is fine-tuned from **Llama-3.1-8B-Instruct** on **ReasonMed**, a dataset of chain-of-thought medical reasoning over multiple-choice clinical questions. The model restates the question, evaluates each answer option, and reaches a final conclusion, following the reasoning structure of its training data.

## Details

- **Base model:** [unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit)
- **Dataset:** [lingshu-medical-mllm/ReasonMed](https://huggingface.co/datasets/lingshu-medical-mllm/ReasonMed) — 10,000 samples used for training
- **Method:** QLoRA (4-bit), rank 16, via [Unsloth](https://github.com/unslothai/unsloth)
- **License:** Apache 2.0

## Usage

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Rumiii/LlamaMed-3.1-8B-Reasoner",
    max_seq_length = 1024,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

messages = [
    {"role": "user", "content": "A 45-year-old man presents with polyuria, polydipsia, and weight loss. Fasting blood glucose is 210 mg/dL. What is the most likely diagnosis?"}
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(inputs, max_new_tokens=1024, temperature=0.6, top_p=0.95)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Intended Use

This is a research checkpoint for exploring medical reasoning fine-tunes. It is not validated for clinical use and should not inform real medical decisions.

## License

Apache 2.0
