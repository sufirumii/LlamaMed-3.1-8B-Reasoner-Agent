---
license: apache-2.0
base_model: unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
datasets:
- lingshu-medical-mllm/ReasonMed
library_name: transformers
pipeline_tag: text-generation
tags:
- medical
- reasoning
- llama-3.1
- reasonmed
- chain-of-thought
language:
- en
---

# LlamaMed-3.1-8B-Reasoner

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/66e00ba55e4fd4bfead4a97c/SpwOsRpS_Bd9pmC4k_I98.png" alt="LlamaMed-3.1-8B-Reasoner" width="100%">
</p>

LlamaMed-3.1-8B-Reasoner is a fine-tune of **Llama-3.1-8B-Instruct** trained
on **ReasonMed**, a dataset of chain-of-thought medical reasoning over
multiple-choice clinical questions. The model works through a question
step by step — considering each answer option in turn — before giving a
final answer, in the same structured reasoning style as its training data.

## Model Details

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

## Training

Trained on a single Tesla T4 GPU using Unsloth for memory-efficient QLoRA
fine-tuning, with periodic adapter checkpoints saved during training.

## Intended Use

This model is a research checkpoint intended for exploring medical
reasoning fine-tunes. It is not validated for clinical use and should
not be used to inform real medical decisions.
