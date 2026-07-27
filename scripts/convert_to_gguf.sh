#!/usr/bin/env bash
# Converts the Rumiii/LlamaMed-3.1-8B-Reasoner HF checkpoint to GGUF and
# produces a quantized version for the gguf backend.
#
# Run this once. After it finishes you'll have, e.g.:
#   models/LlamaMed-3.1-8B-Reasoner.f16.gguf   (unquantized, large)
#   models/LlamaMed-3.1-8B-Reasoner.Q4_K_M.gguf (quantized, ~4-5GB, default)
#
# Usage:
#   bash scripts/convert_to_gguf.sh [HF_REPO] [QUANT_TYPE]
#   bash scripts/convert_to_gguf.sh Rumiii/LlamaMed-3.1-8B-Reasoner Q4_K_M

set -euo pipefail

HF_REPO="${1:-Rumiii/LlamaMed-3.1-8B-Reasoner}"
QUANT_TYPE="${2:-Q4_K_M}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="$ROOT_DIR/third_party/llama.cpp"
MODELS_DIR="$ROOT_DIR/models"
HF_LOCAL_DIR="$MODELS_DIR/hf/$(basename "$HF_REPO")"

mkdir -p "$MODELS_DIR" "$(dirname "$LLAMA_CPP_DIR")"

echo ">> Downloading $HF_REPO from Hugging Face..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${HF_REPO}', local_dir='${HF_LOCAL_DIR}')
"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
  echo ">> Cloning llama.cpp..."
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi

echo ">> Installing llama.cpp Python conversion requirements..."
pip install -q -r "$LLAMA_CPP_DIR/requirements/requirements-convert_hf_to_gguf.txt"

F16_OUT="$MODELS_DIR/$(basename "$HF_REPO").f16.gguf"
echo ">> Converting HF checkpoint -> GGUF (f16): $F16_OUT"
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$HF_LOCAL_DIR" --outfile "$F16_OUT" --outtype f16

echo ">> Building llama-quantize..."
cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -DGGML_CUDA=OFF > /dev/null
cmake --build "$LLAMA_CPP_DIR/build" --target llama-quantize -j > /dev/null

QUANT_OUT="$MODELS_DIR/$(basename "$HF_REPO").${QUANT_TYPE}.gguf"
echo ">> Quantizing -> $QUANT_OUT ($QUANT_TYPE)"
"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$F16_OUT" "$QUANT_OUT" "$QUANT_TYPE"

echo ""
echo "Done. Point model.gguf_path in config.yaml at:"
echo "  $QUANT_OUT"
