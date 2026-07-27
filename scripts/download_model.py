"""Pre-downloads model weights so the first `llamamed-agent` run doesn't
stall on a large download.

Examples:
    # Download a pre-built GGUF quant from a HF repo you've uploaded
    python scripts/download_model.py gguf --repo Rumiii/LlamaMed-3.1-8B-Reasoner-GGUF \\
        --file LlamaMed-3.1-8B-Reasoner.Q4_K_M.gguf --out models/

    # Download the full HF checkpoint for the transformers backend
    python scripts/download_model.py hf --repo Rumiii/LlamaMed-3.1-8B-Reasoner --out models/hf/
"""

from __future__ import annotations

import argparse

from huggingface_hub import hf_hub_download, snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="kind", required=True)

    gguf = sub.add_parser("gguf", help="Download a single GGUF file")
    gguf.add_argument("--repo", required=True)
    gguf.add_argument("--file", required=True)
    gguf.add_argument("--out", default="models")

    hf = sub.add_parser("hf", help="Download a full HF checkpoint")
    hf.add_argument("--repo", required=True)
    hf.add_argument("--out", default="models/hf")

    args = parser.parse_args()

    if args.kind == "gguf":
        path = hf_hub_download(repo_id=args.repo, filename=args.file, local_dir=args.out)
    else:
        path = snapshot_download(repo_id=args.repo, local_dir=args.out)

    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()
