#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p data
export HF_HOME="$PWD/data/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"

"${PYTHON_BIN:-python}" - <<'PY'
from datasets import load_dataset

dataset = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train")
dataset.to_csv("data/quora-duplicates-pair.csv", index=False)
PY
