#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

"${PYTHON_BIN:-python}" - <<'PY'
from datasets import load_dataset

load_dataset("sentence-transformers/quora-duplicates", "pair", split="train")
PY
