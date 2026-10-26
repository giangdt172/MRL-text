#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
    echo "Usage: $0. Configure options with environment variables (MODEL_KEY, DATA_PATH, OUTPUT_DIR, ...)." >&2
    exit 2
fi

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# "bert": {"model_name": "google-bert/bert-base-uncased"}
# "tinybert": {"model_name": "huawei-noah/TinyBERT_General_6L_768D"}
# "bgem3": {"model_name": "BAAI/bge-m3"}
# "qwen": {"model_name": "Qwen/Qwen3-Embedding-0.6B"}

MODEL_KEY="${MODEL_KEY:-bgem3}"
DATA_PATH="${DATA_PATH:-./final-data/final_data.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/checkpoints/MRL_base/$MODEL_KEY}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_WORK_DIR="${EVAL_WORK_DIR:-$PROJECT_DIR/MRL}"

MODEL_ARGS=(--model_key "$MODEL_KEY")
if [[ -n "${MODEL_NAME:-}" ]]; then
    MODEL_ARGS=(--model_name "$MODEL_NAME")
fi
[[ "$OUTPUT_DIR" == /* ]] || OUTPUT_DIR="$PROJECT_DIR/$OUTPUT_DIR"

"$PYTHON_BIN" -u MRL/train.py \
    "${MODEL_ARGS[@]}" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" --precision "${PRECISION:-auto}" \
    --epochs "${EPOCHS:-5}" --batch_size "${BATCH_SIZE:-32}" --lr "${LR:-2e-5}" \
    --max_length "${MAX_LENGTH:-256}"

if [[ "$RUN_EVAL" == "1" ]]; then
    (
        cd "$EVAL_WORK_DIR"
        "$PYTHON_BIN" -u "$PROJECT_DIR/MRL/eval.py" \
            --checkpoint_dir "$OUTPUT_DIR" --device "$DEVICE"
    ) 2>&1 | tee "$OUTPUT_DIR/eval.log"
fi
