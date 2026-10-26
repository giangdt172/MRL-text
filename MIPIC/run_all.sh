#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
    echo "Usage: $0. Configure options with environment variables (MODEL_KEY, DATA_PATH, OUTPUT_DIR, ...)." >&2
    exit 2
fi

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Models: bert=google-bert/bert-base-uncased; tinybert=huawei-noah/TinyBERT_General_6L_768D
#         bgem3=BAAI/bge-m3; qwen=Qwen/Qwen3-Embedding-0.6B.

MODEL_KEY="${MODEL_KEY:-bgem3}"
DATA_PATH="${DATA_PATH:-./final-data/final_data.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/checkpoints/MIPIC/$MODEL_KEY}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_WORK_DIR="${EVAL_WORK_DIR:-$PROJECT_DIR}"
case "$RUN_EVAL" in
    0|1) ;;
    *) echo "RUN_EVAL must be 0 or 1." >&2; exit 2 ;;
esac
[[ "$OUTPUT_DIR" == /* ]] || OUTPUT_DIR="$PROJECT_DIR/$OUTPUT_DIR"
[[ "$EVAL_WORK_DIR" == /* ]] || EVAL_WORK_DIR="$PROJECT_DIR/$EVAL_WORK_DIR"

MODEL_ARGS=(--model_key "$MODEL_KEY")
if [[ -n "${MODEL_NAME:-}" ]]; then
    MODEL_ARGS=(--model_name "$MODEL_NAME")
fi
EXTRA_ARGS=()
if [[ "${GRADIENT_CHECKPOINTING:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--gradient_checkpointing)
fi

"$PYTHON_BIN" -u MIPIC/train.py \
    "${MODEL_ARGS[@]}" \
    --data_path "$DATA_PATH" --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" --precision "${PRECISION:-auto}" \
    --epochs "${EPOCHS:-5}" --batch_size "${BATCH_SIZE:-32}" --lr "${LR:-2e-5}" \
    --max_length "${MAX_LENGTH:-256}" --seed "${SEED:-42}" \
    --weight_decay "${WEIGHT_DECAY:-0.01}" --temperature "${TEMPERATURE:-0.07}" \
    --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
    --num_workers "${NUM_WORKERS:-2}" --log_every "${LOG_EVERY:-10}" \
    --run_eval "$RUN_EVAL" --eval_work_dir "$EVAL_WORK_DIR" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-64}" \
    --pooling "${POOLING:-cls}" --base_k "${BASE_K:-64}" --d_att "${D_ATT:-64}" \
    "${EXTRA_ARGS[@]}"
