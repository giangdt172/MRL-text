#!/usr/bin/env bash
set -euo pipefail

# Example: FLAG_OBJECTIVE=trace_ratio bash FlagMRL/run_all.sh
# Available objectives are registered in FlagMRL/objectives.py.
FLAG_OBJECTIVE="${FLAG_OBJECTIVE:-trace_ratio}"

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# "bert": {"model_name": "google-bert/bert-base-uncased"}
# "tinybert": {"model_name": "huawei-noah/TinyBERT_General_6L_768D"}
# "bgem3": {"model_name": "BAAI/bge-m3"}
# "qwen": {"model_name": "Qwen/Qwen3-Embedding-0.6B"}

MODEL_KEY="${MODEL_KEY:-bert}"
# Empty DATA_PATH uses project data/quora-duplicates-pair.csv from download.sh.
DATA_PATH="${DATA_PATH:-}"
DATA_ARGS=(--max_train_samples "${MAX_TRAIN_SAMPLES:-150000}")

# Default: the 21 tasks in common/mteb_tasks.py. EVAL_TASKS can override them.
# Example: EVAL_TASKS="STSBenchmark SICK-R".
read -r -a MTEB_TASKS <<< "${EVAL_TASKS:-}"
EVAL_ARGS=(--eval_batch_size "${EVAL_BATCH_SIZE:-64}" --eval_max_length "${EVAL_MAX_LENGTH:-256}"
           --eval_splits test --eval_languages eng)
if (( ${#MTEB_TASKS[@]} )); then EVAL_ARGS+=(--eval_tasks "${MTEB_TASKS[@]}"); fi
if [[ -n "$DATA_PATH" ]]; then DATA_ARGS+=(--data_path "$DATA_PATH"); fi
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/checkpoints/FlagMRL/$FLAG_OBJECTIVE/quora-duplicates-pair/$MODEL_KEY}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_EVAL="${RUN_EVAL:-1}"

MODEL_ARGS=(--model_key "$MODEL_KEY")
if [[ -n "${MODEL_NAME:-}" ]]; then
    MODEL_ARGS=(--model_name "$MODEL_NAME")
fi
[[ "$OUTPUT_DIR" == /* ]] || OUTPUT_DIR="$PROJECT_DIR/$OUTPUT_DIR"

"$PYTHON_BIN" -u FlagMRL/train.py \
    "${MODEL_ARGS[@]}" \
    --flag_objective "$FLAG_OBJECTIVE" \
    "${DATA_ARGS[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" --precision "${PRECISION:-auto}" \
    --epochs "${EPOCHS:-5}" --batch_size "${BATCH_SIZE:-32}" --lr "${LR:-2e-5}" \
    --num_workers "${NUM_WORKERS:-2}" \
    --run_eval "$RUN_EVAL" "${EVAL_ARGS[@]}" \
    --max_length "${MAX_LENGTH:-256}" \
    --flag_init "${FLAG_INIT:-identity}" --flag_lr "${FLAG_LR:-0.1}" \
    --flag_update_every "${FLAG_UPDATE_EVERY:-1}" \
    --flag_retraction "${FLAG_RETRACTION:-svd}" \
    --flag_lambda_cross "${FLAG_LAMBDA_CROSS:-0.0}"
