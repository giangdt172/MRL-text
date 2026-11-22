#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# ── Configurable defaults ──────────────────────────────────────────────────
# Empty DATA_PATH loads Quora pair/train from Hugging Face.
DATA_PATH="${DATA_PATH:-}"
DATA_ARGS=(--max_train_samples "${MAX_TRAIN_SAMPLES:-150000}")

# Default: the 21 tasks in common/mteb_tasks.py. EVAL_TASKS can override them.
# Example: EVAL_TASKS="STSBenchmark SICK-R".
read -r -a MTEB_TASKS <<< "${EVAL_TASKS:-}"
EVAL_ARGS=(--eval_batch_size "${EVAL_BATCH_SIZE:-64}" --eval_max_length "${EVAL_MAX_LENGTH:-256}"
           --eval_splits test --eval_languages eng)
if (( ${#MTEB_TASKS[@]} )); then EVAL_ARGS+=(--eval_tasks "${MTEB_TASKS[@]}"); fi
if [[ -n "$DATA_PATH" ]]; then DATA_ARGS+=(--data_path "$DATA_PATH"); fi
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-2e-5}"
SAVE_ROOT="${SAVE_ROOT:-$PROJECT_DIR/checkpoints/ESE/quora-duplicates-pair}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/ESE}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"

# ── Parse flags ───────────────────────────────────────────────────────────
PARALLEL=false
EVAL_ONLY=false
TRAIN_ONLY=false

for arg in "$@"; do
    case $arg in
        --parallel)   PARALLEL=true ;;
        --eval-only)  EVAL_ONLY=true ;;
        --train-only) TRAIN_ONLY=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if $EVAL_ONLY && $TRAIN_ONLY; then
    echo "--eval-only and --train-only are mutually exclusive." >&2
    exit 2
fi

mkdir -p "$SAVE_ROOT" "$LOG_ROOT"

# ── Model registry ─────────────────────────────────────────────────────────
# Format: "model_key"
MODELS=("tinybert" "bert" "qwen" "bgem3")

# ── Helper: train one model ────────────────────────────────────────────────
train_model() {
    local KEY="$1"
    local CKPT_DIR="${SAVE_ROOT}/${KEY}"
    local LOG_FILE="${LOG_ROOT}/train_${KEY}.log"

    echo ""
    echo "================================================================"
    echo " TRAINING: ${KEY}"
    echo " Checkpoint dir : ${CKPT_DIR}"
    echo " Log file       : ${LOG_FILE}"
    echo "================================================================"

    mkdir -p "$CKPT_DIR"

    local RUN_EVAL="${RUN_EVAL:-1}"
    if $TRAIN_ONLY; then RUN_EVAL=0; fi
    "$PYTHON_BIN" -u ESE/train.py \
        --model_key   "${KEY}" \
        "${DATA_ARGS[@]}" \
        --epochs      "${EPOCHS}" \
        --batch_size  "${BATCH_SIZE}" \
        --lr          "${LR}" \
        --save_dir    "${CKPT_DIR}" \
        --device "$DEVICE" --precision "${PRECISION:-auto}" \
        --ese_compression_size "${ESE_COMPRESSION_SIZE:-128}" \
        --ese_kl_temperature "${ESE_KL_TEMPERATURE:-1.0}" \
        --pooling "${POOLING:-mean}" \
        --run_eval "$RUN_EVAL" "${EVAL_ARGS[@]}" \
        2>&1 | tee "${LOG_FILE}"

    echo "✅ Training finished for ${KEY}"
}

# ── Helper: evaluate one model ─────────────────────────────────────────────
eval_model() {
    local KEY="$1"
    local CKPT_DIR="${SAVE_ROOT}/${KEY}"
    local LOG_FILE="${LOG_ROOT}/eval_${KEY}.log"

    echo ""
    echo "================================================================"
    echo " EVALUATING: ${KEY}"
    echo " Checkpoint dir : ${CKPT_DIR}"
    echo " Log file       : ${LOG_FILE}"
    echo "================================================================"

    if [[ ! -f "$CKPT_DIR/config.json" ]]; then
        echo "No trained checkpoint found at $CKPT_DIR." >&2
        return 1
    fi

    "$PYTHON_BIN" -u ESE/eval.py \
        --checkpoint_dir "$CKPT_DIR" --device "$DEVICE" \
        "${EVAL_ARGS[@]}" \
        2>&1 | tee "${LOG_FILE}"

    echo "✅ Evaluation finished for ${KEY}"
}

# ── Run ────────────────────────────────────────────────────────────────────
PIDS=()

for KEY in "${MODELS[@]}"; do
    if $PARALLEL; then
        # Launch each model in the background
        (
            if ! $EVAL_ONLY; then
                train_model "$KEY"
            fi
            if $EVAL_ONLY; then
                eval_model "$KEY"
            fi
        ) &
        PIDS+=($!)
    else
        # Sequential
        if ! $EVAL_ONLY; then
            train_model "$KEY"
        fi
        if $EVAL_ONLY; then
            eval_model "$KEY"
        fi
    fi
done

# Wait for all background jobs (parallel mode only)
if $PARALLEL; then
    echo ""
    echo "Waiting for all ${#PIDS[@]} parallel jobs to finish..."
    FAILED=0
    for PID in "${PIDS[@]}"; do
        wait "$PID" || { echo "Job $PID failed"; FAILED=$((FAILED+1)); }
    done
    if [ "$FAILED" -gt 0 ]; then
        echo "❌ $FAILED job(s) failed."
        exit 1
    fi
fi

echo ""
echo "================================================================"
echo " ALL DONE"
echo " Checkpoints : ${SAVE_ROOT}/"
echo " Logs        : ${LOG_ROOT}/"
echo "================================================================"
