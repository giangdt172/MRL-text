#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# ── Configurable defaults ──────────────────────────────────────────────────
DATA_PATH="${DATA_PATH:-$PROJECT_DIR/final-data/final_data.csv}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-2e-5}"
SAVE_ROOT="${SAVE_ROOT:-$PROJECT_DIR/checkpoints/ESE}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/ESE}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
EVAL_WORK_DIR="${EVAL_WORK_DIR:-$PROJECT_DIR/ESE}"

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

    local RUN_EVAL=1
    if $TRAIN_ONLY; then RUN_EVAL=0; fi
    "$PYTHON_BIN" -u ESE/train.py \
        --model_key   "${KEY}" \
        --data_path   "${DATA_PATH}" \
        --epochs      "${EPOCHS}" \
        --batch_size  "${BATCH_SIZE}" \
        --lr          "${LR}" \
        --save_dir    "${CKPT_DIR}" \
        --device "$DEVICE" --precision "${PRECISION:-auto}" \
        --ese_compression_size "${ESE_COMPRESSION_SIZE:-128}" \
        --ese_kl_temperature "${ESE_KL_TEMPERATURE:-1.0}" \
        --pooling "${POOLING:-mean}" \
        --run_eval "$RUN_EVAL" --eval_work_dir "$EVAL_WORK_DIR" \
        --eval_batch_size "${EVAL_BATCH_SIZE:-64}" \
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
        --eval_work_dir "$EVAL_WORK_DIR" --batch_size "${EVAL_BATCH_SIZE:-64}" \
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
