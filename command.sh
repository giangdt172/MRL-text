#!/usr/bin/env bash
set -e

# Edit the settings below, then run: bash command.sh
# Each method evaluates after every training epoch.
cd "$(dirname "${BASH_SOURCE[0]}")"
PROJECT_DIR="$PWD"

METHOD="flag_mrl"  # mrl, mipic, flag_mrl
FLAG_OBJECTIVE="trace_ratio"  # trace_ratio, spherical; registered in FlagMRL/objectives.py
MODEL_NAME="google-bert/bert-base-uncased"
CKPT_NAME="bert_run1"
CUDA_DEVICE=1  # GPU to expose to both training and evaluation.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
DEVICE="cuda:0"  # The selected GPU is index 0 inside this process.
PYTHON_BIN="python"
DATA_PATH="$PROJECT_DIR/final-data/final_data.csv"
OUTPUT_DIR="$PROJECT_DIR/checkpoints/$METHOD/$CKPT_NAME"

MRL_EVAL_WORK_DIR="$PROJECT_DIR/MRL"
MIPIC_EVAL_WORK_DIR="$PROJECT_DIR/MIPIC"
FLAG_EVAL_WORK_DIR="$PROJECT_DIR/FlagMRL"
EVAL_BATCH_SIZE=64

COMMON_ARGS=(
    --model_name "$MODEL_NAME"
    --data_path "$DATA_PATH"
    --output_dir "$OUTPUT_DIR"
    --device "$DEVICE" --precision auto
    --epochs 8 --batch_size 64 --lr 2e-5
    --max_length 256 --seed 42
    --weight_decay 0.01 --temperature 0.07
    --log_every 10
    # --matryoshka_dims 16 32 64 128 256 512
    # --gradient_checkpointing
)

MRL_ARGS=(
    --max_grad_norm 0.0 --min_lr 2e-6
    --num_workers 2
    # Default warmup: 10% of training steps.
    # --warmup_steps 100
    # --level_weights 1 1 1 1 1 1 1
)

MIPIC_ARGS=(
    --max_grad_norm 1.0 --warmup_steps 0
    --pooling cls --base_k 64 --d_att 64
    --num_workers 2
    # Layers and pipeline default to the selected model's configuration.
    # Example overrides for a 12-layer, 768-dimensional encoder:
    # --align_layers 4 8
    # --pipeline_pairs 3 16 5 32 7 64 8 128 9 256 10 512 11 768
)

FLAG_ARGS=(
    --flag_objective "$FLAG_OBJECTIVE"
    --max_grad_norm 1.0 --warmup_steps 0
    --flag_init identity --flag_init_batches 8
    --flag_lr 0.1 --flag_eps 1e-4
    --flag_update_every 1 --flag_retraction svd
    --flag_lambda_neg 1.0 --flag_lambda_cross 0.0
    --flag_far_mode uniform --flag_far_tau 0.1
    # --flag_use_neg_mask
)

mrl() {
    "$PYTHON_BIN" -u MRL/train.py "${COMMON_ARGS[@]}" "${MRL_ARGS[@]}" \
        --run_eval 1 --eval_work_dir "$MRL_EVAL_WORK_DIR" --eval_batch_size "$EVAL_BATCH_SIZE"
}

mipic() {
    "$PYTHON_BIN" -u MIPIC/train.py "${COMMON_ARGS[@]}" "${MIPIC_ARGS[@]}" \
        --run_eval 1 --eval_work_dir "$MIPIC_EVAL_WORK_DIR" --eval_batch_size "$EVAL_BATCH_SIZE"
}

flag_mrl() {
    "$PYTHON_BIN" -u FlagMRL/train.py "${COMMON_ARGS[@]}" "${FLAG_ARGS[@]}" \
        --run_eval 1 --eval_work_dir "$FLAG_EVAL_WORK_DIR" --eval_batch_size "$EVAL_BATCH_SIZE"
}

"$METHOD"
