#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

METHODS=(mrl mipic flag_mrl ese)  # Run sequentially in this order.
# Models: bert=google-bert/bert-base-uncased; tinybert=huawei-noah/TinyBERT_General_6L_768D
#         bgem3=BAAI/bge-m3; qwen=Qwen/Qwen3-Embedding-0.6B.
MODEL_NAME="google-bert/bert-base-uncased"
CUDA_DEVICE=1

# Run bash download.sh first. Training reads data/quora-duplicates-pair.csv.
# Evaluation uses the 21 tasks in common/mteb_tasks.py and downloads their data automatically.
declare -A TRAIN_DIRS=(
    [mrl]=MRL [mipic]=MIPIC [flag_mrl]=FlagMRL [ese]=ESE
)
declare -A METHOD_ARGS=(
    [mrl]="--max_grad_norm 0.0 --min_lr 2e-6"
    [mipic]="--pooling cls --base_k 64 --d_att 64"
    [flag_mrl]="--flag_objective trace_ratio --flag_lr 0.05"
    [ese]="--pooling mean --ese_compression_size 128 --ese_kl_temperature 1.0"
)

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

for METHOD in "${METHODS[@]}"; do
    OUTPUT_DIR="checkpoints/$METHOD/${MODEL_NAME##*/}"
    read -r -a EXTRA_ARGS <<< "${METHOD_ARGS[$METHOD]}"
    echo "Running $METHOD -> $OUTPUT_DIR"

    python -u "${TRAIN_DIRS[$METHOD]}/train.py" \
        --model_name "$MODEL_NAME" --output_dir "$OUTPUT_DIR" \
        --device cuda:0 --precision auto \
        --epochs 2 --batch_size 64 --lr 2e-5 \
        --max_train_samples 150000 --max_length 256 --seed 42 \
        --weight_decay 0.01 --temperature 0.07 \
        --num_workers 2 --log_every 10 \
        --run_eval 1 --eval_batch_size 64 --eval_max_length 256 \
        "${EXTRA_ARGS[@]}"
done
