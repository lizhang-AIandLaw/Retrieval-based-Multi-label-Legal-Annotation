#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Limit to GPU 0
export CUDA_VISIBLE_DEVICES=0

SIZES=(100 500 1000)

echo "=== Running BERT Small Data (100, 500, 1000) ==="

for SIZE in "${SIZES[@]}"; do
    echo ">>> Training with ${SIZE} samples..."
    OUTPUT_DIR="./output/bert_scaling_${SIZE}"
    
    python train.py configs/bert_config.json \
        --output_dir "${OUTPUT_DIR}" \
        --max_train_samples ${SIZE} \
        --num_train_epochs 20 \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --load_best_model_at_end True \
        --metric_for_best_model "f1_micro" \
        --overwrite_output_dir True
done

