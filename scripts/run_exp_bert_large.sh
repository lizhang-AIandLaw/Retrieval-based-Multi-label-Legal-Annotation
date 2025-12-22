#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Limit to GPU 1
export CUDA_VISIBLE_DEVICES=1

SIZES=(2000 4500 9000)

echo "=== Running BERT Large Data (2000, 4500, 9000) ==="

for SIZE in "${SIZES[@]}"; do
    echo ">>> Training with ${SIZE} samples..."
    OUTPUT_DIR="./output/bert_scaling_${SIZE}"
    
    python train.py configs/bert_config.json \
        --output_dir "${OUTPUT_DIR}" \
        --max_train_samples ${SIZE} \
        --num_train_epochs 5 \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --load_best_model_at_end True \
        --metric_for_best_model "f1_micro" \
        --overwrite_output_dir True
done

