#!/bin/bash
# Terminal 1: RAG 0.6B (GPU 0)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

DATASETS=("ecthr_a" "ecthr_b" "eurlex")
# Use fixed numbers for fair comparison across datasets
SIZES="100,500,1000,2000,4500,9000"

for DS in "${DATASETS[@]}"; do
    echo "=== Running RAG 0.6B on ${DS} ==="
    python run_data_scaling_experiment.py \
        --model_name_or_path "Qwen/Qwen3-Embedding-0.6B" \
        --dataset_name "coastalcph/lex_glue" \
        --dataset_config_name "${DS}" \
        --data_sizes "${SIZES}" \
        --output_dir "./output/data_scaling_results_0.6b" \
        --max_seq_length 2048 \
        --batch_size 32 \
        --k 10 \
        --threshold 0.4 \
        --bf16 true
done

