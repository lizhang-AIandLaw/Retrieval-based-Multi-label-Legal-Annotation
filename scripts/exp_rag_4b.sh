#!/bin/bash
# Terminal 2: RAG 4B (8192 Length)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

DATASETS=("ecthr_a" "ecthr_b" "eurlex")
SIZES="100,500,1000,2000,4500,9000"

for DS in "${DATASETS[@]}"; do
    echo "=== Running RAG 4B on ${DS} ==="
    python run_data_scaling_experiment.py \
        --model_name_or_path "Qwen/Qwen3-Embedding-4B" \
        --dataset_name "coastalcph/lex_glue" \
        --dataset_config_name "${DS}" \
        --data_sizes "${SIZES}" \
        --output_dir "./output/data_scaling_results_4b_8k" \
        --max_seq_length 8192 \
        --batch_size 2 \
        --k 10 \
        --threshold 0.4 \
        --bf16 true
done
