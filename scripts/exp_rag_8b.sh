#!/bin/bash
# Terminal 3: RAG 8B (8192 Length) - Updated for Tuning & Linear Probe & NCC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Prioritize ecthr_a
DATASETS=("ecthr_a" "ecthr_b" "eurlex")
SIZES="100,500,1000,2000,4500,9000"

for DS in "${DATASETS[@]}"; do
    echo "=== Running RAG 8B on ${DS} ==="
    python run_data_scaling_experiment.py \
        --model_name_or_path "Qwen/Qwen3-Embedding-8B" \
        --dataset_name "coastalcph/lex_glue" \
        --dataset_config_name "${DS}" \
        --data_sizes "${SIZES}" \
        --output_dir "./output/data_scaling_results_8b_8k_ncc" \
        --max_seq_length 8192 \
        --batch_size 1 \
        --tune_params True \
        --use_linear_probe True \
        --use_ncc True \
        --bf16 true
done
