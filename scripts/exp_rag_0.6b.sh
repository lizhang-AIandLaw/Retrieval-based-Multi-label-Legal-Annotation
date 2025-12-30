#!/bin/bash
# Terminal 1: RAG 0.6B (8192 Length) - Updated for Tuning & Linear Probe & SVM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Prioritize ecthr_a as requested
DATASETS=("ecthr_a" "ecthr_b" "eurlex")
SIZES="100,500,1000,2000,4500,9000"

for DS in "${DATASETS[@]}"; do
    echo "=== Running RAG 0.6B on ${DS} ==="
    python run_data_scaling_experiment.py \
        --model_name_or_path "Qwen/Qwen3-Embedding-0.6B" \
        --dataset_name "coastalcph/lex_glue" \
        --dataset_config_name "${DS}" \
        --data_sizes "${SIZES}" \
        --output_dir "./output/data_scaling_results_0.6b_8k_svm" \
        --max_seq_length 8192 \
        --batch_size 16 \
        --tune_params True \
        --use_linear_probe True \
        --use_svm True \
        --bf16 true
done
