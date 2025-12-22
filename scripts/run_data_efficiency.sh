#!/bin/bash

# Script to run Data Efficiency Experiments for RAG
# Evaluates k-NN performance across different training set sizes.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python run_data_efficiency.py \
    --model_name_or_path "Qwen/Qwen3-Embedding-0.6B" \
    --dataset_name "coastalcph/lex_glue" \
    --dataset_config_name "ecthr_a" \
    --output_dir "./output/data_efficiency_qwen" \
    --data_sizes_str "100,500,1000,2000,4500,9000" \
    --max_seq_length 2048 \
    --batch_size 32 \
    --k 10 \
    --threshold 0.4 \
    --bf16 true

