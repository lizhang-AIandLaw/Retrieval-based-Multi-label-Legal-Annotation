#!/bin/bash

# Analyze clustering quality of Base Qwen Embedding Model on Training Set

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install visualization deps if needed
# pip install matplotlib seaborn scikit-learn umap-learn

python analyze_clusters.py \
    --model_name_or_path "Qwen/Qwen3-Embedding-0.6B" \
    --dataset_name "coastalcph/lex_glue" \
    --dataset_config_name "ecthr_a" \
    --max_seq_length 4096 \
    --batch_size 4 \
    --output_dir "./output/analysis_base_qwen" \
    --bf16

