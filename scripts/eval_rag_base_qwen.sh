#!/bin/bash

# Run RAG evaluation with Base Qwen Embedding Model (Zero-shot/Training-free)
# This uses the training set as the knowledge base.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Using Qwen-0.6B
# Batch size 4 to be safe with 8192 context
python eval_rag.py \
    --model_name_or_path "Qwen/Qwen3-Embedding-0.6B" \
    --dataset_name "coastalcph/lex_glue" \
    --dataset_config_name "ecthr_a" \
    --max_seq_length 8192 \
    --batch_size 4 \
    --k_neighbors 10 \
    --output_dir "./output/rag_base_qwen" \
    --bf16 true

