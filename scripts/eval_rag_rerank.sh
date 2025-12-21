#!/bin/bash

# Script to evaluate RAG with Reranking using Qwen-Embedding + Qwen3-Reranker

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Note: rerank_batch_size is kept small (4) because reranking involves long context (Query + Doc)
# and Qwen3-Reranker is a generation model.

python eval_rag_rerank.py \
    --model_name_or_path "Qwen/Qwen3-Embedding-0.6B" \
    --reranker_model_name_or_path "Qwen/Qwen3-Reranker-0.6B" \
    --dataset_name "coastalcph/lex_glue" \
    --dataset_config_name "ecthr_a" \
    --cache_dir "./output/rag_results/cache" \
    --max_seq_length 2048 \
    --rerank_max_length 2048 \
    --batch_size 32 \
    --rerank_batch_size 4 \
    --retrieve_top_n 50 \
    --k 10 \
    --threshold 0.4 \
    --bf16 true
