#!/bin/bash
# GPU 1 Pipeline: RAG 8B -> BERT Large
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure scripts are executable
chmod +x scripts/exp_rag_8b.sh
chmod +x scripts/exp_bert_multidata_large.sh

# 1. RAG 8B (Max Len 8192)
echo "--------------------------------------"
echo "Starting RAG 8B Pipeline..."
echo "--------------------------------------"
./scripts/exp_rag_8b.sh

# 2. BERT Large (Hierarchical + LoRA)
echo "--------------------------------------"
echo "Starting BERT Large Pipeline..."
echo "--------------------------------------"
./scripts/exp_bert_multidata_large.sh

echo "All GPU 1 tasks completed!"

