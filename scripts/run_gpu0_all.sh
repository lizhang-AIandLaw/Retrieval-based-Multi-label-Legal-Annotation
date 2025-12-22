#!/bin/bash
# GPU 0 Pipeline: RAG 0.6B -> RAG 4B -> BERT Small
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure scripts are executable
chmod +x scripts/exp_rag_0.6b.sh
chmod +x scripts/exp_rag_4b.sh
chmod +x scripts/exp_bert_multidata_small.sh

# 1. RAG 0.6B (Max Len 8192)
echo "--------------------------------------"
echo "Starting RAG 0.6B Pipeline..."
echo "--------------------------------------"
./scripts/exp_rag_0.6b.sh

# 2. RAG 4B (Max Len 8192)
echo "--------------------------------------"
echo "Starting RAG 4B Pipeline..."
echo "--------------------------------------"
./scripts/exp_rag_4b.sh

# 3. BERT Small (Hierarchical + LoRA)
echo "--------------------------------------"
echo "Starting BERT Small Pipeline..."
echo "--------------------------------------"
./scripts/exp_bert_multidata_small.sh

echo "All GPU 0 tasks completed!"

