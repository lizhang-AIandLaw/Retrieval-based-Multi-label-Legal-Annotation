#!/bin/bash
# GPU 2: RAG 8B (Inference)
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
chmod +x scripts/exp_rag_8b.sh

echo ">>> Starting RAG 8B Pipeline on GPU 2..."
./scripts/exp_rag_8b.sh
echo ">>> GPU 2 Finished!"

