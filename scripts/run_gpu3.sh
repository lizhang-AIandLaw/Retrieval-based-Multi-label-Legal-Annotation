#!/bin/bash
# GPU 3: RAG 4B -> RAG 0.6B (Inference)
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
chmod +x scripts/exp_rag_4b.sh
chmod +x scripts/exp_rag_0.6b.sh

echo ">>> Starting RAG 4B Pipeline on GPU 3..."
./scripts/exp_rag_4b.sh

echo ">>> Starting RAG 0.6B Pipeline on GPU 3..."
./scripts/exp_rag_0.6b.sh

echo ">>> GPU 3 Finished!"

