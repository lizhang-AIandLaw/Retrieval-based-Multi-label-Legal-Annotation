#!/bin/bash
# GPU 0: BERT Large (Training)
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
chmod +x scripts/exp_bert_multidata_large.sh

echo ">>> Starting BERT Large Pipeline on GPU 0..."
./scripts/exp_bert_multidata_large.sh
echo ">>> GPU 0 Finished!"

