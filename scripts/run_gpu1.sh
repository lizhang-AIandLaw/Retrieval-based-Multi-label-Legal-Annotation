#!/bin/bash
# GPU 1: BERT Small (Training)
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
chmod +x scripts/exp_bert_multidata_small.sh

echo ">>> Starting BERT Small Pipeline on GPU 1..."
./scripts/exp_bert_multidata_small.sh
echo ">>> GPU 1 Finished!"

