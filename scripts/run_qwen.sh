#!/bin/bash

# Run Qwen training
# Explicitly set batch size to 1 to avoid OOM with 8192 context length
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py configs/qwen_config.json \
    --push_to_hub true \
    --report_to wandb \
    --project "legal-classification" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true
