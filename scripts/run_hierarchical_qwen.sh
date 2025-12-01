#!/bin/bash

# Run Hierarchical Qwen training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py configs/hierarchical_qwen_config.json \
    --push_to_hub true \
    --report_to wandb \
    --project "legal-classification" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true
