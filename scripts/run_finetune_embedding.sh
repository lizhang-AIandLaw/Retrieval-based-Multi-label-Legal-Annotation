#!/bin/bash

# Script to Finetune Qwen Embedding Model using LoRA and Contrastive Loss
# This aligns the embedding space for the RAG task.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training
# Note: Batch size is critical for Contrastive Learning (larger is better for more negatives).
# With Qwen-0.6B and 2048 seq len, batch size 4-8 fits on 24GB VRAM.
# Gradient accumulation doesn't help with "Batch Size" for contrastive loss negatives, 
# only for optimization stability. So we try to maximize physical batch size.

# REPLACE 'YourUsername/qwen-legal-embedding-lora' with your actual Hugging Face repo ID
# Ensure you are logged in with `huggingface-cli login`

python train_embedding.py \
    --model_name_or_path "Qwen/Qwen3-Embedding-0.6B" \
    --dataset_name "coastalcph/lex_glue" \
    --dataset_config_name "ecthr_a" \
    --output_dir "./output/qwen_embedding_finetuned" \
    --max_seq_length 2048 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --bf16 true \
    --push_to_hub true \
    --hub_model_id "Liz239/qwen-legal-embedding-lora"
