#!/bin/bash
# Terminal: BERT LoRA Fixed 3 Epochs - Medium LR (2e-4)
# Hypothesis: Standard LoRA baseline.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
export HF_HOME="/ix/kashley/liz239/.cache/huggingface"
export HF_DATASETS_CACHE="/ix/kashley/liz239/.cache/huggingface/datasets"

DATASETS=("ecthr_a" "ecthr_b" "eurlex" "scotus")
SIZES=(100 500 1000 2000 4500 9000)

for DS in "${DATASETS[@]}"; do
    echo "### Processing Dataset: ${DS} (LoRA 2e-4, Fixed 3 Epochs) ###"
    
    if [ "$DS" == "ecthr_a" ] || [ "$DS" == "ecthr_b" ]; then NLABELS=10; fi
    if [ "$DS" == "eurlex" ]; then NLABELS=100; fi
    if [ "$DS" == "scotus" ]; then NLABELS=14; fi
    
    for SIZE in "${SIZES[@]}"; do
        echo ">>> Training BERT LoRA (LR=2e-4) on ${DS} with ${SIZE} samples..."
        OUTPUT_DIR="./output/bert_lora_fixed3ep_2e4_${DS}_${SIZE}"
        
        python train.py configs/bert_config.json \
            --dataset_config_name "${DS}" \
            --output_dir "${OUTPUT_DIR}" \
            --max_train_samples ${SIZE} \
            --num_train_epochs 3 \
            --num_labels ${NLABELS} \
            --learning_rate 2e-4 \
            --hierarchical True \
            --max_segments 64 \
            --max_segment_length 128 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 8 \
            --gradient_checkpointing True \
            --use_lora True \
            --lora_r 8 \
            --lora_alpha 16 \
            --lora_target_modules "query" "value" \
            --save_total_limit 1 \
            --overwrite_output_dir True
    done
done
