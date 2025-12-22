#!/bin/bash
# Terminal 4: Hierarchical BERT (Small Data) + LoRA - GPU 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

DATASETS=("ecthr_a" "ecthr_b" "eurlex")
SIZES=(100 500 1000)

for DS in "${DATASETS[@]}"; do
    echo "### Processing Dataset: ${DS} ###"
    
    if [ "$DS" == "ecthr_a" ] || [ "$DS" == "ecthr_b" ]; then NLABELS=10; fi
    if [ "$DS" == "eurlex" ]; then NLABELS=100; fi
    
    for SIZE in "${SIZES[@]}"; do
        echo ">>> Training Hierarchical BERT (LoRA) on ${DS} with ${SIZE} samples..."
        OUTPUT_DIR="./output/bert_hier_lora_scaling_${DS}_${SIZE}"
        
        if [ ${SIZE} -le 500 ]; then EPOCHS=20; else EPOCHS=10; fi
        
        python train.py configs/bert_config.json \
            --dataset_config_name "${DS}" \
            --output_dir "${OUTPUT_DIR}" \
            --max_train_samples ${SIZE} \
            --num_train_epochs ${EPOCHS} \
            --num_labels ${NLABELS} \
            --hierarchical True \
            --max_segments 64 \
            --max_segment_length 128 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 8 \
            --gradient_checkpointing True \
            --use_lora True \
            --lora_r 8 \
            --lora_alpha 16 \
            --save_total_limit 1 \
            --overwrite_output_dir True
    done
done
