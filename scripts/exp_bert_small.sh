#!/bin/bash
# Terminal 2: Run BERT Fine-tuning on Small Data (100, 500, 1000)
# Suggest running on GPU 0: CUDA_VISIBLE_DEVICES=0 ./scripts/exp_bert_small.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SIZES=(100 500 1000)

for SIZE in "${SIZES[@]}"; do
    echo ">>> [Small Group] Training BERT with ${SIZE} samples..."
    OUTPUT_DIR="./output/bert_scaling_${SIZE}"
    
    # Small data needs more epochs to converge
    if [ ${SIZE} -le 500 ]; then
        EPOCHS=20
    else
        EPOCHS=10
    fi
    
    python train.py configs/bert_config.json \
        --output_dir "${OUTPUT_DIR}" \
        --max_train_samples ${SIZE} \
        --num_train_epochs ${EPOCHS} \
        --save_total_limit 1 \
        --overwrite_output_dir True
        
    echo ">>> Finished Size ${SIZE}"
done

