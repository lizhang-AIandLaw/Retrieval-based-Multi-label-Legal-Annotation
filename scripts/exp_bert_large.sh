#!/bin/bash
# Terminal 3: Run BERT Fine-tuning on Large Data (2000, 4500, 9000)
# Suggest running on GPU 1: CUDA_VISIBLE_DEVICES=1 ./scripts/exp_bert_large.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SIZES=(2000 4500 9000)

for SIZE in "${SIZES[@]}"; do
    echo ">>> [Large Group] Training BERT with ${SIZE} samples..."
    OUTPUT_DIR="./output/bert_scaling_${SIZE}"
    
    # Larger data converges faster per epoch
    EPOCHS=5
    
    python train.py configs/bert_config.json \
        --output_dir "${OUTPUT_DIR}" \
        --max_train_samples ${SIZE} \
        --num_train_epochs ${EPOCHS} \
        --save_total_limit 1 \
        --overwrite_output_dir True
        
    echo ">>> Finished Size ${SIZE}"
done

