#!/bin/bash
# Terminal 4: Run BERT Fine-tuning (Large Data) on MULTIPLE datasets
# Suggest GPU 1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASETS=("ecthr_a" "ecthr_b" "eurlex")
SIZES=(2000 4500 9000 20000)

for DS in "${DATASETS[@]}"; do
    if [ "$DS" == "ecthr_a" ] || [ "$DS" == "ecthr_b" ]; then NLABELS=10; fi
    if [ "$DS" == "scotus" ]; then NLABELS=14; fi
    if [ "$DS" == "eurlex" ] || [ "$DS" == "ledgar" ]; then NLABELS=100; fi
    
    for SIZE in "${SIZES[@]}"; do
        echo ">>> Training BERT on ${DS} with ${SIZE} samples..."
        OUTPUT_DIR="./output/bert_scaling_${DS}_${SIZE}"
        
        EPOCHS=5
        
        python train.py configs/bert_config.json \
            --dataset_config_name "${DS}" \
            --output_dir "${OUTPUT_DIR}" \
            --max_train_samples ${SIZE} \
            --num_train_epochs ${EPOCHS} \
            --num_labels ${NLABELS} \
            --save_total_limit 1 \
            --overwrite_output_dir True
    done
done

