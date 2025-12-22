#!/bin/bash
# Terminal 1: Run RAG Scaling with Qwen-4B on MULTIPLE datasets

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASETS=("ecthr_a" "ecthr_b" "eurlex")

for DS in "${DATASETS[@]}"; do
    echo "========================================================"
    echo "Running Qwen-4B RAG Scaling on ${DS}..."
    echo "========================================================"
    
    python run_data_scaling_experiment.py \
        --model_name_or_path "Qwen/Qwen3-Embedding-4B" \
        --dataset_name "coastalcph/lex_glue" \
        --dataset_config_name "${DS}" \
        --data_sizes "100,500,1000,2000,4500,9000,20000" \
        --output_dir "./output/data_scaling_results_4b" \
        --max_seq_length 4096 \
        --batch_size 4 \
        --k 10 \
        --threshold 0.4 \
        --bf16 true
done

