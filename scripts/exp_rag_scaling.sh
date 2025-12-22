#!/bin/bash
# Terminal 1: Run RAG Scaling Experiment (Training-Free)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting RAG Scaling..."
python run_data_scaling_experiment.py \
    --model_name_or_path "Qwen/Qwen3-Embedding-0.6B" \
    --dataset_name "coastalcph/lex_glue" \
    --dataset_config_name "ecthr_a" \
    --data_sizes "100,500,1000,2000,4500,9000" \
    --output_dir "./output/data_scaling_results" \
    --max_seq_length 2048 \
    --batch_size 32 \
    --k 10 \
    --threshold 0.4 \
    --bf16 true

