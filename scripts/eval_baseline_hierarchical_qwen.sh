#!/bin/bash

# Evaluate Original Qwen Base in Hierarchical Mode (Untrained Head)

python train.py configs/hierarchical_qwen_config.json \
    --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
    --do_train false \
    --do_eval false \
    --do_predict true \
    --load_best_model_at_end false \
    --output_dir ./output/qwen_hierarchical_baseline_eval

