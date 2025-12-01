#!/bin/bash

# Evaluate Hierarchical Qwen on Test Split
python train.py configs/hierarchical_qwen_config.json \
    --model_name_or_path ./output/qwen_hierarchical_ecthr_a \
    --do_train false \
    --do_eval false \
    --do_predict true \
    --load_best_model_at_end false \
    --output_dir ./output/qwen_hierarchical_ecthr_a_eval
