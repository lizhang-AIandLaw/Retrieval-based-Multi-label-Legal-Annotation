#!/bin/bash

# Evaluate Original Qwen Base (Untrained Head / Zero-shot Baseline)
# Note: Since this model has not been fine-tuned on the dataset, 
# the classification head is randomly initialized. Results will be near random.

python train.py configs/qwen_config.json \
    --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
    --do_train false \
    --do_eval false \
    --do_predict true \
    --load_best_model_at_end false \
    --output_dir ./output/qwen_base_baseline_eval

