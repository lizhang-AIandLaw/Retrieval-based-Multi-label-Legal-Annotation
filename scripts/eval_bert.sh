#!/bin/bash

# Evaluate BERT on Test Split (using local fine-tuned model)
python train.py configs/bert_config.json \
    --model_name_or_path ./output/bert_ecthr_a \
    --do_train false \
    --do_eval false \
    --do_predict true \
    --load_best_model_at_end false \
    --output_dir ./output/bert_ecthr_a_eval
