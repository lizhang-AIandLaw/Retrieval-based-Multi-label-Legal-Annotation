#!/bin/bash

# Evaluate Original BERT Base in Hierarchical Mode (Untrained Head)

python train.py configs/hierarchical_bert_config.json \
    --model_name_or_path nlpaueb/legal-bert-base-uncased \
    --do_train false \
    --do_eval false \
    --do_predict true \
    --load_best_model_at_end false \
    --output_dir ./output/bert_hierarchical_baseline_eval

