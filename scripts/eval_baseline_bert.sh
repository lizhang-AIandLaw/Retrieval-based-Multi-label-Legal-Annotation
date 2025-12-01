#!/bin/bash

# Evaluate Original BERT Base (Untrained Head / Zero-shot Baseline)
# Note: Since this model has not been fine-tuned on the dataset, 
# the classification head is randomly initialized. Results will be near random.

python train.py configs/bert_config.json \
    --model_name_or_path nlpaueb/legal-bert-base-uncased \
    --do_train false \
    --do_eval false \
    --do_predict true \
    --load_best_model_at_end false \
    --output_dir ./output/bert_base_baseline_eval

