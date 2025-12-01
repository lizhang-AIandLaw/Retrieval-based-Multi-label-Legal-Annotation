#!/bin/bash

# Evaluate BERT on Test Split
python train.py configs/bert_config.json --do_train false --do_eval false --do_predict true --output_dir ./output/bert_ecthr_a_eval

