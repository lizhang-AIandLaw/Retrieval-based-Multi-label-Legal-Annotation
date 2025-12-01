#!/bin/bash

# Evaluate Hierarchical BERT on Test Split
python train.py configs/hierarchical_bert_config.json --do_train false --do_eval false --do_predict true --output_dir ./output/bert_hierarchical_ecthr_a_eval

