#!/bin/bash

# Evaluate Hierarchical Qwen on Test Split
python train.py configs/hierarchical_qwen_config.json --do_train false --do_eval false --do_predict true --output_dir ./output/qwen_hierarchical_ecthr_a_eval

