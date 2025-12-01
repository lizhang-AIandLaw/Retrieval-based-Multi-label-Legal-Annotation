#!/bin/bash

# Evaluate Qwen on Test Split
python train.py configs/qwen_config.json --do_train false --do_eval false --do_predict true --output_dir ./output/qwen_ecthr_a_eval

