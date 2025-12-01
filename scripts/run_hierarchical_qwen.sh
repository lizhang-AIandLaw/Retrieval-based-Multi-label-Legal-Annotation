#!/bin/bash

# Run Hierarchical Qwen training
python train.py configs/hierarchical_qwen_config.json --push_to_hub true --report_to wandb --project "legal-classification"

