#!/bin/bash

# Run BERT training
python train.py configs/bert_config.json --push_to_hub true --report_to wandb --project "legal-classification"

