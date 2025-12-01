#!/bin/bash

# Run Hierarchical BERT training
python train.py configs/hierarchical_bert_config.json --push_to_hub true --report_to wandb --project "legal-classification"

