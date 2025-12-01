#!/bin/bash

echo "=== 1. Evaluating Baselines (Untrained / Zero-shot) ==="
echo "Running Baseline BERT..."
./scripts/eval_baseline_bert.sh
# echo "Running Baseline Qwen..."
# ./scripts/eval_baseline_qwen.sh
# echo "Running Baseline Hierarchical BERT..."
# ./scripts/eval_baseline_hierarchical_bert.sh
# echo "Running Baseline Hierarchical Qwen..."
# ./scripts/eval_baseline_hierarchical_qwen.sh

# echo -e "\n=== 2. Evaluating Fine-Tuned Models ==="
# echo "Running Fine-Tuned BERT..."
# ./scripts/eval_bert.sh
# echo "Running Fine-Tuned Qwen..."
# ./scripts/eval_qwen.sh
# echo "Running Fine-Tuned Hierarchical BERT..."
# ./scripts/eval_hierarchical_bert.sh
# echo "Running Fine-Tuned Hierarchical Qwen..."
# ./scripts/eval_hierarchical_qwen.sh

echo -e "\nAll evaluations completed."
