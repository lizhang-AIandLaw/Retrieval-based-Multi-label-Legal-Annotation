#!/bin/bash

# Full Evaluation Script for LLM on Legal Classification
# WARNING: This will incur costs on the OpenAI API.
# It runs on the FULL test set for all 3 datasets.

# Exit on error
set -e

# Load environment variables
source ./scripts/.env

MODEL="gpt-5.1-2025-11-13"
OUTPUT_DIR="output/llm_results_$(date +%Y%m%d)"
mkdir -p "$OUTPUT_DIR"

echo "Results will be saved to: $OUTPUT_DIR"

# 1. ECtHR A (Zero-shot)
echo "========================================================"
echo "Starting ECtHR A Evaluation"
echo "========================================================"
python eval_llm.py \
    --dataset_name coastalcph/lex_glue \
    --config_name ecthr_a \
    --model "$MODEL" \
    --output_file "$OUTPUT_DIR/ecthr_a.jsonl" \
    --env_file ./scripts/.env

# 2. ECtHR B (Zero-shot)
echo "========================================================"
echo "Starting ECtHR B Evaluation"
echo "========================================================"
python eval_llm.py \
    --dataset_name coastalcph/lex_glue \
    --config_name ecthr_b \
    --model "$MODEL" \
    --output_file "$OUTPUT_DIR/ecthr_b.jsonl" \
    --env_file ./scripts/.env

# 3. Eurlex (Zero-shot)
# Note: Using Zero-shot by default as discussed.
# If Few-shot is desired, add --few_shot 1 or 2
echo "========================================================"
echo "Starting Eurlex Evaluation"
echo "========================================================"
python eval_llm.py \
    --dataset_name coastalcph/lex_glue \
    --config_name eurlex \
    --model "$MODEL" \
    --output_file "$OUTPUT_DIR/eurlex.jsonl" \
    --env_file ./scripts/.env

echo "========================================================"
echo "All Evaluations Complete!"
echo "Summary:"
tail -n 1 "$OUTPUT_DIR/"*.jsonl
echo "========================================================"
