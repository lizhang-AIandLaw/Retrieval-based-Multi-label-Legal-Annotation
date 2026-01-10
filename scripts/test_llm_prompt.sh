#!/bin/bash

# Test script to verify LLM prompts and output format
# Runs on a small subset (limit 3) to save costs and verify logic

# Load environment variables
source ./scripts/.env

# Directory setup
mkdir -p output/llm_test

# 1. Test Zero-shot on ECtHR A (10 labels)
echo "--------------------------------------------------------"
echo "Running Test 1: ECtHR A (Zero-shot, Limit 3)"
echo "--------------------------------------------------------"
python eval_llm.py \
    --dataset_name coastalcph/lex_glue \
    --config_name ecthr_a \
    --model gpt-5.1-2025-11-13 \
    --limit 3 \
    --output_file output/llm_test/test_ecthr_a.jsonl \
    --env_file ./scripts/.env

# 2. Test Few-shot on Eurlex (100 labels)
echo "--------------------------------------------------------"
echo "Running Test 2: Eurlex (Zero-shot, Limit 2)"
echo "--------------------------------------------------------"
python eval_llm.py \
    --dataset_name coastalcph/lex_glue \
    --config_name eurlex \
    --model gpt-5.1-2025-11-13 \
    # --few_shot 1 \
    --limit 2 \
    --output_file output/llm_test/test_eurlex_fewshot.jsonl \
    --env_file ./scripts/.env

echo "--------------------------------------------------------"
echo "Test Complete. Check logs above for the Prompt structure."
echo "Output files are in output/llm_test/"
echo "--------------------------------------------------------"
