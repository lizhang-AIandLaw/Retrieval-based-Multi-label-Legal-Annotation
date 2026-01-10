import os
import json
import argparse
import logging
import random
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_environment(env_path: str):
    """Load environment variables from .env file."""
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    else:
        logger.warning(f"No .env file found at {env_path}. Relying on system env vars.")

def truncate_text(text: str, max_words: int = 6000) -> str:
    """Truncate text to a maximum number of words to avoid context limit issues."""
    words = text.split()
    if len(words) > max_words:
        # Keep first 2/3 and last 1/3 of the allowed budget
        head_limit = int(max_words * 0.66)
        tail_limit = max_words - head_limit
        return " ".join(words[:head_limit]) + " ... [TRUNCATED] ... " + " ".join(words[-tail_limit:])
    return text

def get_label_names(dataset) -> List[str]:
    """Extract label names from the dataset features."""
    try:
        # Common structure for Multi-label datasets in HF
        if hasattr(dataset['train'].features['labels'], 'feature'):
            return dataset['train'].features['labels'].feature.names
        # If it's a simple ClassLabel (unlikely for multi-label but possible in some formats)
        if hasattr(dataset['train'].features['labels'], 'names'):
            return dataset['train'].features['labels'].names
            
        logger.warning("Could not automatically extract label names from features. Checking first example.")
        return [] 
    except Exception as e:
        logger.error(f"Error extracting label names: {e}")
        return []

def format_few_shot_examples(examples: List[Dict], label_names: List[str]) -> str:
    """Format few-shot examples for the prompt."""
    formatted = ""
    for ex in examples:
        labels = [label_names[i] for i in ex['labels']]
        text = truncate_text(ex['text'], max_words=1000) # Shorter limit for examples
        formatted += f"Text: {text}\nLabels: {json.dumps(labels)}\n\n"
    return formatted

def construct_prompt(text: str, label_names: List[str], few_shot_examples: str = "") -> str:
    """Construct the prompt for the LLM."""
    
    prompt = f"""You are a legal expert AI assistant. Your task is to perform multi-label classification on legal documents.

The possible labels are:
{json.dumps(label_names, indent=2)}

"""
    if few_shot_examples:
        prompt += f"""Here are some examples of correctly classified documents:
{few_shot_examples}
"""

    prompt += f"""Now, classify the following document.
1. Read the text carefully.
2. Select ALL applicable labels from the provided list.
3. Return the result strictly as a valid JSON object with a single key "labels" containing a list of strings.

Text:
{text}

Output JSON:"""
    return prompt

def parse_llm_response(response: str, valid_labels: List[str]) -> List[int]:
    """Parse the LLM response and map labels back to indices."""
    try:
        # Clean markdown code blocks if present
        cleaned = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        
        pred_labels = data.get("labels", [])
        if not isinstance(pred_labels, list):
            logger.warning(f"Invalid format: 'labels' is not a list. Got: {pred_labels}")
            return []
            
        label_indices = []
        for l in pred_labels:
            if l in valid_labels:
                label_indices.append(valid_labels.index(l))
            else:
                logger.warning(f"Predicted label '{l}' not in valid label list.")
        
        return list(set(label_indices)) # Unique indices
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON: {response}")
        return []
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
        return []

def evaluate_predictions(references: List[List[int]], predictions: List[List[int]], num_classes: int):
    """Compute and print metrics."""
    mlb = MultiLabelBinarizer(classes=range(num_classes))
    y_true = mlb.fit_transform(references)
    y_pred = mlb.transform(predictions)
    
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    return micro_f1, macro_f1

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on Legal Classification Tasks")
    parser.add_argument("--dataset_name", type=str, default="coastalcph/lex_glue", help="Dataset name")
    parser.add_argument("--config_name", type=str, default="ecthr_a", help="Dataset config (ecthr_a, ecthr_b, eurlex)")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate on")
    parser.add_argument("--model", type=str, default="gpt-5.1-2025-11-13", help="OpenAI model name")
    parser.add_argument("--few_shot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of samples (for testing)")
    parser.add_argument("--output_file", type=str, default="llm_results.jsonl", help="Output file for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--env_file", type=str, default="./scripts/.env", help="Path to .env file")
    
    args = parser.parse_args()
    
    # Setup
    random.seed(args.seed)
    load_environment(args.env_file)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        return

    client = OpenAI(api_key=api_key)
    
    # Load Data
    logger.info(f"Loading dataset: {args.dataset_name} / {args.config_name}")
    dataset = load_dataset(args.dataset_name, args.config_name, trust_remote_code=True)
    
    label_names = get_label_names(dataset)
    if not label_names:
        logger.error("Could not determine label names. Aborting.")
        return
    logger.info(f"Found {len(label_names)} labels: {label_names}")
    
    eval_data = dataset[args.split]
    if args.limit > 0:
        eval_data = eval_data.select(range(min(len(eval_data), args.limit)))
        logger.info(f"Limited evaluation to {len(eval_data)} samples.")
    
    # Prepare Few-Shot Examples (if any)
    few_shot_context = ""
    if args.few_shot > 0:
        logger.info(f"Selecting {args.few_shot} few-shot examples from 'train' split.")
        train_data = dataset['train']
        # Simple random sampling; could be improved with retrieval
        indices = random.sample(range(len(train_data)), args.few_shot)
        examples = [train_data[i] for i in indices]
        few_shot_context = format_few_shot_examples(examples, label_names)

    results = []
    all_preds = []
    all_refs = []
    
    logger.info("Starting evaluation...")
    with open(args.output_file, 'w') as f_out:
        for idx, item in tqdm(enumerate(eval_data), total=len(eval_data)):
            text = truncate_text(item['text'])
            true_labels = item['labels']
            
            prompt = construct_prompt(text, label_names, few_shot_context)
            
            # Log the first prompt for verification
            if idx == 0:
                logger.info("="*30)
                logger.info("VERIFICATION: FIRST PROMPT GENERATED")
                logger.info("="*30)
                logger.info(prompt)
                logger.info("="*30)
            
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful legal expert assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0, # Deterministic
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                pred_indices = parse_llm_response(content, label_names)
                
                all_preds.append(pred_indices)
                all_refs.append(true_labels)
                
                result_entry = {
                    "idx": idx,
                    "true_labels": true_labels,
                    "pred_labels": pred_indices,
                    "pred_names": [label_names[i] for i in pred_indices],
                    "raw_response": content
                }
                
                f_out.write(json.dumps(result_entry) + "\n")
                f_out.flush() # Save progress
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                all_preds.append([]) # Treat as empty prediction on error
                all_refs.append(true_labels)

    # Compute Final Metrics
    micro, macro = evaluate_predictions(all_refs, all_preds, len(label_names))
    
    logger.info("="*30)
    logger.info(f"Results for {args.config_name} with {args.model}")
    logger.info(f"Micro-F1: {micro:.4f}")
    logger.info(f"Macro-F1: {macro:.4f}")
    logger.info("="*30)

    # Append summary to output file
    with open(args.output_file, 'a') as f_out:
        f_out.write(json.dumps({"summary": {"micro_f1": micro, "macro_f1": macro}}) + "\n")

if __name__ == "__main__":
    main()
