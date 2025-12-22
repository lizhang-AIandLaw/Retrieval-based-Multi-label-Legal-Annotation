import logging
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass, field
from typing import List
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    HfArgumentParser
)
from sklearn.metrics import f1_score

# Reuse the encoding logic from eval_rag.py (simplified here)
# Ideally we would import it, but to be standalone and robust to path changes, we redefine helper
# Or we can import if eval_rag is in path. Let's redefine for safety.

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

@dataclass
class DataEffConfig:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained embedding model"}
    )
    dataset_name: str = field(
        default="coastalcph/lex_glue",
        metadata={"help": "The name of the dataset to use."}
    )
    dataset_config_name: str = field(
        default="ecthr_a",
        metadata={"help": "The configuration name of the dataset to use."}
    )
    output_dir: str = field(
        default="./output/data_efficiency",
        metadata={"help": "Directory to save results and plots."}
    )
    # Define data sizes as a list of integers
    # We will pass this as a string "100,500,1000" and parse it
    data_sizes_str: str = field(
        default="100,500,1000,2000,4500,9000",
        metadata={"help": "Comma-separated list of training set sizes to evaluate."}
    )
    max_seq_length: int = field(default=2048)
    batch_size: int = field(default=32)
    k: int = field(default=10)
    threshold: float = field(default=0.4)
    bf16: bool = field(default=True)
    seed: int = field(default=42)

def encode_dataset(model, tokenizer, dataset, max_length, batch_size, device):
    model.eval()
    all_embeddings = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Encoding", leave=False):
        batch = dataset[i : i + batch_size]
        texts = [" ".join(t) for t in batch["text"]]
        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Last token pooling for Qwen
            if tokenizer.padding_side == "left":
                embeddings = outputs.last_hidden_state[:, -1, :]
            else:
                sequence_lengths = (inputs.attention_mask.sum(dim=1) - 1)
                embeddings = outputs.last_hidden_state[torch.arange(outputs.last_hidden_state.shape[0]), sequence_lengths]
            
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.float().cpu().numpy())
    return np.vstack(all_embeddings)

def evaluate_rag(train_embeddings, train_dataset, test_embeddings, test_dataset, k, threshold):
    # k-NN prediction
    num_classes = 10 # Hardcoded for ecthr_a, logic needs adaptation for generic
    # Check max label in dataset just in case
    
    y_true = []
    y_pred = []
    
    # Pre-compute train labels matrix for speed
    train_labels_matrix = np.zeros((len(train_dataset), num_classes))
    for i, ex in enumerate(train_dataset):
        for label in ex["labels"]:
            train_labels_matrix[i, label] = 1
            
    # Compute similarity matrix (Test x Train)
    # Caution: If Train is large, this might OOM. But for 9k it's fine.
    # 1k x 9k matrix = 9M floats = 36MB. Safe.
    sim_matrix = np.dot(test_embeddings, train_embeddings.T) # (num_test, num_train)
    
    for i in range(len(test_dataset)):
        # Get top-k neighbors
        sims = sim_matrix[i]
        top_k_indices = np.argsort(sims)[-k:][::-1]
        top_k_scores = sims[top_k_indices]
        
        # Weighted vote
        class_votes = np.zeros(num_classes)
        total_weight = 0.0
        
        for rank, idx in enumerate(top_k_indices):
            weight = top_k_scores[rank] # Cosine sim as weight
            # Optional: threshold weight to avoid noise
            if weight < 0: weight = 0
            
            class_votes += train_labels_matrix[idx] * weight
            total_weight += weight
            
        if total_weight > 0:
            class_probs = class_votes / total_weight
        else:
            class_probs = class_votes
            
        predictions = (class_probs > threshold).astype(int)
        if predictions.sum() == 0:
            predictions[np.argmax(class_probs)] = 1
            
        y_pred.append(predictions)
        
        true_labels = np.zeros(num_classes)
        for label in test_dataset[i]["labels"]:
            true_labels[label] = 1
        y_true.append(true_labels)
        
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return micro_f1, macro_f1

def main():
    parser = HfArgumentParser((DataEffConfig,))
    config = parser.parse_args_into_dataclasses()[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 1. Load Model
    logger.info(f"Loading Model: {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        config.model_name_or_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32
    ).to(device)
    
    # 2. Load Full Data
    logger.info(f"Loading Dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name, config.dataset_config_name)
    full_train_dataset = dataset["train"].shuffle(seed=config.seed) # Shuffle to ensure random subset
    test_dataset = dataset["test"] # or validation
    
    # 3. Encode Test Set (Once)
    logger.info("Encoding Test Set...")
    test_embeddings = encode_dataset(model, tokenizer, test_dataset, config.max_seq_length, config.batch_size, device)
    
    # 4. Encode Full Training Set (Once)
    # We encode all 9k first, then slice the embeddings. 
    # This is much faster than re-encoding for every subset.
    logger.info("Encoding Full Training Set...")
    full_train_embeddings = encode_dataset(model, tokenizer, full_train_dataset, config.max_seq_length, config.batch_size, device)
    
    # 5. Iterate over Data Sizes
    data_sizes = [int(s) for s in config.data_sizes_str.split(",")]
    # Ensure sizes don't exceed actual data
    data_sizes = [s for s in data_sizes if s <= len(full_train_dataset)]
    if len(full_train_dataset) not in data_sizes:
        data_sizes.append(len(full_train_dataset))
    
    results = []
    
    logger.info(f"Starting Data Efficiency Loop: {data_sizes}")
    
    for size in data_sizes:
        logger.info(f"Evaluating subset size: {size}")
        
        # Slice embeddings and dataset
        # Since we shuffled, first N is a random subset
        train_embeddings_subset = full_train_embeddings[:size]
        train_dataset_subset = full_train_dataset.select(range(size))
        
        micro, macro = evaluate_rag(
            train_embeddings_subset, 
            train_dataset_subset, 
            test_embeddings, 
            test_dataset, 
            config.k, 
            config.threshold
        )
        
        logger.info(f"Size {size}: Micro-F1={micro:.4f}, Macro-F1={macro:.4f}")
        results.append({
            "size": size,
            "micro_f1": micro,
            "macro_f1": macro
        })
        
    # 6. Save Results & Plot
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(config.output_dir, "data_efficiency_results.csv"), index=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df["size"], df["micro_f1"], marker='o', label='Micro-F1')
    plt.plot(df["size"], df["macro_f1"], marker='s', label='Macro-F1')
    plt.xlabel('Training Data Size (Number of Documents)')
    plt.ylabel('F1 Score')
    plt.title(f'RAG Performance vs. Data Size ({config.model_name_or_path})')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(config.output_dir, "learning_curve.png"))
    logger.info(f"Results saved to {config.output_dir}")

if __name__ == "__main__":
    main()

