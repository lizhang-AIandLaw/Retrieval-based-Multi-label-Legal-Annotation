import logging
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, HfArgumentParser
from sklearn.metrics import f1_score

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

@dataclass
class ExpConfig:
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
    data_sizes: str = field(
        default="100,500,1000,2000,4500,9000",
        metadata={"help": "Comma-separated list of training set sizes to evaluate."}
    )
    output_dir: str = field(
        default="./output/data_scaling_results",
        metadata={"help": "Directory to save results."}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "The maximum total input sequence length."}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for encoding."}
    )
    k: int = field(
        default=10,
        metadata={"help": "k for k-NN."}
    )
    threshold: float = field(
        default=0.4,
        metadata={"help": "Threshold for multi-label classification."}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bf16."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for data sampling."}
    )

def encode_dataset(model, tokenizer, dataset, max_length, batch_size, device):
    model.eval()
    all_embeddings = []
    
    # Process text in batches
    # ecthr_a 'text' is list of strings, join them
    texts = [" ".join(ex["text"]) for ex in dataset]
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
        batch_texts = texts[i : i + batch_size]
        
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Last token pooling for Qwen (padding side left usually)
            if tokenizer.padding_side == "left":
                embeddings = outputs.last_hidden_state[:, -1, :]
            else:
                # If right padded, find last non-pad token
                sequence_lengths = (inputs.attention_mask.sum(dim=1) - 1)
                embeddings = outputs.last_hidden_state[torch.arange(outputs.last_hidden_state.shape[0]), sequence_lengths]
                
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.float().cpu().numpy())
            
    return np.vstack(all_embeddings)

def run_rag_eval(train_embeddings, train_labels, test_embeddings, test_labels, k, threshold):
    """
    Run k-NN retrieval and evaluation.
    """
    num_test = test_embeddings.shape[0]
    num_classes = 10 # Hardcoded for ecthr_a for now, ideally dynamic
    
    y_pred = []
    y_true = []
    
    # Process test queries in chunks to save memory
    chunk_size = 100
    
    for i in range(0, num_test, chunk_size):
        end_idx = min(i + chunk_size, num_test)
        query_chunk = test_embeddings[i:end_idx] # (chunk, dim)
        
        # Sim matrix: (chunk, num_train)
        sims = np.dot(query_chunk, train_embeddings.T)
        
        # Top-k indices: (chunk, k)
        # argpartition is faster than argsort for Top-k
        top_k_indices = np.argpartition(sims, -k, axis=1)[:, -k:]
        
        # We need sorted scores for weighting? Actually simple sum is robust enough or simple weight
        # Let's do simple sum of labels weighted by similarity
        
        batch_preds = np.zeros((end_idx - i, num_classes))
        
        for r in range(end_idx - i):
            indices = top_k_indices[r] # indices in train set
            scores = sims[r, indices] # similarities
            
            # Weighted Vote
            for idx, score in zip(indices, scores):
                # Ensure positive weight
                weight = max(score, 0)
                labels = train_labels[idx]
                for label in labels:
                    batch_preds[r, label] += weight
            
            # Normalize? Or just threshold raw sum?
            # Standard: prob = sum(weights * label) / sum(weights)
            # But here sum(weights) varies. 
            # Let's normalize by sum of weights (Top-k sum)
            total_weight = np.sum(scores[scores > 0]) + 1e-9
            batch_preds[r] /= total_weight
            
        # Apply threshold
        binary_preds = (batch_preds > threshold).astype(int)
        
        # Fallback: if no label predicted, pick max
        for r in range(len(binary_preds)):
            if binary_preds[r].sum() == 0:
                top_c = np.argmax(batch_preds[r])
                binary_preds[r, top_c] = 1
                
        y_pred.extend(binary_preds)
        
        # True labels
        chunk_true_labels = test_labels[i:end_idx]
        batch_true = np.zeros((len(chunk_true_labels), num_classes))
        for r, labels in enumerate(chunk_true_labels):
            for label in labels:
                batch_true[r, label] = 1
        y_true.extend(batch_true)
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    micro = f1_score(y_true, y_pred, average='micro')
    macro = f1_score(y_true, y_pred, average='macro')
    
    return micro, macro

def main():
    parser = HfArgumentParser((ExpConfig,))
    config = parser.parse_args_into_dataclasses()[0]
    
    # Setup
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load Data
    dataset = load_dataset(config.dataset_name, config.dataset_config_name)
    full_train_set = dataset["train"]
    test_set = dataset["test"] # or validation
    
    # Load Model
    logger.info(f"Loading Model: {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModel.from_pretrained(
        config.model_name_or_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32
    ).to(device)
    
    # 1. Encode Test Set (Once)
    logger.info("Encoding Test Set...")
    test_embeddings = encode_dataset(model, tokenizer, test_set, config.max_seq_length, config.batch_size, device)
    test_labels = [ex["labels"] for ex in test_set]
    
    # 2. Parse Data Sizes
    sizes = [int(s) for s in config.data_sizes.split(",")]
    # Ensure sizes don't exceed dataset
    sizes = [s for s in sizes if s <= len(full_train_set)]
    if len(full_train_set) not in sizes:
        sizes.append(len(full_train_set)) # Always run full set
    sizes = sorted(list(set(sizes)))
    
    logger.info(f"Running experiments for sizes: {sizes}")
    
    results = []
    
    # 3. Main Loop
    # Optimization: Encode full train set once, then slice embeddings?
    # Yes! That saves huge time.
    logger.info("Encoding Full Training Set (for slicing)...")
    full_train_embeddings = encode_dataset(model, tokenizer, full_train_set, config.max_seq_length, config.batch_size, device)
    full_train_labels = [ex["labels"] for ex in full_train_set]
    
    # Set seed for reproducibility of shuffling
    np.random.seed(config.seed)
    # Generate a random permutation of indices
    permuted_indices = np.random.permutation(len(full_train_set))
    
    for size in sizes:
        logger.info(f"--- Evaluating Size: {size} ---")
        
        # Slice the pre-computed embeddings
        # We select the first 'size' indices from the permutation to simulate random sampling
        subset_indices = permuted_indices[:size]
        
        train_emb_subset = full_train_embeddings[subset_indices]
        train_lbl_subset = [full_train_labels[i] for i in subset_indices]
        
        micro, macro = run_rag_eval(
            train_emb_subset, 
            train_lbl_subset, 
            test_embeddings, 
            test_labels, 
            config.k, 
            config.threshold
        )
        
        logger.info(f"Size {size}: Micro={micro:.4f}, Macro={macro:.4f}")
        results.append({
            "train_size": size,
            "micro_f1": micro,
            "macro_f1": macro
        })
        
    # 4. Save Results
    df = pd.DataFrame(results)
    save_path = os.path.join(config.output_dir, f"scaling_results_{config.dataset_config_name}.csv")
    df.to_csv(save_path, index=False)
    logger.info(f"Results saved to {save_path}")
    print(df)

if __name__ == "__main__":
    main()

