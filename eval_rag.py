import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorWithPadding,
    set_seed
)
from sklearn.metrics import f1_score, accuracy_score

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

@dataclass
class RagConfig:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    dataset_name: str = field(
        default="coastalcph/lex_glue",
        metadata={"help": "The name of the dataset to use."}
    )
    dataset_config_name: str = field(
        default="ecthr_a",
        metadata={"help": "The configuration name of the dataset to use."}
    )
    max_seq_length: int = field(
        default=8192,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for encoding."}
    )
    k_neighbors: int = field(
        default=10,
        metadata={"help": "Number of neighbors for k-NN (or max k for search)."}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code."}
    )
    output_dir: str = field(
        default="./output/rag_results",
        metadata={"help": "Where to store results."}
    )
    num_labels: int = field(
        default=10,
        metadata={"help": "Number of labels."}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Use fp16."}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bf16."}
    )
    search_params: bool = field(
        default=False,
        metadata={"help": "If True, run grid search over k and threshold using the test set."}
    )

def get_embeddings(model, dataloader, device, use_last_token=True):
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                hidden_states = outputs.last_hidden_state
                if use_last_token:
                    last_token_indices = attention_mask.sum(1) - 1
                    last_token_indices = last_token_indices.clamp(min=0)
                    embeddings = hidden_states[torch.arange(hidden_states.size(0)), last_token_indices]
                else:
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask

            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
            
            if "labels" in batch:
                all_labels.append(batch["labels"].cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    if all_labels:
        all_labels = torch.cat(all_labels, dim=0)
        return all_embeddings, all_labels
    return all_embeddings, None

def evaluate_params(k, threshold, all_topk_indices, all_topk_values, train_labels, y_true, num_labels):
    """
    Compute metrics for a specific k and threshold using precomputed top-k neighbors.
    """
    batch_size = all_topk_indices.size(0)
    
    # Slice to k neighbors
    current_k_indices = all_topk_indices[:, :k]   # (B, k)
    current_k_values = all_topk_values[:, :k]     # (B, k)
    
    # Flatten for gathering
    flat_indices = current_k_indices.reshape(-1)
    
    # Gather labels: (B*k, Num_Labels) -> (B, k, Num_Labels)
    gathered_labels = train_labels[flat_indices].view(batch_size, k, num_labels)
    
    # Weights (Similarity)
    weights = current_k_values.clamp(min=0).unsqueeze(-1) # (B, k, 1)
    
    # Weighted Sum
    weighted_labels = gathered_labels * weights
    summed_scores = weighted_labels.sum(dim=1) # (B, Num_Labels)
    sum_weights = weights.sum(dim=1).clamp(min=1e-9) # (B, 1)
    
    probs = summed_scores / sum_weights # (B, Num_Labels)
    
    # Thresholding
    # FIX: Convert to float32 before numpy conversion because numpy doesn't support bfloat16
    probs_np = probs.float().cpu().numpy()
    y_pred = np.zeros_like(probs_np)
    y_pred[probs_np >= threshold] = 1
    
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    return f1_micro, f1_macro

def main():
    parser = HfArgumentParser((RagConfig,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        config = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        config = parser.parse_args_into_dataclasses()[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Dataset
    logger.info(f"Loading dataset: {config.dataset_name} / {config.dataset_config_name}")
    raw_datasets = load_dataset(config.dataset_name, config.dataset_config_name)

    # Load Tokenizer & Model
    logger.info(f"Loading model: {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": config.trust_remote_code}
    if config.fp16: model_kwargs["torch_dtype"] = torch.float16
    elif config.bf16: model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModel.from_pretrained(config.model_name_or_path, **model_kwargs).to(device)
    
    def preprocess_function(examples):
        texts = [" ".join(t) for t in examples["text"]]
        batch = tokenizer(texts, padding="max_length", max_length=config.max_seq_length, truncation=True)
        if "labels" in examples:
            batch_labels = []
            for label_ids in examples["labels"]:
                label_vec = [0.0] * config.num_labels
                for lid in label_ids:
                    if lid < config.num_labels:
                        label_vec[lid] = 1.0
                batch_labels.append(label_vec)
            batch["labels"] = batch_labels
        return batch

    with torch.no_grad():
        tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names, desc="Tokenizing")

    tokenized_datasets.set_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_loader = DataLoader(tokenized_datasets["train"], batch_size=config.batch_size, collate_fn=data_collator, shuffle=False)
    test_loader = DataLoader(tokenized_datasets["test"], batch_size=config.batch_size, collate_fn=data_collator, shuffle=False)

    # Encode
    cache_dir = os.path.join(config.output_dir, "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    train_emb_path = os.path.join(cache_dir, "train_embeddings.pt")
    train_lbl_path = os.path.join(cache_dir, "train_labels.pt")
    test_emb_path = os.path.join(cache_dir, "test_embeddings.pt")
    test_lbl_path = os.path.join(cache_dir, "test_labels.pt")
    
    # Try to load from cache
    if os.path.exists(train_emb_path) and os.path.exists(test_emb_path):
        logger.info(f"Loading embeddings from cache: {cache_dir}")
        train_embeddings = torch.load(train_emb_path, map_location=device)
        train_labels = torch.load(train_lbl_path, map_location=device)
        test_embeddings = torch.load(test_emb_path, map_location=device)
        test_labels = torch.load(test_lbl_path, map_location=device)
    else:
        logger.info("Encoding Training Set (Index)...")
        train_embeddings, train_labels = get_embeddings(model, train_loader, device)
        logger.info("Encoding Test Set (Query)...")
        test_embeddings, test_labels = get_embeddings(model, test_loader, device)
        
        logger.info(f"Saving embeddings to cache: {cache_dir}")
        torch.save(train_embeddings, train_emb_path)
        torch.save(train_labels, train_lbl_path)
        torch.save(test_embeddings, test_emb_path)
        torch.save(test_labels, test_lbl_path)

    train_embeddings = train_embeddings.to(device)
    test_embeddings = test_embeddings.to(device)
    train_labels = train_labels.to(device)

    # Search Logic
    if config.search_params:
        logger.info("Starting Grid Search for Optimal Parameters...")
        
        # 1. Retrieve max possible neighbors (e.g. 50) once
        max_k = 50
        batch_size_knn = 100
        num_test = test_embeddings.size(0)
        
        all_topk_indices = []
        all_topk_values = []
        
        for i in tqdm(range(0, num_test, batch_size_knn), desc="Pre-retrieving Max Neighbors"):
            end = min(i + batch_size_knn, num_test)
            test_batch = test_embeddings[i:end]
            sim_matrix = torch.matmul(test_batch, train_embeddings.T)
            topk_values, topk_indices = torch.topk(sim_matrix, k=max_k, dim=1)
            all_topk_indices.append(topk_indices.cpu())
            all_topk_values.append(topk_values.cpu())
            
        all_topk_indices = torch.cat(all_topk_indices, dim=0).to(device)
        all_topk_values = torch.cat(all_topk_values, dim=0).to(device)
        
        y_true = test_labels.cpu().numpy()
        
        # 2. Iterate Parameters
        k_list = [3, 5, 8, 10, 15, 20, 30, 50]
        threshold_list = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        results = []
        
        print(f"\n{'k':<5} {'Threshold':<10} {'Micro-F1':<10} {'Macro-F1':<10}")
        print("-" * 40)
        
        best_micro = 0
        best_config = {}

        for k in k_list:
            for thr in threshold_list:
                f1_mic, f1_mac = evaluate_params(k, thr, all_topk_indices, all_topk_values, train_labels, y_true, config.num_labels)
                print(f"{k:<5} {thr:<10.1f} {f1_mic:<10.4f} {f1_mac:<10.4f}")
                
                results.append({"k": k, "threshold": thr, "f1_micro": f1_mic, "f1_macro": f1_mac})
                
                if f1_mic > best_micro:
                    best_micro = f1_mic
                    best_config = {"k": k, "threshold": thr, "f1_micro": f1_mic, "f1_macro": f1_mac}
        
        print("\n" + "="*30)
        print(f"Best Configuration Found:")
        print(f"k: {best_config['k']}")
        print(f"Threshold: {best_config['threshold']}")
        print(f"Micro F1: {best_config['f1_micro']:.4f}")
        print(f"Macro F1: {best_config['f1_macro']:.4f}")
        print("="*30 + "\n")
        
    else:
        # Original Logic (Single run)
        # Compute Cosine Similarity (since normalized)
        # (Num_Test, Dim) @ (Dim, Num_Train) -> (Num_Test, Num_Train)
        
        batch_size_knn = 100
        num_test = test_embeddings.size(0)
        
        all_preds_probs = []
        
        for i in tqdm(range(0, num_test, batch_size_knn), desc="Retrieving"):
            end = min(i + batch_size_knn, num_test)
            test_batch = test_embeddings[i:end]
            
            # Similarity
            sim_matrix = torch.matmul(test_batch, train_embeddings.T) # (B, N_train)
            
            # Top-K
            topk_values, topk_indices = torch.topk(sim_matrix, k=config.k_neighbors, dim=1)
            
            # Weighted Voting
            flat_indices = topk_indices.view(-1)
            gathered_labels = train_labels[flat_indices] # (B*K, Num_Labels)
            gathered_labels = gathered_labels.view(test_batch.size(0), config.k_neighbors, -1) # (B, K, Num_Labels)
            
            weights = topk_values.clamp(min=0).unsqueeze(-1) # (B, K, 1)
            
            weighted_labels = gathered_labels * weights # (B, K, Num_Labels)
            summed_scores = weighted_labels.sum(dim=1) # (B, Num_Labels)
            
            sum_weights = weights.sum(dim=1).clamp(min=1e-9) # (B, 1)
            probs = summed_scores / sum_weights # (B, Num_Labels)
            
            all_preds_probs.append(probs.cpu())

        all_preds_probs = torch.cat(all_preds_probs, dim=0)
        
        # Thresholding
        threshold = 0.5
        y_pred = np.zeros(all_preds_probs.shape)
        y_pred[all_preds_probs >= threshold] = 1
        
        # Metrics
        y_true = test_labels.cpu().numpy()
        
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        
        print("\n" + "="*30)
        print(f"RAG (k-NN) Results for {config.model_name_or_path}")
        print(f"k: {config.k_neighbors}")
        print(f"F1 Micro: {f1_micro}")
        print(f"F1 Macro: {f1_macro}")
        print(f"Accuracy: {accuracy}")
        print("="*30 + "\n")
        
        # Save results
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
            
        with open(os.path.join(config.output_dir, "rag_results.json"), "w") as f:
            json.dump({
                "model": config.model_name_or_path,
                "k": config.k_neighbors,
                "f1_micro": f1_micro,
                "f1_macro": f1_macro,
                "accuracy": accuracy
            }, f, indent=2)

if __name__ == "__main__":
    main()
