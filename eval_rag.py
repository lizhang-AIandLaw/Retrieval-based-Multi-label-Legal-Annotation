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
        metadata={"help": "Number of neighbors for k-NN."}
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

def get_embeddings(model, dataloader, device, use_last_token=True):
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Pooling strategy
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                # BERT-like
                embeddings = outputs.pooler_output
            else:
                # Qwen/GPT-like: Use last token
                hidden_states = outputs.last_hidden_state
                if use_last_token:
                    # Find last non-padding token
                    # sum(mask) - 1
                    last_token_indices = attention_mask.sum(1) - 1
                    last_token_indices = last_token_indices.clamp(min=0)
                    embeddings = hidden_states[torch.arange(hidden_states.size(0)), last_token_indices]
                else:
                    # Mean pooling as fallback?
                    # Mask out padding
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask

            # Normalize embeddings for cosine similarity
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu())
            
            if "labels" in batch:
                all_labels.append(batch["labels"].cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    if all_labels:
        all_labels = torch.cat(all_labels, dim=0)
        return all_embeddings, all_labels
    return all_embeddings, None

def main():
    parser = HfArgumentParser((RagConfig,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        config = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        config = parser.parse_args_into_dataclasses()[0]

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Dataset
    logger.info(f"Loading dataset: {config.dataset_name} / {config.dataset_config_name}")
    raw_datasets = load_dataset(config.dataset_name, config.dataset_config_name)

    # Load Tokenizer & Model
    logger.info(f"Loading model: {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, 
        trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
    }
    if config.fp16:
        model_kwargs["torch_dtype"] = torch.float16
    elif config.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModel.from_pretrained(
        config.model_name_or_path,
        **model_kwargs
    ).to(device)
    
    # Preprocessing
    def preprocess_function(examples):
        # Tokenize
        texts = [" ".join(t) for t in examples["text"]]
        batch = tokenizer(
            texts,
            padding="max_length",
            max_length=config.max_seq_length,
            truncation=True,
        )
        
        # Process labels
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

    with torch.no_grad(): # Disable gradient tracking for mapping if any torch ops are used (unlikely but safe)
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Tokenizing"
        )

    # Set format to torch
    tokenized_datasets.set_format("torch")

    # Data Loaders
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_loader = DataLoader(
        tokenized_datasets["train"], 
        batch_size=config.batch_size, 
        collate_fn=data_collator,
        shuffle=False
    )
    
    test_loader = DataLoader(
        tokenized_datasets["test"], 
        batch_size=config.batch_size, 
        collate_fn=data_collator,
        shuffle=False
    )

    # 1. Encode Training Set (Knowledge Base)
    logger.info("Encoding Training Set (Building Index)...")
    train_embeddings, train_labels = get_embeddings(model, train_loader, device)
    logger.info(f"Training Embeddings Shape: {train_embeddings.shape}")

    # 2. Encode Test Set (Queries)
    logger.info("Encoding Test Set...")
    test_embeddings, test_labels = get_embeddings(model, test_loader, device)
    logger.info(f"Test Embeddings Shape: {test_embeddings.shape}")

    # 3. Retrieval & Classification
    logger.info(f"Running k-NN Classification (k={config.k_neighbors})...")
    
    # Move to GPU for fast matrix multiplication if possible
    train_embeddings = train_embeddings.to(device)
    test_embeddings = test_embeddings.to(device)
    train_labels = train_labels.to(device)
    
    # Compute Cosine Similarity (since normalized)
    # (Num_Test, Dim) @ (Dim, Num_Train) -> (Num_Test, Num_Train)
    # Chunking to avoid OOM if matrix is huge. 
    # 1000 x 9000 is small, but let's be safe for scaling.
    
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
        # We want to aggregate the labels of the neighbors weighted by their similarity.
        # Retrieve labels: (B, K, Num_Labels)
        # train_labels: (N_train, Num_Labels)
        # topk_indices: (B, K)
        
        # Gather labels
        # Expand indices to match label dimension? No, use advanced indexing
        # But gather needs indices in same dim.
        # Or just loop? No, vectorize.
        
        # topk_indices.view(-1) gives flat indices.
        flat_indices = topk_indices.view(-1)
        gathered_labels = train_labels[flat_indices] # (B*K, Num_Labels)
        gathered_labels = gathered_labels.view(test_batch.size(0), config.k_neighbors, -1) # (B, K, Num_Labels)
        
        # Weights: topk_values (B, K)
        # We can use Softmax on similarities to make them sum to 1, or just use raw similarities.
        # Raw similarity is better because it reflects absolute closeness.
        # But let's clamp negative similarities to 0 just in case.
        weights = topk_values.clamp(min=0).unsqueeze(-1) # (B, K, 1)
        
        # Weighted Sum
        weighted_labels = gathered_labels * weights # (B, K, Num_Labels)
        summed_scores = weighted_labels.sum(dim=1) # (B, Num_Labels)
        
        # Normalize by sum of weights to get "probability"
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

