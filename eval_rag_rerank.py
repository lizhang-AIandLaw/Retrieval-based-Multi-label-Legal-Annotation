import logging
import os
import sys
import json
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    HfArgumentParser, 
    AutoModelForSequenceClassification
)
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

@dataclass
class EvalConfig:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained embedding model (Retriever)"}
    )
    reranker_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained reranker model"}
    )
    dataset_name: str = field(
        default="coastalcph/lex_glue",
        metadata={"help": "The name of the dataset to use."}
    )
    dataset_config_name: str = field(
        default="ecthr_a",
        metadata={"help": "The configuration name of the dataset to use."}
    )
    cache_dir: str = field(
        default="./output/rag_results/cache",
        metadata={"help": "Directory to cache embeddings."}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for encoding."}
    )
    k: int = field(
        default=10,
        metadata={"help": "Number of neighbors to use for classification after reranking."}
    )
    retrieve_top_n: int = field(
        default=50,
        metadata={"help": "Number of candidates to retrieve before reranking."}
    )
    threshold: float = field(
        default=0.4,
        metadata={"help": "Threshold for multi-label classification voting."}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bf16."}
    )

def encode_dataset(model, tokenizer, dataset, max_length, batch_size, device, desc="Encoding"):
    """
    Encodes a dataset into a matrix of embeddings.
    """
    model.eval()
    all_embeddings = []
    
    # Process in batches
    batch_text = []
    
    for i in tqdm(range(0, len(dataset), batch_size), desc=desc):
        batch = dataset[i : i + batch_size]
        texts = batch["text"] # List of list of strings for ecthr_a
        # Flatten list of strings to single string
        texts = [" ".join(t) for t in texts]
        
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use last token embedding for Qwen (Decoder-only)
            # Assuming padding is on the right, we need to find the last non-pad token
            if tokenizer.padding_side == "left":
                embeddings = outputs.last_hidden_state[:, -1, :]
            else:
                 # Check for padding token
                sequence_lengths = (inputs.attention_mask.sum(dim=1) - 1)
                embeddings = outputs.last_hidden_state[torch.arange(outputs.last_hidden_state.shape[0]), sequence_lengths]
                
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
            
    return np.vstack(all_embeddings)

def get_rerank_scores(reranker, tokenizer, query_text, candidate_texts, max_length, device, batch_size=16):
    """
    Computes relevance scores for (query, candidate) pairs using the Cross-Encoder Reranker.
    """
    reranker.eval()
    scores = []
    
    pairs = [[query_text, cand] for cand in candidate_texts]
    
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            inputs = tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            
            outputs = reranker(**inputs)
            # Reranker outputs logits, usually a single value for regression or binary classification
            # For Qwen-Reranker, it usually outputs a single logit.
            batch_scores = outputs.logits.view(-1).float()
            # Apply sigmoid if trained with BCE, but raw logits are fine for ranking
            scores.extend(batch_scores.cpu().numpy())
            
    return np.array(scores)

def main():
    parser = HfArgumentParser((EvalConfig,))
    config = parser.parse_args_into_dataclasses()[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. Load Data
    logger.info(f"Loading dataset: {config.dataset_name} / {config.dataset_config_name}")
    dataset = load_dataset(config.dataset_name, config.dataset_config_name)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"] # or validation
    
    # 2. Setup Retriever (Embedding Model)
    logger.info(f"Loading Retriever: {config.model_name_or_path}")
    retriever_tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    if retriever_tokenizer.pad_token is None:
        retriever_tokenizer.pad_token = retriever_tokenizer.eos_token
        
    retriever_model = AutoModel.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32
    ).to(device)

    # 3. Setup Reranker
    logger.info(f"Loading Reranker: {config.reranker_model_name_or_path}")
    reranker_tokenizer = AutoTokenizer.from_pretrained(config.reranker_model_name_or_path, trust_remote_code=True)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(
        config.reranker_model_name_or_path,
        trust_remote_code=True,
        num_labels=1,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32
    ).to(device)
    
    # 4. Encode/Cache Training Set (Knowledge Base)
    os.makedirs(config.cache_dir, exist_ok=True)
    train_cache_path = os.path.join(config.cache_dir, f"train_embeddings_{config.dataset_config_name}.npy")
    
    if os.path.exists(train_cache_path):
        logger.info(f"Loading cached training embeddings from {train_cache_path}")
        train_embeddings = np.load(train_cache_path)
    else:
        logger.info("Encoding training set...")
        train_embeddings = encode_dataset(retriever_model, retriever_tokenizer, train_dataset, config.max_seq_length, config.batch_size, device)
        np.save(train_cache_path, train_embeddings)
        
    # 5. Encode Test Set (Queries)
    logger.info("Encoding test set...")
    test_embeddings = encode_dataset(retriever_model, retriever_tokenizer, test_dataset, config.max_seq_length, config.batch_size, device, desc="Encoding Queries")
    
    # 6. Evaluation Loop with Retrieval + Reranking
    logger.info(f"Starting Evaluation: Retrieve Top-{config.retrieve_top_n} -> Rerank -> Top-{config.k}")
    
    # Convert training labels to matrix
    num_classes = 10 # for ecthr_a
    y_train = np.zeros((len(train_dataset), num_classes))
    for i, ex in enumerate(train_dataset):
        for label in ex["labels"]:
            y_train[i, label] = 1
            
    y_true = []
    y_pred = []
    
    # Iterate over each test query
    # Doing this one by one or in small batches because Reranking is expensive
    
    for i in tqdm(range(len(test_dataset)), desc="Processing Queries"):
        query_vec = test_embeddings[i].reshape(1, -1) # (1, dim)
        
        # --- Step 1: Retrieval ---
        # Compute similarities with all training docs
        sims = np.dot(train_embeddings, query_vec.T).flatten() # (num_train,)
        
        # Get Top-N candidates
        top_n_indices = np.argsort(sims)[-config.retrieve_top_n:][::-1]
        
        # --- Step 2: Reranking ---
        query_text = " ".join(test_dataset[i]["text"])
        candidate_texts = [" ".join(train_dataset[int(idx)]["text"]) for idx in top_n_indices]
        
        # Compute reranker scores
        rerank_scores = get_rerank_scores(
            reranker_model, 
            reranker_tokenizer, 
            query_text, 
            candidate_texts, 
            max_length=512, # Reranker usually has shorter context limit or is very slow on long context
            device=device
        )
        
        # Sort by reranker scores
        # We need to map back to the original training indices
        reranked_local_indices = np.argsort(rerank_scores)[-config.k:][::-1]
        final_top_k_indices = [top_n_indices[idx] for idx in reranked_local_indices]
        final_scores = rerank_scores[reranked_local_indices]
        
        # --- Step 3: k-NN Prediction ---
        # Weighted vote
        class_votes = np.zeros(num_classes)
        total_weight = 0.0
        
        for rank, idx in enumerate(final_top_k_indices):
            # Use reranker score as weight (or could use rank weight)
            # Ensure score is positive for weighting (sigmoid)
            weight = 1.0 / (1.0 + np.exp(-final_scores[rank])) # simple sigmoid
            
            labels = train_dataset[int(idx)]["labels"]
            for label in labels:
                class_votes[label] += weight
            total_weight += weight
            
        # Normalize
        if total_weight > 0:
            class_probs = class_votes / total_weight
        else:
            class_probs = class_votes
            
        # Threshold
        predictions = (class_probs > config.threshold).astype(int)
        
        # Handle zero prediction case (fallback to top class)
        if predictions.sum() == 0:
            top_class = np.argmax(class_probs)
            predictions[top_class] = 1
            
        y_pred.append(predictions)
        
        # True labels
        true_labels = np.zeros(num_classes)
        for label in test_dataset[i]["labels"]:
            true_labels[label] = 1
        y_true.append(true_labels)
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Metrics
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    logger.info(f"Results for RAG + Rerank (k={config.k}, N={config.retrieve_top_n}):")
    logger.info(f"Micro-F1: {micro_f1:.4f}")
    logger.info(f"Macro-F1: {macro_f1:.4f}")
    
    # Save results
    output_file = os.path.join(config.cache_dir, "rerank_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "k": config.k,
            "retrieve_top_n": config.retrieve_top_n,
            "threshold": config.threshold
        }, f, indent=4)
        
if __name__ == "__main__":
    main()

