import logging
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Union
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, HfArgumentParser
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.calibration import CalibratedClassifierCV

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
        metadata={"help": "Default k for k-NN (if not tuning)."}
    )
    threshold: float = field(
        default=0.4,
        metadata={"help": "Default threshold for multi-label classification (if not tuning)."}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bf16."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for data sampling."}
    )
    tune_params: bool = field(
        default=False,
        metadata={"help": "Whether to tune k and threshold on the validation set."}
    )
    use_linear_probe: bool = field(
        default=False,
        metadata={"help": "Whether to also evaluate using a Linear Probe (Logistic Regression)."}
    )
    use_svm: bool = field(
        default=False,
        metadata={"help": "Whether to also evaluate using SVM (LinearSVC)."}
    )
    cache_embeddings: bool = field(
        default=True,
        metadata={"help": "Whether to cache embeddings to disk to speed up re-runs."}
    )

def normalize_labels(example):
    """
    Ensure labels are always a list of integers.
    Handle both Multi-label (list) and Single-label (int) datasets.
    """
    # Check for 'labels' (standard) or 'label' (common in GLUE/scotus sometimes)
    if "labels" in example:
        lbl = example["labels"]
    elif "label" in example:
        lbl = example["label"]
    else:
        raise ValueError("Could not find 'labels' or 'label' field in dataset example.")
        
    if isinstance(lbl, int):
        return {"labels": [lbl]}
    return {"labels": lbl}

def encode_dataset(model, tokenizer, dataset, max_length, batch_size, device, cache_path=None):
    # Check cache
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    model.eval()
    all_embeddings = []
    
    # Process text in batches
    # ecthr_a 'text' is list of strings, join them
    # scotus 'text' might be single string? Handle both.
    texts = []
    for ex in dataset:
        if isinstance(ex["text"], list):
            texts.append(" ".join(ex["text"]))
        else:
            texts.append(ex["text"])
    
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
    
    embeddings = np.vstack(all_embeddings)
    
    # Save cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info(f"Saved embeddings to {cache_path}")
        
    return embeddings

def run_rag_eval(train_embeddings, train_labels, test_embeddings, test_labels, k, threshold, num_classes=10):
    """
    Run k-NN retrieval and evaluation.
    """
    num_test = test_embeddings.shape[0]
    
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
        if k >= train_embeddings.shape[0]:
            k_eff = train_embeddings.shape[0]
        else:
            k_eff = k
            
        top_k_indices = np.argpartition(sims, -k_eff, axis=1)[:, -k_eff:]
        
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
            
            # Normalize by total weight (Top-k sum)
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

def tune_rag_params(train_embeddings, train_labels, val_embeddings, val_labels, num_classes):
    """
    Grid search for best k and threshold using validation set.
    """
    k_values = [5, 10, 20, 50]
    thresholds = [0.2, 0.3, 0.4, 0.5]
    
    best_micro = -1
    best_k = 10
    best_thresh = 0.4
    
    logger.info("Tuning Hyperparameters on Validation Set...")
    
    # Pre-compute similarity matrix once for Validation
    # Caution: If Val and Train are HUGE, this might OOM. 
    # But for tuning we usually use full train? 
    # For speed, let's just loop.
    
    for k in k_values:
        for t in thresholds:
            micro, macro = run_rag_eval(train_embeddings, train_labels, val_embeddings, val_labels, k, t, num_classes)
            if micro > best_micro:
                best_micro = micro
                best_k = k
                best_thresh = t
                
    logger.info(f"Best Params Found: k={best_k}, threshold={best_thresh} (Val Micro-F1: {best_micro:.4f})")
    return best_k, best_thresh

def run_linear_probe(train_embeddings, train_labels, test_embeddings, test_labels, num_classes):
    """
    Train a Logistic Regression (OneVsRest) classifier.
    """
    logger.info("Training Linear Probe (Logistic Regression)...")
    
    # Convert labels to Multi-Hot
    mlb = MultiLabelBinarizer(classes=range(num_classes))
    y_train = mlb.fit_transform(train_labels)
    y_test = mlb.transform(test_labels)
    
    # Train
    # C=10.0 usually good for normalized embeddings (cosine-like)
    clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=10.0, max_iter=1000, random_state=42))
    clf.fit(train_embeddings, y_train)
    
    # Predict
    y_pred = clf.predict(test_embeddings)
    
    micro = f1_score(y_test, y_pred, average='micro')
    macro = f1_score(y_test, y_pred, average='macro')
    
    return micro, macro

def run_svm(train_embeddings, train_labels, test_embeddings, test_labels, num_classes):
    """
    Train a SVM (LinearSVC) classifier adapted for Multi-Label.
    LinearSVC typically works better for sparse/high-dim data than LogisticRegression.
    """
    logger.info("Running SVM (LinearSVC)...")
    
    mlb = MultiLabelBinarizer(classes=range(num_classes))
    y_train = mlb.fit_transform(train_labels)
    y_test = mlb.transform(test_labels)
    
    # Use OneVsRest with LinearSVC
    # class_weight='balanced' can help with small classes
    # CalibratedClassifierCV allows us to get probabilities if needed, but LinearSVC direct is faster
    svc = LinearSVC(C=1.0, dual=False, max_iter=2000, class_weight='balanced', random_state=42)
    clf = OneVsRestClassifier(svc)
    
    clf.fit(train_embeddings, y_train)
    y_pred = clf.predict(test_embeddings)
    
    micro = f1_score(y_test, y_pred, average='micro')
    macro = f1_score(y_test, y_pred, average='macro')
    
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
    
    # Normalize Label Format (List of ints)
    # Map all splits to standardized format
    for split in dataset.keys():
        dataset[split] = dataset[split].map(normalize_labels, load_from_cache_file=False)
    
    full_train_set = dataset["train"]
    validation_set = dataset["validation"]
    test_set = dataset["test"]
    
    # Determine num_classes
    # ecthr_a/b: 10 labels, eurlex: 100 labels (127 actually, but used 100 in many papers? No, LexGlue uses full label set usually)
    # scotus: 14 labels (single label)
    # Check max label index
    all_labels = [l for ex in full_train_set for l in ex["labels"]]
    num_classes = max(all_labels) + 1
    logger.info(f"Detected Number of Classes: {num_classes}")

    # Load Model
    logger.info(f"Loading Model: {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModel.from_pretrained(
        config.model_name_or_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        attn_implementation="flash_attention_2"
    ).to(device)
    
    # Define cache paths
    # Use model name safe string
    model_safe_name = config.model_name_or_path.replace("/", "_")
    ds_name = config.dataset_config_name
    cache_dir = "./embeddings_cache"
    
    test_cache = f"{cache_dir}/{model_safe_name}_{ds_name}_test.npy"
    val_cache = f"{cache_dir}/{model_safe_name}_{ds_name}_val.npy"
    train_cache = f"{cache_dir}/{model_safe_name}_{ds_name}_train.npy"
    
    # 1. Encode Test Set & Val Set (Once)
    logger.info("Encoding Test Set...")
    test_embeddings = encode_dataset(model, tokenizer, test_set, config.max_seq_length, config.batch_size, device, cache_path=test_cache if config.cache_embeddings else None)
    test_labels = [ex["labels"] for ex in test_set]

    if config.tune_params:
        logger.info("Encoding Validation Set...")
        val_embeddings = encode_dataset(model, tokenizer, validation_set, config.max_seq_length, config.batch_size, device, cache_path=val_cache if config.cache_embeddings else None)
        val_labels = [ex["labels"] for ex in validation_set]
    
    # 2. Parse Data Sizes
    raw_inputs = config.data_sizes.split(",")
    sizes = []
    for s in raw_inputs:
        val = float(s)
        if val <= 1.0 and val > 0:
            size = int(len(full_train_set) * val)
            if size == 0: size = 1
            sizes.append(size)
        else:
            sizes.append(int(val))
    sizes = [s for s in sizes if s <= len(full_train_set)]
    if len(full_train_set) not in sizes:
        sizes.append(len(full_train_set))
    sizes = sorted(list(set(sizes)))
    
    logger.info(f"Running experiments for sizes: {sizes}")
    
    results = []
    
    # 3. Encode Full Train Set
    logger.info("Encoding Full Training Set...")
    full_train_embeddings = encode_dataset(model, tokenizer, full_train_set, config.max_seq_length, config.batch_size, device, cache_path=train_cache if config.cache_embeddings else None)
    full_train_labels = [ex["labels"] for ex in full_train_set]
    
    # Random Permutation
    np.random.seed(config.seed)
    permuted_indices = np.random.permutation(len(full_train_set))
    
    for size in sizes:
        logger.info(f"--- Evaluating Size: {size} ---")
        
        subset_indices = permuted_indices[:size]
        train_emb_subset = full_train_embeddings[subset_indices]
        train_lbl_subset = [full_train_labels[i] for i in subset_indices]
        
        # --- Tune Params (Optional) ---
        current_k = config.k
        current_thresh = config.threshold
        
        if config.tune_params:
            best_k, best_t = tune_rag_params(train_emb_subset, train_lbl_subset, val_embeddings, val_labels, num_classes)
            current_k = best_k
            current_thresh = best_t
            
        # --- Run RAG ---
        micro_rag, macro_rag = run_rag_eval(
            train_emb_subset, 
            train_lbl_subset, 
            test_embeddings, 
            test_labels, 
            current_k, 
            current_thresh,
            num_classes
        )
        logger.info(f"[RAG] Size {size}: Micro={micro_rag:.4f}, Macro={macro_rag:.4f} (k={current_k}, t={current_thresh})")
        
        entry = {
            "train_size": size,
            "rag_micro_f1": micro_rag,
            "rag_macro_f1": macro_rag,
            "rag_best_k": current_k,
            "rag_best_t": current_thresh
        }
        
        # --- Run Linear Probe (Optional) ---
        if config.use_linear_probe:
            micro_lp, macro_lp = run_linear_probe(
                train_emb_subset,
                train_lbl_subset,
                test_embeddings,
                test_labels,
                num_classes
            )
            logger.info(f"[LinearProbe] Size {size}: Micro={micro_lp:.4f}, Macro={macro_lp:.4f}")
            entry["lp_micro_f1"] = micro_lp
            entry["lp_macro_f1"] = macro_lp
        
        # --- Run SVM (Optional) ---
        if config.use_svm:
            micro_svm, macro_svm = run_svm(
                train_emb_subset,
                train_lbl_subset,
                test_embeddings,
                test_labels,
                num_classes
            )
            logger.info(f"[SVM] Size {size}: Micro={micro_svm:.4f}, Macro={macro_svm:.4f}")
            entry["svm_micro_f1"] = micro_svm
            entry["svm_macro_f1"] = macro_svm
            
        results.append(entry)
        
    # 4. Save Results
    df = pd.DataFrame(results)
    save_path = os.path.join(config.output_dir, f"scaling_results_{config.dataset_config_name}.csv")
    df.to_csv(save_path, index=False)
    logger.info(f"Results saved to {save_path}")
    print(df)

if __name__ == "__main__":
    main()
