import logging
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List
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
    data_sizes: List[int] = field(
        default_factory=lambda: [100, 500, 1000, 2000, 4500, 9000],
        metadata={"help": "List of training set sizes to evaluate."}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Max sequence length for embedding."}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for encoding."}
    )
    k: int = field(
        default=10,
        metadata={"help": "Number of neighbors for k-NN."}
    )
    threshold: float = field(
        default=0.4,
        metadata={"help": "Threshold for classification voting."}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bf16."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for sampling."}
    )

def encode_dataset(model, tokenizer, dataset, max_length, batch_size, device):
    model.eval()
    all_embeddings = []
    
    # Process text
    # Handle ecthr_a list of strings
    texts = [" ".join(ex["text"]) if isinstance(ex["text"], list) else ex["text"] for ex in dataset]
    
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
            # Last token pooling (assuming padding on right)
            if tokenizer.padding_side == "left":
                embeddings = outputs.last_hidden_state[:, -1, :]
            else:
                sequence_lengths = (inputs.attention_mask.sum(dim=1) - 1)
                embeddings = outputs.last_hidden_state[torch.arange(outputs.last_hidden_state.shape[0]), sequence_lengths]
                
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.float().cpu().numpy())
            
    return np.vstack(all_embeddings)

def evaluate_knn(train_embeddings, train_labels, test_embeddings, test_labels, k=10, threshold=0.4):
    num_test = test_embeddings.shape[0]
    num_classes = 10 # for ecthr_a
    y_pred = []
    
    # Iterate test samples
    # Optimization: Process in chunks if test set is huge, but 1k is fine
    
    # Compute similarity matrix: (Num_Test, Num_Train)
    # Note: If Num_Train is small (100), this is tiny. If 9000, it's 1000x9000 matrix.
    sim_matrix = np.dot(test_embeddings, train_embeddings.T)
    
    for i in range(num_test):
        # Top-k
        top_k_indices = np.argsort(sim_matrix[i])[-k:][::-1]
        top_k_sims = sim_matrix[i][top_k_indices]
        
        votes = np.zeros(num_classes)
        total_weight = 0.0
        
        for rank, idx in enumerate(top_k_indices):
            # Weight by similarity
            weight = top_k_sims[rank]
            # Get labels of neighbor
            # train_labels is list of lists
            neighbor_labels = train_labels[idx]
            
            for label in neighbor_labels:
                votes[label] += weight
            total_weight += weight
            
        if total_weight > 0:
            probs = votes / total_weight
        else:
            probs = votes
            
        preds = (probs > threshold).astype(int)
        if preds.sum() == 0:
            preds[np.argmax(probs)] = 1
        y_pred.append(preds)
        
    # Convert test labels to multi-hot
    y_true = np.zeros((num_test, num_classes))
    for i, labels in enumerate(test_labels):
        for label in labels:
            y_true[i, label] = 1
            
    micro = f1_score(y_true, y_pred, average='micro')
    macro = f1_score(y_true, y_pred, average='macro')
    
    return micro, macro

def main():
    parser = HfArgumentParser((DataEffConfig,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        config = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
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
    dataset = load_dataset(config.dataset_name, config.dataset_config_name)
    full_train = dataset["train"]
    test_set = dataset["test"] # or validation
    
    # Encode Test Set ONCE (it doesn't change)
    logger.info("Encoding Test Set...")
    test_embeddings = encode_dataset(model, tokenizer, test_set, config.max_seq_length, config.batch_size, device)
    test_labels = test_set["labels"]
    
    results = []
    
    # 3. Data Efficiency Loop
    logger.info("Starting Data Efficiency Evaluation...")
    
    # Sort sizes just in case
    sizes = sorted([s for s in config.data_sizes if s <= len(full_train)])
    if sizes[-1] < len(full_train):
        sizes.append(len(full_train)) # Ensure full set is included
        
    for size in sizes:
        logger.info(f"--- Evaluating with N={size} training samples ---")
        
        # Slice Data
        # We use seed to ensure 'subset' is reproducible but 'random' selection logic
        # For data efficiency curve, usually we want to "add more data", so size=500 should include the size=100 data
        # So we just shuffle once and take top N
        shuffled_train = full_train.shuffle(seed=config.seed)
        subset = shuffled_train.select(range(size))
        
        # Encode Subset
        # (Could optimization: encode full train once and just slice array, but encoding is fast enough)
        subset_embeddings = encode_dataset(model, tokenizer, subset, config.max_seq_length, config.batch_size, device)
        subset_labels = subset["labels"]
        
        # Evaluate RAG
        micro, macro = evaluate_knn(subset_embeddings, subset_labels, test_embeddings, test_labels, k=config.k, threshold=config.threshold)
        
        logger.info(f"N={size} -> Micro-F1: {micro:.4f}, Macro-F1: {macro:.4f}")
        results.append({
            "size": size,
            "micro_f1": micro,
            "macro_f1": macro
        })
        
    # 4. Save and Plot
    json_path = os.path.join(config.output_dir, "data_efficiency_rag.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    # Plotting
    sizes = [r["size"] for r in results]
    micros = [r["micro_f1"] for r in results]
    macros = [r["macro_f1"] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, micros, marker='o', label='Micro-F1')
    plt.plot(sizes, macros, marker='s', label='Macro-F1')
    plt.title(f'Data Efficiency: RAG (Qwen-Embedding) on {config.dataset_config_name}')
    plt.xlabel('Number of Training Samples (Annotation Count)')
    plt.ylabel('F1 Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(config.output_dir, "learning_curve_rag.png"))
    logger.info(f"Results and plot saved to {config.output_dir}")

if __name__ == "__main__":
    main()

