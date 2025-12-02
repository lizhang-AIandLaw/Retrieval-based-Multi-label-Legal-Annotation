import logging
import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances

# Try importing UMAP, fallback to TSNE if not available
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def get_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Use last token for Qwen/GPT or pooler for BERT
            # Assuming Qwen/GPT style as requested
            hidden_states = outputs.last_hidden_state
            # Find last non-padding token
            last_token_indices = attention_mask.sum(1) - 1
            last_token_indices = last_token_indices.clamp(min=0)
            embeddings = hidden_states[torch.arange(hidden_states.size(0)), last_token_indices]
            
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
            
            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())

    return np.vstack(all_embeddings), np.vstack(all_labels)

def analyze_clusters(embeddings, labels, output_dir, num_labels=10):
    """
    Analyze and visualize clusters.
    Since it's multi-label, we can't assign a single color to each point easily.
    Strategy:
    1. Visualize Top-K most frequent classes separately (Binary: Has Label vs No Label)
    2. Visualize dominant label (if we force single-label for visualization)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Dimensionality Reduction
    # Use PCA first to reduce to reasonable dim (e.g. 50) before t-SNE/UMAP to save time
    logger.info("Running PCA...")
    pca = PCA(n_components=min(50, embeddings.shape[1]))
    embeddings_pca = pca.fit_transform(embeddings)
    
    reducer_name = "UMAP" if HAS_UMAP else "t-SNE"
    logger.info(f"Running {reducer_name} for visualization...")
    
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings) # UMAP usually works better on raw high-dim than PCA-reduced, but PCA is safer for speed
    else:
        tsne = TSNE(n_components=2, random_state=42, metric='cosine', init='pca', learning_rate='auto')
        embeddings_2d = tsne.fit_transform(embeddings_pca)

    # 2. Quantitative Metrics
    # Intra-class vs Inter-class distances
    # For multi-label, we iterate over each class
    
    logger.info("Calculating distance metrics...")
    metrics = {}
    
    # Calculate full distance matrix (be careful with memory)
    # If N > 10k, this might be big (10k*10k * 4 bytes ~ 400MB, OK)
    dist_matrix = cosine_distances(embeddings)
    
    class_metrics = []
    
    for label_idx in range(num_labels):
        # Identify indices belonging to this class
        # labels shape: (N, num_labels) - multi-hot
        indices = np.where(labels[:, label_idx] == 1)[0]
        
        if len(indices) < 2:
            continue
            
        # Intra-class distance (Average distance between points in this class)
        # Extract sub-matrix
        intra_dists = dist_matrix[np.ix_(indices, indices)]
        # Upper triangle excluding diagonal
        intra_dists_triu = intra_dists[np.triu_indices(len(indices), k=1)]
        mean_intra_dist = np.mean(intra_dists_triu)
        
        # Inter-class distance (Average distance to points NOT in this class)
        non_indices = np.where(labels[:, label_idx] == 0)[0]
        if len(non_indices) > 0:
            inter_dists = dist_matrix[np.ix_(indices, non_indices)]
            mean_inter_dist = np.mean(inter_dists)
        else:
            mean_inter_dist = 0.0
            
        class_metrics.append({
            "label": int(label_idx),
            "count": int(len(indices)),
            "intra_dist": float(mean_intra_dist),
            "inter_dist": float(mean_inter_dist),
            "ratio": float(mean_intra_dist / (mean_inter_dist + 1e-9)) # Smaller is better
        })
        
        # Plot specific class distribution
        plt.figure(figsize=(10, 8))
        
        # Plot all points as gray background
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='lightgray', alpha=0.1, s=1, label='Other')
        
        # Plot class points
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], c='red', alpha=0.5, s=5, label=f'Label {label_idx}')
        
        plt.title(f'{reducer_name} Visualization for Label {label_idx} (Intra: {mean_intra_dist:.3f}, Inter: {mean_inter_dist:.3f})')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'cluster_label_{label_idx}.png'))
        plt.close()

    # Save metrics
    with open(os.path.join(output_dir, "clustering_metrics.json"), "w") as f:
        json.dump(class_metrics, f, indent=2)
        
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    
    # Print summary
    avg_intra = np.mean([m['intra_dist'] for m in class_metrics])
    avg_inter = np.mean([m['inter_dist'] for m in class_metrics])
    logger.info(f"Average Intra-class Distance: {avg_intra:.4f}")
    logger.info(f"Average Inter-class Distance: {avg_inter:.4f}")
    logger.info(f"Ratio (Intra/Inter): {avg_intra/avg_inter:.4f} (Lower is better)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="coastalcph/lex_glue")
    parser.add_argument("--dataset_config_name", type=str, default="ecthr_a")
    parser.add_argument("--max_seq_length", type=int, default=2048) # Smaller for analysis to be fast? Or same as RAG.
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./output/analysis")
    parser.add_argument("--num_labels", type=int, default=10)
    parser.add_argument("--bf16", action="store_true")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Dataset (Use Train set for analysis)
    dataset = load_dataset(args.dataset_name, args.dataset_config_name, split="train")
    
    # Sample for speed if dataset is huge (ecthr_a is ~9k, which is fine for full analysis)
    # dataset = dataset.select(range(1000)) 
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model_kwargs = {"trust_remote_code": True}
    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
        
    model = AutoModel.from_pretrained(args.model_name_or_path, **model_kwargs).to(device)
    
    def preprocess(examples):
        texts = [" ".join(t) for t in examples["text"]]
        batch = tokenizer(texts, padding="max_length", max_length=args.max_seq_length, truncation=True)
        
        # Multi-hot labels
        batch_labels = []
        for label_ids in examples["labels"]:
            vec = [0.0] * args.num_labels
            for lid in label_ids:
                if lid < args.num_labels:
                    vec[lid] = 1.0
            batch_labels.append(vec)
        batch["labels"] = batch_labels
        return batch
        
    dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    dataset.set_format("torch")
    
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=DataCollatorWithPadding(tokenizer))
    
    embeddings, labels = get_embeddings(model, loader, device)
    
    analyze_clusters(embeddings, labels, args.output_dir, args.num_labels)

if __name__ == "__main__":
    main()

