import logging
import os
import sys
import json
import torch
import numpy as np
from dataclasses import dataclass, field
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    HfArgumentParser, 
    AutoModelForCausalLM
)
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
class EvalConfig:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained embedding model (Retriever)"}
    )
    reranker_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained reranker model (CausalLM)"}
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
        metadata={"help": "The maximum total input sequence length for embedding."}
    )
    rerank_max_length: int = field(
        default=2048,
        metadata={"help": "The maximum total input sequence length for reranking."}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for encoding."}
    )
    rerank_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for reranking (usually smaller due to length)."}
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

# --- Qwen Reranker Logic (from Official Docs) ---

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a legal case description, retrieve relevant legal precedents.'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction, query=query, doc=doc)
    return output

def process_reranker_inputs(pairs, tokenizer, max_length, prefix_tokens, suffix_tokens, device):
    # pairs is list of (instruction, query, doc)
    
    formatted_texts = [format_instruction(inst, q, d) for inst, q, d in pairs]
    
    # Custom padding logic from official doc
    # We pre-calculate length to truncate properly
    
    inputs = tokenizer(
        formatted_texts, 
        padding=False, 
        truncation='longest_first',
        return_attention_mask=False, 
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        
    # Pad manually
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    
    for key in inputs:
        inputs[key] = inputs[key].to(device)
        
    return inputs

def compute_logits(model, inputs, token_true_id, token_false_id):
    with torch.no_grad():
        outputs = model(**inputs)
        batch_scores = outputs.logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().cpu().numpy() # Return probability of "yes"
    return scores

# --- Retriever Logic ---

def encode_dataset(model, tokenizer, dataset, max_length, batch_size, device, desc="Encoding"):
    model.eval()
    all_embeddings = []
    
    for i in tqdm(range(0, len(dataset), batch_size), desc=desc):
        batch = dataset[i : i + batch_size]
        texts = [" ".join(t) for t in batch["text"]]
        
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            if tokenizer.padding_side == "left":
                embeddings = outputs.last_hidden_state[:, -1, :]
            else:
                sequence_lengths = (inputs.attention_mask.sum(dim=1) - 1)
                embeddings = outputs.last_hidden_state[torch.arange(outputs.last_hidden_state.shape[0]), sequence_lengths]
                
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            # IMPORTANT: Convert bf16 to float32 for numpy compatibility
            all_embeddings.append(embeddings.float().cpu().numpy())
            
    return np.vstack(all_embeddings)

def main():
    parser = HfArgumentParser((EvalConfig,))
    config = parser.parse_args_into_dataclasses()[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. Load Data
    logger.info(f"Loading dataset: {config.dataset_name} / {config.dataset_config_name}")
    dataset = load_dataset(config.dataset_name, config.dataset_config_name)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    # 2. Setup Retriever
    logger.info(f"Loading Retriever: {config.model_name_or_path}")
    ret_tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    if ret_tokenizer.pad_token is None:
        ret_tokenizer.pad_token = ret_tokenizer.eos_token
    ret_model = AutoModel.from_pretrained(
        config.model_name_or_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32
    ).to(device)

    # 3. Setup Reranker (CausalLM)
    logger.info(f"Loading Reranker: {config.reranker_model_name_or_path}")
    rerank_tokenizer = AutoTokenizer.from_pretrained(config.reranker_model_name_or_path, trust_remote_code=True, padding_side='left')
    
    # Set pad_token for Reranker (Critical for batch processing)
    if rerank_tokenizer.pad_token is None:
        rerank_tokenizer.pad_token = rerank_tokenizer.eos_token
        
    rerank_model = AutoModelForCausalLM.from_pretrained(
        config.reranker_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        attn_implementation="flash_attention_2"
    ).to(device).eval()
    
    # Prepare Reranker Tokens
    token_false_id = rerank_tokenizer.convert_tokens_to_ids("no")
    token_true_id = rerank_tokenizer.convert_tokens_to_ids("yes")
    
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = rerank_tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = rerank_tokenizer.encode(suffix, add_special_tokens=False)
    
    instruction = "Given a legal case facts description, retrieve relevant legal precedents that share the same violation labels."

    # 4. Encode/Cache Training Set
    os.makedirs(config.cache_dir, exist_ok=True)
    train_cache_path = os.path.join(config.cache_dir, f"train_embeddings_{config.dataset_config_name}.npy")
    
    if os.path.exists(train_cache_path):
        logger.info(f"Loading cached training embeddings from {train_cache_path}")
        train_embeddings = np.load(train_cache_path)
    else:
        logger.info("Encoding training set...")
        train_embeddings = encode_dataset(ret_model, ret_tokenizer, train_dataset, config.max_seq_length, config.batch_size, device)
        np.save(train_cache_path, train_embeddings)
        
    # 5. Encode Test Set
    logger.info("Encoding test set...")
    test_embeddings = encode_dataset(ret_model, ret_tokenizer, test_dataset, config.max_seq_length, config.batch_size, device, desc="Encoding Queries")
    
    # 6. Evaluation Loop
    logger.info(f"Starting Evaluation: Retrieve Top-{config.retrieve_top_n} -> Rerank -> Top-{config.k}")
    
    num_classes = 10
    y_true = []
    y_pred = []
    
    for i in tqdm(range(len(test_dataset)), desc="Processing Queries"):
        query_vec = test_embeddings[i].reshape(1, -1)
        
        # Step 1: Retrieve
        sims = np.dot(train_embeddings, query_vec.T).flatten()
        top_n_indices = np.argsort(sims)[-config.retrieve_top_n:][::-1]
        
        # Step 2: Rerank
        query_text = " ".join(test_dataset[i]["text"])
        candidate_texts = [" ".join(train_dataset[int(idx)]["text"]) for idx in top_n_indices]
        
        # Prepare Batch for Reranker
        rerank_inputs_list = [(instruction, query_text, cand) for cand in candidate_texts]
        
        all_rerank_scores = []
        for j in range(0, len(rerank_inputs_list), config.rerank_batch_size):
            batch_pairs = rerank_inputs_list[j : j + config.rerank_batch_size]
            inputs = process_reranker_inputs(
                batch_pairs, 
                rerank_tokenizer, 
                config.rerank_max_length, 
                prefix_tokens, 
                suffix_tokens, 
                device
            )
            batch_scores = compute_logits(rerank_model, inputs, token_true_id, token_false_id)
            all_rerank_scores.extend(batch_scores)
            
        all_rerank_scores = np.array(all_rerank_scores)
        
        # Step 3: k-NN
        reranked_local_indices = np.argsort(all_rerank_scores)[-config.k:][::-1]
        final_top_k_indices = [top_n_indices[idx] for idx in reranked_local_indices]
        final_scores = all_rerank_scores[reranked_local_indices]
        
        class_votes = np.zeros(num_classes)
        total_weight = 0.0
        
        for rank, idx in enumerate(final_top_k_indices):
            weight = final_scores[rank] # Already a probability [0, 1]
            labels = train_dataset[int(idx)]["labels"]
            for label in labels:
                class_votes[label] += weight
            total_weight += weight
            
        if total_weight > 0:
            class_probs = class_votes / total_weight
        else:
            class_probs = class_votes
            
        predictions = (class_probs > config.threshold).astype(int)
        if predictions.sum() == 0:
            top_class = np.argmax(class_probs)
            predictions[top_class] = 1
            
        y_pred.append(predictions)
        
        true_labels = np.zeros(num_classes)
        for label in test_dataset[i]["labels"]:
            true_labels[label] = 1
        y_true.append(true_labels)
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    logger.info(f"Results for RAG + Rerank (k={config.k}, N={config.retrieve_top_n}):")
    logger.info(f"Micro-F1: {micro_f1:.4f}")
    logger.info(f"Macro-F1: {macro_f1:.4f}")
    
    output_file = os.path.join(config.cache_dir, "rerank_results_qwen3.json")
    with open(output_file, "w") as f:
        json.dump({
            "micro_f1": micro_f1,
            "macro_f1": macro_f1
        }, f, indent=4)

if __name__ == "__main__":
    main()

