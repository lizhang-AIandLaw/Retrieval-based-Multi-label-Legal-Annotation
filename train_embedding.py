import logging
import os
import sys
import json
import math
from dataclasses import dataclass, field
from typing import Optional, List, Union
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    get_scheduler,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
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
    output_dir: str = field(
        default="./output/qwen_embedding_finetuned",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "The initial learning rate for AdamW."}
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per GPU/CPU for training."}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Total number of training epochs to perform."}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    temperature: float = field(
        default=0.05,
        metadata={"help": "Temperature for InfoNCE loss."}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bf16."}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing to save memory."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."}
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the model to the Hub."}
    )
    hub_model_id: str = field(
        default=None,
        metadata={"help": "The name of the repository to keep in sync with the local datasets."}
    )
    hub_token: str = field(
        default=None,
        metadata={"help": "The token to use to push to the Model Hub."}
    )

class TripletDataset(Dataset):
    """
    Dataset that yields (Anchor, Positive) pairs.
    Negatives are handled in-batch (other samples in the batch are negatives).
    
    For Multi-label:
    - Two documents are 'Positive' if they share at least one label (or satisfy some Jaccard similarity).
    - To keep it efficient: For each document (Anchor), we randomly sample another document that shares at least one label.
    """
    def __init__(self, dataset, tokenizer, max_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Pre-compute label to indices map for fast positive sampling
        # ecthr_a labels are lists of integers
        logger.info("Building label index...")
        self.label_to_indices = {}
        for idx, ex in enumerate(dataset):
            for label in ex["labels"]:
                if label not in self.label_to_indices:
                    self.label_to_indices[label] = []
                self.label_to_indices[label].append(idx)
        
        # Filter out labels with only 1 example (cannot find positive)
        self.valid_indices = []
        for idx, ex in enumerate(dataset):
            has_pair = False
            for label in ex["labels"]:
                if len(self.label_to_indices[label]) > 1:
                    has_pair = True
                    break
            if has_pair:
                self.valid_indices.append(idx)
                
        logger.info(f"Found {len(self.valid_indices)} valid anchors out of {len(dataset)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 1. Select Anchor
        anchor_idx = self.valid_indices[idx]
        anchor_data = self.dataset[anchor_idx]
        
        # 2. Select Positive
        # Randomly pick one of its labels
        target_label = random.choice(anchor_data["labels"])
        
        # Pick a random index that also has this label, but is not the anchor itself
        candidates = self.label_to_indices[target_label]
        if len(candidates) == 1:
             # Should be caught by __init__ filter, but just in case
             pos_idx = anchor_idx 
        else:
            pos_idx = random.choice(candidates)
            while pos_idx == anchor_idx:
                pos_idx = random.choice(candidates)
                
        pos_data = self.dataset[pos_idx]
        
        return anchor_data["text"], pos_data["text"]

    def collate_fn(self, batch):
        # batch is list of (anchor_text, pos_text)
        anchors = [" ".join(item[0]) for item in batch] # ecthr_a text is list of str
        positives = [" ".join(item[1]) for item in batch]
        
        # Tokenize separately
        # We want to feed [Anchor_1, Pos_1, Anchor_2, Pos_2, ...] to model?
        # Or [Anchor_1, ..., Anchor_N] and [Pos_1, ..., Pos_N]?
        # Separate is easier for InfoNCE calculation
        
        anchor_enc = self.tokenizer(
            anchors,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        pos_enc = self.tokenizer(
            positives,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return anchor_enc, pos_enc

def train():
    parser = HfArgumentParser((TrainConfig,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        config = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        config = parser.parse_args_into_dataclasses()[0]
        
    set_seed(config.seed)
    
    # 1. Load Model & Tokenizer
    logger.info(f"Loading model: {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModel.from_pretrained(
        config.model_name_or_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32
    )
    
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # 2. Setup LoRA
    # Target modules for Qwen: usually c_attn or query_key_value depending on version.
    # For Qwen2/3 (HF version): q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    # Let's target all linear layers for best embedding quality
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, # or None
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 3. Load Data
    raw_dataset = load_dataset(config.dataset_name, config.dataset_config_name, split="train")
    # For debugging/speed
    # raw_dataset = raw_dataset.select(range(100))
    
    train_dataset = TripletDataset(raw_dataset, tokenizer, max_length=config.max_seq_length)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=train_dataset.collate_fn,
        drop_last=True
    )
    
    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    num_update_steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * max_train_steps),
        num_training_steps=max_train_steps,
    )
    
    # 5. Training Loop
    logger.info("Starting training...")
    model.train()
    
    global_step = 0
    total_loss = 0.0
    
    # InfoNCE Loss (Multiple Negatives Ranking Loss)
    # Batch: [A1, P1], [A2, P2] ...
    # Anchors: [A1, A2]
    # Candidates: [P1, P2]
    # A1 should match P1, and not P2 (in-batch negative)
    # Labels: diagonal
    
    cross_entropy = torch.nn.CrossEntropyLoss()
    
    for epoch in range(config.num_train_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, (anchor_enc, pos_enc) in enumerate(progress_bar):
            
            # Forward Anchors
            input_ids = anchor_enc["input_ids"].to(device)
            att_mask = anchor_enc["attention_mask"].to(device)
            
            out_a = model(input_ids=input_ids, attention_mask=att_mask)
            # Last token pooling for Qwen
            last_idx = att_mask.sum(1) - 1
            emb_a = out_a.last_hidden_state[torch.arange(input_ids.size(0)), last_idx]
            emb_a = F.normalize(emb_a, p=2, dim=1)
            
            # Forward Positives
            input_ids_p = pos_enc["input_ids"].to(device)
            att_mask_p = pos_enc["attention_mask"].to(device)
            
            out_p = model(input_ids=input_ids_p, attention_mask=att_mask_p)
            last_idx_p = att_mask_p.sum(1) - 1
            emb_p = out_p.last_hidden_state[torch.arange(input_ids_p.size(0)), last_idx_p]
            emb_p = F.normalize(emb_p, p=2, dim=1)
            
            # Compute Similarity Matrix (Batch x Batch)
            # scores[i][j] = similarity(A_i, P_j)
            scores = torch.matmul(emb_a, emb_p.T) / config.temperature
            
            # Target is diagonal (0, 1, 2...)
            labels = torch.arange(scores.size(0)).long().to(device)
            
            loss = cross_entropy(scores, labels)
            
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item()
            
            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.set_postfix({"loss": total_loss * config.gradient_accumulation_steps})
                total_loss = 0.0
                global_step += 1
                
    # 6. Save Model
    logger.info(f"Saving model to {config.output_dir}")
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
        
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # 7. Push to Hub
    if config.push_to_hub:
        logger.info("Pushing model to Hugging Face Hub...")
        if config.hub_model_id is None:
            # Default name if not provided: username/model_name
            repo_name = config.output_dir.split("/")[-1]
            logger.warning(f"No hub_model_id provided, using {repo_name}")
            config.hub_model_id = repo_name
            
        try:
            model.push_to_hub(config.hub_model_id, token=config.hub_token)
            tokenizer.push_to_hub(config.hub_model_id, token=config.hub_token)
            logger.info(f"Successfully pushed to {config.hub_model_id}")
        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")

    logger.info("Training complete.")

if __name__ == "__main__":
    train()

