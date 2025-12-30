import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorWithPadding,
    DefaultDataCollator
)

# Import our custom hierarchical model
from models.hierarchical import HierarchicalClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    dataset_name: str = field(
        default="coastalcph/lex_glue",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: str = field(
        default="ecthr_a",
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    num_labels: int = field(
        default=10,
        metadata={"help": "Number of labels for classification."}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading model."}
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Whether to ignore mismatched sizes when loading model."}
    )
    # New arguments for hierarchical model
    hierarchical: bool = field(
        default=False,
        metadata={"help": "Whether to use hierarchical model architecture."}
    )
    max_segments: int = field(
        default=64,
        metadata={"help": "Maximum number of segments (paragraphs) per document for hierarchical model."}
    )
    max_segment_length: int = field(
        default=128,
        metadata={"help": "Maximum length of each segment for hierarchical model."}
    )
    attn_implementation: str = field(
        default=None,
        metadata={"help": "Attention implementation to use (e.g., 'flash_attention_2', 'eager', 'sdpa')."}
    )
    
    # LoRA arguments
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for training."}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA r value."}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha value."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout value."}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"],
        metadata={"help": "List of module names to target for LoRA."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_train_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "For data scaling experiments: truncate the number of training examples to this ratio "
                "(e.g., 0.1 for 10% of data). Overrides max_train_samples if set."
            )
        },
    )

def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[probs >= threshold] = 1
    
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    
    metrics = {
        'f1_micro': f1_micro_average,
        'f1_macro': f1_macro_average,
        'accuracy': accuracy
    }
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids
    )
    return result

def main():
    parser = HfArgumentParser((ModelConfig, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_config, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
            json_file = os.path.abspath(sys.argv[1])
            import json
            with open(json_file, 'r') as f:
                config_dict = json.load(f)
            
            synthetic_args = []
            for k, v in config_dict.items():
                if v is None:
                    continue
                if isinstance(v, bool):
                    synthetic_args.append(f"--{k}")
                    synthetic_args.append(str(v))
                elif isinstance(v, list):
                     synthetic_args.append(f"--{k}")
                     for item in v:
                         synthetic_args.append(str(item))
                else:
                    synthetic_args.append(f"--{k}")
                    synthetic_args.append(str(v))
            
            synthetic_args.extend(sys.argv[2:])
            model_config, training_args = parser.parse_args_into_dataclasses(args=synthetic_args)
        else:
            model_config, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)

    try:
        raw_datasets = load_dataset(
            model_config.dataset_name,
            model_config.dataset_config_name,
            trust_remote_code=model_config.trust_remote_code
        )
    except Exception:
         raw_datasets = load_dataset(
            model_config.dataset_name,
            model_config.dataset_config_name
        )

    num_labels = model_config.num_labels
    
    model_path = model_config.model_name_or_path
    if os.path.isdir(model_path) or model_path.startswith("./") or model_path.startswith("/"):
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
             logger.warning(f"Local path {model_path} does not exist. Using original {model_config.model_name_or_path} as repo ID.")
             model_path = model_config.model_name_or_path
    
    logger.info(f"Loading tokenizer from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=model_config.trust_remote_code
        )
    except OSError:
        if training_args.do_predict and not training_args.do_train:
             logger.error(f"Could not load tokenizer from {model_path}. Ensure the model was trained and saved correctly.")
             raise
        raise
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_config.hierarchical:
        logger.info(f"Initializing Hierarchical Model with base: {model_config.model_name_or_path}")
        base_model_name = model_config.model_name_or_path
        if os.path.isdir(base_model_name) or base_model_name.startswith("./"):
            base_model_name = os.path.abspath(base_model_name)

        model = HierarchicalClassifier(
            base_model_name=base_model_name,
            num_labels=num_labels,
            trust_remote_code=model_config.trust_remote_code,
            num_layers=2,
            nhead=8,
            dim_feedforward=2048,
            attn_implementation=getattr(model_config, "attn_implementation", None)
        )
        
        if model_config.use_lora:
            from peft import get_peft_model, LoraConfig
            logger.info(f"Applying LoRA to Hierarchical Base Model with r={model_config.lora_r}, alpha={model_config.lora_alpha}")
            
            peft_config = LoraConfig(
                inference_mode=False,
                r=model_config.lora_r,
                lora_alpha=model_config.lora_alpha,
                lora_dropout=model_config.lora_dropout,
                target_modules=model_config.lora_target_modules
            )
            model.base_model = get_peft_model(model.base_model, peft_config)
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in model.parameters())
            logger.info(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}")
        
        if os.path.isdir(base_model_name):
            weights_path = os.path.join(base_model_name, "pytorch_model.bin")
            if os.path.exists(weights_path):
                logger.info(f"Loading hierarchical model weights from {weights_path}")
                state_dict = torch.load(weights_path, map_location="cpu")
                load_result = model.load_state_dict(state_dict, strict=False)
                logger.info(f"Weights loaded: {load_result}")
    else:
        logger.info(f"Initializing Standard Sequence Classification Model: {model_config.model_name_or_path}")
        model_path = model_config.model_name_or_path
        if os.path.isdir(model_path) or model_path.startswith("./") or model_path.startswith("/"):
            model_path = os.path.abspath(model_path)
            if not os.path.exists(model_path):
                 logger.warning(f"Local path {model_path} does not exist. Using original {model_config.model_name_or_path} as repo ID.")
                 model_path = model_config.model_name_or_path

        config = AutoConfig.from_pretrained(
            model_path,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            trust_remote_code=model_config.trust_remote_code,
        )
        
        model_kwargs = {
            "config": config,
            "trust_remote_code": model_config.trust_remote_code,
            "ignore_mismatched_sizes": model_config.ignore_mismatched_sizes
        }
        
        if hasattr(model_config, "attn_implementation") and model_config.attn_implementation:
             model_kwargs["attn_implementation"] = model_config.attn_implementation
             logger.info(f"Using attention implementation: {model_config.attn_implementation}")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        if model_config.use_lora:
            from peft import get_peft_model, LoraConfig, TaskType
            logger.info(f"Applying LoRA to model with r={model_config.lora_r}, alpha={model_config.lora_alpha}")
            
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=model_config.lora_r,
                lora_alpha=model_config.lora_alpha,
                lora_dropout=model_config.lora_dropout,
                target_modules=model_config.lora_target_modules
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    if hasattr(model, "config") and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    def preprocess_function(examples):
        # Determine label key (scotus/glue use 'label', others 'labels')
        label_key = "labels"
        if "label" in examples:
            label_key = "label"
        elif "labels" not in examples:
            pass
            
        batch_labels = []
        # Check if examples[label_key] exists to avoid KeyError
        if label_key in examples:
            for label_val in examples[label_key]:
                label_vec = [0.0] * num_labels
                
                if isinstance(label_val, int):
                    if label_val < num_labels:
                        label_vec[label_val] = 1.0
                elif isinstance(label_val, list):
                    for lid in label_val:
                        if lid < num_labels:
                            label_vec[lid] = 1.0
                
                batch_labels.append(label_vec)
        else:
            # Fallback for test set without labels? Or just dummy
            batch_labels = [[0.0] * num_labels for _ in range(len(examples["text"]))]
        
        if model_config.hierarchical:
            batch_input_ids = []
            batch_attention_mask = []
            
            for text_entry in examples["text"]:
                if isinstance(text_entry, str):
                    segments = [text_entry] 
                else:
                    segments = text_entry

                segments = segments[:model_config.max_segments]
                
                tokenized_segments = tokenizer(
                    segments,
                    padding="max_length",
                    max_length=model_config.max_segment_length,
                    truncation=True,
                    return_tensors="pt"
                )
                
                input_ids = tokenized_segments["input_ids"]
                attention_mask = tokenized_segments["attention_mask"]
                
                current_segments = input_ids.size(0)
                pad_segments = model_config.max_segments - current_segments
                
                if pad_segments > 0:
                    input_padding = torch.full((pad_segments, model_config.max_segment_length), tokenizer.pad_token_id, dtype=torch.long)
                    mask_padding = torch.zeros((pad_segments, model_config.max_segment_length), dtype=torch.long)
                    
                    input_ids = torch.cat([input_ids, input_padding], dim=0)
                    attention_mask = torch.cat([attention_mask, mask_padding], dim=0)
                elif pad_segments < 0:
                    input_ids = input_ids[:model_config.max_segments]
                    attention_mask = attention_mask[:model_config.max_segments]
                
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
            
            return {
                "input_ids": batch_input_ids, 
                "attention_mask": batch_attention_mask, 
                "labels": batch_labels
            }
            
        else:
            texts = []
            for t in examples["text"]:
                if isinstance(t, list):
                    texts.append(" ".join(t))
                else:
                    texts.append(t)
            
            result = tokenizer(
                texts,
                padding="max_length" if model_config.pad_to_max_length else False,
                max_length=model_config.max_seq_length,
                truncation=True,
            )
            result["labels"] = batch_labels
            return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
            remove_columns=raw_datasets["train"].column_names
        )

    train_dataset = processed_datasets["train"]
    if training_args.do_train:
        max_train_samples = len(train_dataset)
        
        if model_config.max_train_ratio is not None:
            max_train_samples = int(len(train_dataset) * model_config.max_train_ratio)
            if max_train_samples == 0 and model_config.max_train_ratio > 0:
                max_train_samples = 1
            logger.info(f"Using max_train_ratio={model_config.max_train_ratio} -> {max_train_samples} samples")
            
        elif model_config.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), model_config.max_train_samples)
            
        if max_train_samples < len(train_dataset):
            train_dataset = train_dataset.shuffle(seed=training_args.seed)
            train_dataset = train_dataset.select(range(max_train_samples))
            logger.info(f"*** Training on a SUBSET of {max_train_samples} samples (Total available: {len(processed_datasets['train'])}) ***")

    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]

    if model_config.hierarchical:
        data_collator = DefaultDataCollator()
    else:
        class DataCollatorWithPaddingAndFloatLabels(DataCollatorWithPadding):
            def __call__(self, features):
                batch = super().__call__(features)
                if "labels" in batch:
                    batch["labels"] = batch["labels"].float()
                return batch
        data_collator = DataCollatorWithPaddingAndFloatLabels(tokenizer=tokenizer, padding="longest")
    
    if hasattr(model, "config"):
        model.config.problem_type = "multi_label_classification"

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model() 
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        
        total_flos = train_result.metrics.get("train_flos", train_result.metrics.get("total_flos", 0))
        logger.info(f">>> TRAINING_COST_FLOPs: {total_flos}")

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="test")
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
        
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        y_pred = np.zeros(probs.shape)
        y_pred[probs >= 0.5] = 1
        
        output_predictions_file = os.path.join(training_args.output_dir, "test_predictions.json")
        with open(output_predictions_file, "w") as writer:
            preds_list = []
            for row in y_pred:
                indices = np.where(row == 1)[0]
                indices = indices + 1
                preds_list.append(indices.tolist())
            json.dump(preds_list, writer)
            
        logger.info(f"Predictions saved to {output_predictions_file}")
        
        print("\n" + "="*30)
        print(f"Test Results for {model_config.model_name_or_path}")
        print(f"F1 Micro: {metrics.get('test_f1_micro', 'N/A')}")
        print(f"F1 Macro: {metrics.get('test_f1_macro', 'N/A')}")
        print("="*30 + "\n")

if __name__ == "__main__":
    import transformers
    main()
