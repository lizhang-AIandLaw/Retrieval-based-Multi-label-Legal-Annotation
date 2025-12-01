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
    # Check if arguments are provided via command line flags (not just json file)
    # The error suggests sys.argv[1] is ending with .sh or not being parsed as json file correctly
    # if user mixes flags and json file, HfArgumentParser handles it if we pass everything.
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Pure JSON config file case
        model_config, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Command line arguments (potentially mixing with defaults or partials)
        # If a json file is passed as one of the args, we might need to handle it differently,
        # but usually standard usage is: python train.py config.json OR python train.py --arg value
        # The issue "train.py: error: the following arguments are required: --model_name_or_path"
        # means HfArgumentParser expects arguments that were missing when parsed from CLI.
        # But we are passing the config json file as the first argument in the shell script.
        # Let's double check how we parse.
        
        # If the first arg is a json file, we should use parse_json_file, but allow for overrides?
        # HfArgumentParser.parse_json_file returns (dataclass, dataclass).
        # If we want to support overrides (like --do_train false), we should use parse_args_into_dataclasses
        # but we need to tell it where to look for the config file if it's not a standard arg.
        
        # Improved logic:
        # If first arg ends in .json, load it.
        # Then if there are more args, we might need to override.
        # But parse_json_file doesn't support extra args easily.
        # Instead, we can use parse_args_into_dataclasses and pass the json file contents as defaults? No.
        
        # Common pattern:
        # python train.py config.json --arg1 value
        
        if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
            json_file = os.path.abspath(sys.argv[1])
            # Load json manually to update defaults or just use as base
            # However, HfArgumentParser doesn't mix json file path + CLI args natively in one call easily unless we implementation custom logic.
            # Let's try to read the json file and simulate args if we are in this mixed mode.
            
            # Actually, HuggingFace `TrainingArguments` doesn't natively support "config.json + overrides" in one line 
            # via `parse_json_file`. `parse_json_file` only reads the file.
            # `parse_args_into_dataclasses` reads from sys.argv.
            
            # Workaround: If we detect a json file as first arg, we read it, 
            # construct a list of arguments from it, append the rest of sys.argv[2:], 
            # and then call parse_args_into_dataclasses.
            
            import json
            with open(json_file, 'r') as f:
                config_dict = json.load(f)
            
            # Convert dict to CLI args
            # We need to be careful about boolean flags (e.g. --do_train) which might take no value or "True"/"False" depending on parser.
            # HfArgumentParser handles booleans well if we pass --do_train True or --no_do_train.
            
            synthetic_args = []
            for k, v in config_dict.items():
                if v is None:
                    continue
                if isinstance(v, bool):
                    # For boolean, HfArgumentParser (via argparse) usually uses store_true/false or explicit True/False
                    # TrainingArguments uses boolean_optional_action usually.
                    # Safer to use --param_name True/False string for our parsing if we just inject them.
                    # But standard argparse for bools: --do_train (implies true).
                    # Let's assume we construct explicit --key value.
                    synthetic_args.append(f"--{k}")
                    synthetic_args.append(str(v))
                elif isinstance(v, list):
                     # handle list arguments (e.g. report_to)
                     synthetic_args.append(f"--{k}")
                     for item in v:
                         synthetic_args.append(str(item))
                else:
                    synthetic_args.append(f"--{k}")
                    synthetic_args.append(str(v))
            
            # Add the rest of the real CLI args (overrides)
            # They will be appended at the end, so argparse should let them override earlier values.
            synthetic_args.extend(sys.argv[2:])
            
            model_config, training_args = parser.parse_args_into_dataclasses(args=synthetic_args)
            
        else:
            model_config, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load dataset
    # ecthr_a: input is a list of strings (text), output is a list of class indices.
    raw_datasets = load_dataset(
        model_config.dataset_name,
        model_config.dataset_config_name,
        trust_remote_code=model_config.trust_remote_code
    )

    # Labels
    num_labels = model_config.num_labels
    
    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_config.hierarchical:
        logger.info(f"Initializing Hierarchical Model with base: {model_config.model_name_or_path}")
        
        # Check if model_name_or_path is a local directory (trained model)
        # If so, we might need to be careful about what we pass as base_model_name.
        # If it's the output dir, it contains config.json (base model config) and pytorch_model.bin (hierarchical weights).
        # AutoModel.from_pretrained(dir) will try to load weights and fail/warn on mismatch.
        # We want to init the structure, then load our weights.
        
        model = HierarchicalClassifier(
            base_model_name=model_config.model_name_or_path,
            num_labels=num_labels,
            trust_remote_code=model_config.trust_remote_code,
            # Pass hierarchical params from config
            num_layers=2, # default in class, but could be in config
            nhead=8,
            dim_feedforward=2048
        )
        
        # If loading from a locally saved hierarchical model, load state dict manually
        if os.path.isdir(model_config.model_name_or_path):
            weights_path = os.path.join(model_config.model_name_or_path, "pytorch_model.bin")
            if os.path.exists(weights_path):
                logger.info(f"Loading hierarchical model weights from {weights_path}")
                state_dict = torch.load(weights_path, map_location="cpu")
                # We use strict=False because the base model inside might have loaded some keys (if matching) or not.
                # Actually, strict=True should work if the saved model matches the architecture perfectly.
                # The saved pytorch_model.bin contains keys for "base_model...", "transformer_encoder...", etc.
                # Our model has those attributes.
                load_result = model.load_state_dict(state_dict, strict=False)
                logger.info(f"Weights loaded: {load_result}")
    else:
        logger.info(f"Initializing Standard Sequence Classification Model: {model_config.model_name_or_path}")
        config = AutoConfig.from_pretrained(
            model_config.model_name_or_path,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            trust_remote_code=model_config.trust_remote_code,
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_config.model_name_or_path,
            config=config,
            trust_remote_code=model_config.trust_remote_code,
            ignore_mismatched_sizes=model_config.ignore_mismatched_sizes
        )

    # Ensure model pad token id is set if needed (mostly for GPT models)
    if hasattr(model, "config") and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Preprocessing the datasets
    def preprocess_function(examples):
        # Create multi-hot labels
        batch_labels = []
        for label_ids in examples["labels"]:
            label_vec = [0.0] * num_labels
            for lid in label_ids:
                if lid < num_labels: # Safety check
                    label_vec[lid] = 1.0
            batch_labels.append(label_vec)
        
        if model_config.hierarchical:
            # Hierarchical: Process list of strings
            # examples["text"] is a list of list of strings
            
            batch_input_ids = []
            batch_attention_mask = []
            
            for doc_paragraphs in examples["text"]:
                # Limit number of segments
                segments = doc_paragraphs[:model_config.max_segments]
                
                # Tokenize each segment
                # We need to manually pad to ensure (num_segments, seq_len) tensor shape
                tokenized_segments = tokenizer(
                    segments,
                    padding="max_length",
                    max_length=model_config.max_segment_length,
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Current shape: (actual_num_segments, seq_len)
                input_ids = tokenized_segments["input_ids"]
                attention_mask = tokenized_segments["attention_mask"]
                
                # Pad number of segments to max_segments
                current_segments = input_ids.size(0)
                pad_segments = model_config.max_segments - current_segments
                
                if pad_segments > 0:
                    # Create padding tensors
                    # Use pad_token_id for input_ids, 0 for attention_mask
                    input_padding = torch.full((pad_segments, model_config.max_segment_length), tokenizer.pad_token_id, dtype=torch.long)
                    mask_padding = torch.zeros((pad_segments, model_config.max_segment_length), dtype=torch.long)
                    
                    input_ids = torch.cat([input_ids, input_padding], dim=0)
                    attention_mask = torch.cat([attention_mask, mask_padding], dim=0)
                elif pad_segments < 0:
                    # Should be handled by [:model_config.max_segments] above but just in case
                    input_ids = input_ids[:model_config.max_segments]
                    attention_mask = attention_mask[:model_config.max_segments]
                
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
            
            # Stack to create batch tensors? 
            # No, map expects lists, but we are returning tensors inside lists.
            # The collator will stack them.
            
            return {
                "input_ids": batch_input_ids, 
                "attention_mask": batch_attention_mask, 
                "labels": batch_labels
            }
            
        else:
            # Standard: Join paragraphs
            texts = [" ".join(t) for t in examples["text"]]
            
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
            remove_columns=raw_datasets["train"].column_names # Remove original text columns to avoid collation issues
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]

    # Data collator
    if model_config.hierarchical:
        # Default collator works for tensors if they are already uniform shape
        # Our preprocess makes them uniform (max_segments, max_segment_length)
        data_collator = DefaultDataCollator()
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict on Test
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="test")
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
        
        # Save predictions
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        y_pred = np.zeros(probs.shape)
        y_pred[probs >= 0.5] = 1
        
        output_predictions_file = os.path.join(training_args.output_dir, "test_predictions.json")
        with open(output_predictions_file, "w") as writer:
            # Convert to list of lists of 1-based indices
            preds_list = []
            for row in y_pred:
                indices = np.where(row == 1)[0]
                # Convert to 1-based
                indices = indices + 1
                preds_list.append(indices.tolist())
            json.dump(preds_list, writer)
            
        logger.info(f"Predictions saved to {output_predictions_file}")

if __name__ == "__main__":
    import transformers
    main()
