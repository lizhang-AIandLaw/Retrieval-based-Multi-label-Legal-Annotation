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
    # trust_remote_code is deprecated for load_dataset in recent versions and datasets are usually safe/parquet now.
    # Removing it to fix "ERROR:datasets.load:trust_remote_code is not supported anymore."
    try:
        raw_datasets = load_dataset(
            model_config.dataset_name,
            model_config.dataset_config_name,
            trust_remote_code=model_config.trust_remote_code
        )
    except Exception:
         # Fallback if trust_remote_code is rejected (new datasets lib)
         raw_datasets = load_dataset(
            model_config.dataset_name,
            model_config.dataset_config_name
        )

    # Labels
    num_labels = model_config.num_labels
    
    # Load pretrained model and tokenizer
    # If model_name_or_path is a local directory, ensure we use absolute path or relative path correctly.
    # HuggingFace Hub usually expects repo_id (string) or local path.
    # The error "Repo id must be in the form..." suggests it treats "./output/..." as a repo id because it didn't find it locally?
    # Or maybe cached_file thinks it's a repo ID.
    # os.path.abspath might help avoid ambiguity.
    
    model_path = model_config.model_name_or_path
    if os.path.isdir(model_path) or model_path.startswith("./") or model_path.startswith("/"):
        model_path = os.path.abspath(model_path)
        
        # Check if path exists, if not, it might be a HF repo ID.
        if not os.path.exists(model_path):
             # Reset to original if local path doesn't exist (meaning it IS a repo ID or invalid path)
             # But if user passed "./output/...", they intend local.
             logger.warning(f"Local path {model_path} does not exist. Using original {model_config.model_name_or_path} as repo ID.")
             model_path = model_config.model_name_or_path
    
    # Important: When loading from local directory, 'trust_remote_code' might still be passed.
    # But for AutoTokenizer.from_pretrained with local directory, it should be fine.
    # However, if the directory doesn't look like a model directory (missing config.json or tokenizer.json), 
    # transformers might try to hit the hub.
    
    logger.info(f"Loading tokenizer from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=model_config.trust_remote_code
        )
    except OSError:
        # If loading from local failed, it might be because the directory is empty or invalid.
        # In eval mode, this is critical.
        if training_args.do_predict and not training_args.do_train:
             logger.error(f"Could not load tokenizer from {model_path}. Ensure the model was trained and saved correctly.")
             raise
        # If training, maybe we want to fallback to the base model in config if local output dir is empty?
        # But here model_path IS the one we want to use.
        raise
    
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
        
        # Resolve path for local directories
        base_model_name = model_config.model_name_or_path
        if os.path.isdir(base_model_name) or base_model_name.startswith("./"):
            base_model_name = os.path.abspath(base_model_name)

        # Explicitly set Flash Attention if specified in config
        model_kwargs = {}
        if hasattr(model_config, "attn_implementation"):
             # Not all models support this kwarg in from_pretrained, but AutoModel does for recent versions.
             # HierarchicalClassifier will pass it to base_model.
             pass 
             
        model = HierarchicalClassifier(
            base_model_name=base_model_name,
            num_labels=num_labels,
            trust_remote_code=model_config.trust_remote_code,
            # Pass hierarchical params from config
            num_layers=2, # default in class, but could be in config
            nhead=8,
            dim_feedforward=2048,
            attn_implementation=getattr(model_config, "attn_implementation", None)
        )
        
        # Apply LoRA to the base model inside HierarchicalClassifier if enabled
        if model_config.use_lora:
            from peft import get_peft_model, LoraConfig, TaskType
            logger.info(f"Applying LoRA to Hierarchical Base Model with r={model_config.lora_r}, alpha={model_config.lora_alpha}")
            
            # Note: For hierarchical, we are not using TaskType.SEQ_CLS because base model output is used as embedding features
            # We treat it as FEATURE_EXTRACTION technically, but PEFT TaskType usually implies the head.
            # Here we just want to LoRA the base_model.
            # We can apply LoRA to model.base_model
            
            peft_config = LoraConfig(
                inference_mode=False,
                r=model_config.lora_r,
                lora_alpha=model_config.lora_alpha,
                lora_dropout=model_config.lora_dropout,
                target_modules=model_config.lora_target_modules
            )
            model.base_model = get_peft_model(model.base_model, peft_config)
            
            # We need to make sure the rest of the model (transformer_encoder, classifier) is trainable.
            # get_peft_model sets requires_grad=False for non-LoRA params of the wrapped module.
            # But our transformer_encoder and classifier are outside base_model, so they remain trainable by default.
            
            # Print trainable params for verification
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in model.parameters())
            logger.info(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}")
        
        # If loading from a locally saved hierarchical model, load state dict manually
        if os.path.isdir(base_model_name):
            weights_path = os.path.join(base_model_name, "pytorch_model.bin")
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
        
        # Resolve path
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
        
        # Prepare kwargs for Flash Attention
        model_kwargs = {
            "config": config,
            "trust_remote_code": model_config.trust_remote_code,
            "ignore_mismatched_sizes": model_config.ignore_mismatched_sizes
        }
        
        # Manually check for Flash Attention config from JSON parser results (model_config)
        # ModelConfig dataclass doesn't have attn_implementation field by default, 
        # but we might have added it or we can check training_args?
        # Actually, HfArgumentParser parses config file into dataclass fields.
        # If 'attn_implementation' is in json but not in ModelConfig class, it might be lost or put in unused args?
        # Let's add it to ModelConfig or check if we can retrieve it.
        # Better: Add 'attn_implementation' to ModelConfig dataclass.
        
        if hasattr(model_config, "attn_implementation") and model_config.attn_implementation:
             model_kwargs["attn_implementation"] = model_config.attn_implementation
             logger.info(f"Using attention implementation: {model_config.attn_implementation}")
        
        if hasattr(model_config, "torch_dtype"):
             # This field might not exist in ModelConfig yet.
             pass

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        # Apply LoRA if enabled
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
    if training_args.do_train:
        max_train_samples = len(train_dataset)
        
        # Calculate samples from ratio if provided
        if model_config.max_train_ratio is not None:
            max_train_samples = int(len(train_dataset) * model_config.max_train_ratio)
            # Ensure at least 1 sample if ratio > 0
            if max_train_samples == 0 and model_config.max_train_ratio > 0:
                max_train_samples = 1
            logger.info(f"Using max_train_ratio={model_config.max_train_ratio} -> {max_train_samples} samples")
            
        # Or use absolute number if provided and no ratio
        elif model_config.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), model_config.max_train_samples)
            
        # Apply truncation if needed
        if max_train_samples < len(train_dataset):
            train_dataset = train_dataset.select(range(max_train_samples))
            logger.info(f"*** Training on a SUBSET of {max_train_samples} samples (Total available: {len(processed_datasets['train'])}) ***")

    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]

    # For multi-label classification with BCEWithLogitsLoss, labels must be float
    # However, some versions of transformers/torch might default 'labels' to long if not careful.
    # Let's inspect the model problem type and ensure compatibility.
    
    # Data collator
    if model_config.hierarchical:
        # Default collator works for tensors if they are already uniform shape
        # Our preprocess makes them uniform (max_segments, max_segment_length)
        data_collator = DefaultDataCollator()
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
        
    # Ensure labels are floats for BCEWithLogitsLoss in standard collator?
    # DataCollatorWithPadding usually just pads. The type comes from dataset features.
    # When we used `map`, new columns are inferred.
    
    # Explicitly set features of the processed dataset to ensure labels are float32?
    # Or we can cast in compute_loss if needed, but Trainer handles it if problem_type is set.
    # The error "RuntimeError: result type Float can't be cast to the desired output type Long"
    # usually happens in BCEWithLogitsLoss when target (labels) is Long but input (logits) is Float, 
    # OR if the loss calculation expects something else.
    # Wait, BCEWithLogitsLoss expects BOTH to be Float.
    # If target is Long, it fails.
    
    # Let's force the labels column in dataset to be float32.
    try:
        from datasets import Features, Sequence, Value
        # We can't easily cast entire dataset features after map without full re-process or cast_column (which might not exist in all versions).
        # But `preprocess_function` returns lists of floats: label_vec = [0.0] * num_labels.
        # Python floats are double precision. Torch expects float32 usually.
        pass
    except ImportError:
        pass
    
    # We can subclass DataCollatorWithPadding to force float labels
    class DataCollatorWithPaddingAndFloatLabels(DataCollatorWithPadding):
        def __call__(self, features):
            batch = super().__call__(features)
            if "labels" in batch:
                batch["labels"] = batch["labels"].float()
            return batch

    if not model_config.hierarchical:
         data_collator = DataCollatorWithPaddingAndFloatLabels(tokenizer=tokenizer, padding="longest")
    
    # Explicitly check for 'problem_type' in config and force it if possible
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
        
        # Explicitly print Micro and Macro F1 for user visibility
        print("\n" + "="*30)
        print(f"Test Results for {model_config.model_name_or_path}")
        print(f"F1 Micro: {metrics.get('test_f1_micro', 'N/A')}")
        print(f"F1 Macro: {metrics.get('test_f1_macro', 'N/A')}")
        print("="*30 + "\n")

if __name__ == "__main__":
    import transformers
    main()
