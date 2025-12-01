# Multi-Label Legal Classification Project

This project implements multi-label classification on the `ecthr_a` subset of the `coastalcph/lex_glue` dataset using Hugging Face Transformers.

It compares two main approaches:
1. **Standard Sequence Classification**: Truncating documents to fit model context.
2. **Hierarchical Classification**: Splitting documents into segments and aggregating representations ("Divide and Conquer").

Models used:
- `nlpaueb/legal-bert-base-uncased`
- `Qwen/Qwen3-Embedding-0.6B`

## Project Structure

- `configs/`: Configuration files for training.
- `scripts/`: Shell scripts to launch training and evaluation.
- `models/`: Custom model definitions (e.g., HierarchicalClassifier).
- `train.py`: Main training and evaluation script.
- `inference.py`: Script for inference on new text.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Git ignore rules (excluding models and envs).

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Login to services (Required for tracking and model upload):
   ```bash
   # Login to Hugging Face (requires Write token)
   huggingface-cli login
   
   # Login to Weights & Biases
   wandb login
   ```

## Training

Training scripts are configured to:
1. Use **Weights & Biases** (`wandb`) for experiment tracking.
2. Automatically **upload the trained model** to your Hugging Face Hub account upon completion.

### Standard Models

To train and evaluate the standard BERT model (max 512 tokens):
```bash
./scripts/run_bert.sh
```

To train and evaluate the standard Qwen model (max 2048 tokens):
```bash
./scripts/run_qwen.sh
```

### Hierarchical Variants

The hierarchical approach implements a "Divide and Conquer" strategy specifically designed for long legal documents:
1.  **Segmentation**: The document is split into multiple segments (paragraphs).
2.  **Encoding (Layer 1)**: The base model (Legal-BERT or Qwen) independently encodes each segment into a vector.
    - *Legal-BERT*: Uses the `[CLS]` token.
    - *Qwen*: Uses the last valid token embedding.
3.  **Aggregation (Layer 2)**: A Transformer Encoder layer processes the sequence of segment vectors to capture document-level context.
4.  **Classification**: The aggregated representation is used for final multi-label prediction.

To train Hierarchical Legal-BERT:
```bash
./scripts/run_hierarchical_bert.sh
```

To train Hierarchical Qwen-Embedding:
```bash
./scripts/run_hierarchical_qwen.sh
```

## Evaluation Only

To evaluate both Baseline (untrained) and Fine-tuned models on the test split:

```bash
./scripts/eval_all.sh
```

This script sequentially runs evaluation for all 8 configurations (Standard/Hierarchical x BERT/Qwen x Baseline/Fine-tuned).

## Inference

To use a trained model for prediction on a text string:

```bash
python inference.py ./output/bert_ecthr_a "Your legal text here..."
```

## Technical Implementation of Multi-Label Output

The project handles multi-label classification (0-9 classes) through the following pipeline:

1.  **Label Encoding**:
    -   Raw integer labels (e.g., `[2, 5]`) are converted into **Multi-hot Vectors** (e.g., `[0, 0, 1, 0, 0, 1, 0...]`) during preprocessing.
    -   This creates a target vector of length 10 (num_labels) with 1.0s at active indices.

2.  **Model Architecture**:
    -   **Legal-BERT**: Adds a linear classification head on top of the `[CLS]` token embedding.
    -   **Qwen-Embedding**: Adds a linear classification head on top of the last token's hidden state.
    -   Both project the latent representation to 10 logits.

3.  **Loss Function**:
    -   Uses **`BCEWithLogitsLoss`** (Binary Cross Entropy with Logits).
    -   This treats the problem as 10 independent binary classification tasks.
    -   It applies a Sigmoid activation internally to each logit before computing the loss against the multi-hot targets.

4.  **Inference Thresholding**:
    -   Output logits are passed through a **Sigmoid** function to get probabilities (0-1).
    -   A threshold of **0.5** is applied: any class with probability ≥ 0.5 is predicted as present (output "1").

## Notes on Input Strategies

### Standard Input Window
- **BERT**: Inputs are truncated to 512 tokens.
- **Qwen**: Inputs are truncated to 2048 tokens (configurable up to ~32k, but limited by VRAM).

### Hierarchical Input
- **Hierarchical BERT**: Documents are split into up to 32 segments of 128 tokens each.
- **Hierarchical Qwen**: Documents are split into up to 16 segments of 512 tokens each.
- This architecture allows the models to process documents far longer than their native context windows.

## Models Details

- **Legal-BERT**: A specialized BERT model pre-trained on legal text, used here as the segment encoder in the hierarchical setup.
- **Qwen-Embedding**: A decoder-based embedding model. In the hierarchical setup, it functions as a powerful feature extractor for each segment, proving its versatility as a "first layer" encoder despite its decoder architecture.
