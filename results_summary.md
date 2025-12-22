# Experimental Results Summary

This document summarizes the performance comparison between traditional Encoder models (Legal-BERT) and the proposed Decoder-based Embedding models (Qwen-0.6B), across different training regimes on the `ecthr_a` dataset.

## Performance Table (Test Set - Full Data)

| Category | Model | Method | Micro-F1 | Macro-F1 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Legal-BERT (Base) | Zero-shot (CLS) | 0.248 | 0.112 | Limited by 512 context window. |
| **Baseline** | Qwen-Embedding-0.6B | Zero-shot (Last Token) | 0.196 | 0.179 | Raw embedding without adaptation. |
| | | | | | |
| **Supervised** | Qwen-Embedding-0.6B | LoRA Finetuned (Head) | 0.563 | 0.413 | Standard classification head. |
| **Supervised** | **Legal-BERT** | **Full Finetuned** | **0.692** | **0.600** | Strongest standard baseline (Full 9k Data). |
| **Supervised** | Legal-BERT (Hierarchical) | Full Finetuned | 0.756 | 0.678 | SOTA architecture for long docs. |
| | | | | | |
| **RAG (Ours)** | **Qwen-Embedding-0.6B** | **Training-Free k-NN** | **0.678** | **0.581** | **Best Efficiency / Performance Ratio.** <br> ($k=10, \tau=0.4$) |
| **RAG (Ours)** | Qwen-Emb + Qwen-Rerank | Rerank Top-50 -> Top-10 | 0.612 | 0.510 | Reranker (Generic) underperforms Embedding (Domain-Specific) on this specialized task. |

## Data Efficiency Experiment: RAG vs. Fine-Tuning

We investigated the "Cold Start" performance by comparing **Training-Free RAG** against **Full Fine-Tuning (Legal-BERT)** across different training set sizes.

| Training Size | **RAG (Training-Free)** <br> Micro-F1 | **BERT (Fine-Tuned)** <br> Micro-F1 | **Relative Gain** |
| :--- | :--- | :--- | :--- |
| **100** | **0.473** | 0.309 | **+53%** (RAG Wins) |
| **500** | **0.575** | 0.455 | **+26%** (RAG Wins) |
| **1000** | **0.592** | 0.502 | **+18%** (RAG Wins) |
| **2000** | **0.613** | 0.566 | **+8%** (RAG Wins) |
| **4500** | **0.659** | 0.640 | **+3%** (RAG Wins) |
| **9000** | 0.675 | **0.682** | -1% (BERT Wins) |

### Key Findings

1.  **Cold Start Dominance**: RAG significantly outperforms fine-tuning in low-resource settings (< 1000 samples). With only **100 samples**, RAG achieves 0.47 F1, while BERT struggles at 0.31.
2.  **Data Efficiency**: RAG with **500 samples** (0.575) performs comparably to BERT with **2000 samples** (0.566). This suggests RAG can reduce data annotation costs by **75%** for early-stage model deployment.
3.  **Cross-Over Point**: Fine-tuning only surpasses the retrieval-based approach when the dataset size approaches **9,000 samples**. For any dataset smaller than this, RAG is the superior choice in terms of both performance and training cost (zero).

## Next Steps

*   **LoRA-Adapted RAG**: Evaluate the k-NN performance using the Qwen embedding model fine-tuned via Contrastive Loss (in progress). This is expected to bridge the gap to the Hierarchical SOTA.
