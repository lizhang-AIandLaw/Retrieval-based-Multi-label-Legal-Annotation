# Experimental Results Summary

This document summarizes the performance comparison between traditional Encoder models (Legal-BERT) and the proposed Decoder-based Embedding models (Qwen-0.6B), across different training regimes on the `ecthr_a` dataset.

## Performance Table (Test Set)

| Category | Model | Method | Micro-F1 | Macro-F1 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Legal-BERT (Base) | Zero-shot (CLS) | 0.248 | 0.112 | Limited by 512 context window. |
| **Baseline** | Qwen-Embedding-0.6B | Zero-shot (Last Token) | 0.196 | 0.179 | Raw embedding without adaptation. |
| | | | | | |
| **Supervised** | Qwen-Embedding-0.6B | LoRA Finetuned (Head) | 0.563 | 0.413 | Standard classification head. |
| **Supervised** | **Legal-BERT** | **Full Finetuned** | **0.692** | **0.600** | Strongest standard baseline. |
| **Supervised** | Legal-BERT (Hierarchical) | Full Finetuned | 0.756 | 0.678 | SOTA architecture for long docs. |
| | | | | | |
| **RAG (Ours)** | **Qwen-Embedding-0.6B** | **Training-Free k-NN** | **0.678** | **0.581** | **Best Efficiency / Performance Ratio.** <br> ($k=10, \tau=0.4$) |

## Key Findings

1.  **RAG Efficacy:** Our **Training-Free RAG** approach (0.678 Micro-F1) significantly outperforms the standard Zero-shot baselines and even the LoRA-finetuned classification head (0.563).
2.  **Competitiveness:** It nearly matches the performance of the fully fine-tuned **Legal-BERT** (0.692), despite requiring **no gradient updates** or training compute.
3.  **Parameter Sensitivity:** Optimizing the k-NN threshold (from 0.5 to 0.4) yielded a ~4% gain in Macro-F1, highlighting the importance of calibration in retrieval-based classification.

## Next Steps

*   **LoRA-Adapted RAG:** Evaluate the k-NN performance using the Qwen embedding model fine-tuned via Contrastive Loss (in progress). This is expected to bridge the gap to the Hierarchical SOTA.

