# Experimental Results Summary

This document summarizes the performance comparison between traditional Encoder models (Legal-BERT) and the proposed Decoder-based Embedding models (Qwen-0.6B), across different training regimes on the `ecthr_a` dataset.

## Performance Table (Test Set - Full Data)

| Category             | Model                     | Method                   | Micro-F1        | Macro-F1        | Notes                                       |
| :------------------- | :------------------------ | :----------------------- | :-------------- | :-------------- | :------------------------------------------ |
| **Baseline**   | Legal-BERT (Base)         | Zero-shot (CLS)          | 0.248           | 0.112           | Limited by 512 context window.              |
| **Baseline**   | Qwen-Embedding-0.6B       | Zero-shot (Last Token)   | 0.196           | 0.179           | Raw embedding without adaptation.           |
|                      |                           |                          |                 |                 |                                             |
| **Supervised** | Qwen-Embedding-0.6B       | LoRA Finetuned (Head)    | 0.563           | 0.413           | Standard classification head.               |
| **Supervised** | **Legal-BERT**      | **Full Finetuned** | **0.692** | **0.600** | Strongest standard baseline (Full 9k Data). |
| **Supervised** | Legal-BERT (Hierarchical) | Full Finetuned           | 0.756           | 0.678           | SOTA architecture for long docs.            |

## RAG Performance

| Category             | Model                         | Method                       | Micro-F1        | Macro-F1        | Notes                                                                                  |
| :------------------- | :---------------------------- | :--------------------------- | :-------------- | :-------------- | :------------------------------------------------------------------------------------- |
| **RAG (Ours)** | **Qwen-Embedding-0.6B** | **Training-Free k-NN** | **0.678** | **0.581** | **Best Efficiency / Performance Ratio.** `<br>` ($k=10, \tau=0.4$)           |
| **RAG (Ours)** | Qwen-Emb + Qwen-Rerank        | Rerank Top-50 -> Top-10      | 0.612           | 0.510           | Reranker (Generic) underperforms Embedding (Domain-Specific) on this specialized task. |

## Data Efficiency Experiment: RAG vs. Fine-Tuning

We investigated the "Cold Start" performance by comparing **Training-Free RAG** against **Full Fine-Tuning (Legal-BERT)** across different training set sizes.

| Training Size  | **RAG (Training-Free)** `<br>` Micro-F1 | **BERT (Fine-Tuned)** `<br>` Micro-F1 | **Relative Gain**   |
| :------------- | :---------------------------------------------- | :-------------------------------------------- | :------------------------ |
| **100**  | **0.473**                                 | 0.309                                         | **+53%** (RAG Wins) |
| **500**  | **0.575**                                 | 0.455                                         | **+26%** (RAG Wins) |
| **1000** | **0.592**                                 | 0.502                                         | **+18%** (RAG Wins) |
| **2000** | **0.613**                                 | 0.566                                         | **+8%** (RAG Wins)  |
| **4500** | **0.659**                                 | 0.640                                         | **+3%** (RAG Wins)  |
| **9000** | 0.675                                           | **0.682**                               | -1% (BERT Wins)           |

### Key Findings

1. **Cold Start Dominance**: RAG significantly outperforms fine-tuning in low-resource settings (< 1000 samples). With only **100 samples**, RAG achieves 0.47 F1, while BERT struggles at 0.31.
2. **Data Efficiency**: RAG with **500 samples** (0.575) performs comparably to BERT with **2000 samples** (0.566). This suggests RAG can reduce data annotation costs by **75%** for early-stage model deployment.
3. **Cross-Over Point**: Fine-tuning only surpasses the retrieval-based approach when the dataset size approaches **9,000 samples**. For any dataset smaller than this, RAG is the superior choice in terms of both performance and training cost (zero).

## Macro F1 Score Comparison (Different Data Sizes)

| Model                 | Task    | 100    | 500    | 1000   | 2000   | 4500   | 9000   |
| :-------------------- | :------ | :----- | :----- | :----- | :----- | :----- | :----- |
| BERT-H 2e-4           | ecthr_a | 0.0882 | 0.1923 | 0.1753 | 0.1885 | 0.2956 | 0.3325 |
|                       | ecthr_b | 0.1303 | 0.2370 | 0.2328 | 0.2185 | 0.3438 | 0.3727 |
| BERT-H 3e-5           | ecthr_a | 0.0620 | 0.0713 | 0.0831 | 0.2175 | 0.3216 | -      |
| BERT-H full-finetuned | ecthr_a | 0.2208 | 0.5239 | 0.5277 | 0.5596 | 0.6201 | 0.6780 |
|                       | ecthr_b | 0.4413 | -      | -      | -      | -      | -      |
| qwen-3 embedding 0.6B | ecthr_a | 0.2642 | 0.4245 | 0.4348 | 0.4691 | 0.5436 | 0.5902 |
| qwen-3 embedding 4B   | ecthr_a | 0.2729 | 0.4360 | 0.4565 | 0.5288 | 0.5708 | 0.6105 |
| qwen-3 embedding 8B   | ecthr_a | 0.3004 | 0.4573 | 0.4850 | 0.5407 | 0.5979 | 0.6324 |

## Micro F1 Score Comparison (Different Data Sizes)

| Model                 | Task    | 100    | 500    | 1000   | 2000   | 4500   | 9000   |
| :-------------------- | :------ | :----- | :----- | :----- | :----- | :----- | :----- |
| BERT-H 2e-4           | ecthr_a | 0.3045 | 0.3796 | 0.3877 | 0.3923 | 0.4986 | 0.5372 |
|                       | ecthr_b | 0.3452 | 0.4434 | 0.4441 | 0.4134 | 0.5365 | 0.5585 |
| BERT-H 3e-5           | ecthr_a | 0.2888 | 0.2851 | 0.3117 | 0.4064 | 0.5248 | -      |
| BERT-H full-finetuned | ecthr_a | 0.4177 | 0.6809 | 0.6804 | 0.7214 | 0.7342 | 0.7560 |
|                       | ecthr_b | 0.6339 | -      | -      | -      | -      | -      |
| qwen-3 embedding 0.6B | ecthr_a | 0.4729 | 0.5749 | 0.5920 | 0.6131 | 0.6586 | 0.6747 |
| qwen-3 embedding 4B   | ecthr_a | 0.4971 | 0.5926 | 0.6064 | 0.6486 | 0.6793 | 0.7044 |
| qwen-3 embedding 8B   | ecthr_a | 0.5089 | 0.6147 | 0.6395 | 0.6653 | 0.6992 | 0.7234 |

## [update] Macro F1 Score Comparison (Different Data Sizes)

| Model                 | Task    | 100    | 500    | 1000   | 2000   | 4500   | 9000   |
| :-------------------- | :------ | :----- | :----- | :----- | :----- | :----- | :----- |
| BERT-H 1e-3           | ecthr_a | 0.0462 | 0.0710 | 0.0843 |        | 0.0973 |        |
|                       | ecthr_b | 0.0910 | 0.0908 | 0.0858 | 0.0849 | 0.1083 | 0.1068 |
| BERT-H 2e-4           | ecthr_a | 0.0456 | 0.0804 | 0.0841 |        | 0.1993 | 0.2947 |
|                       | ecthr_b |        |        |        |        |        |        |
| BERT-H 5e-5           | ecthr_a | 0.0459 | 0.0731 | 0.0748 |        | 0.1651 | 0.2911 |
|                       | ecthr_b |        |        |        |        |        |        |
| BERT-H full-finetuned | ecthr_a | 0.0467 | 0.2245 | 0.3088 | 0.5468 | 0.5918 | 0.6562 |
|                       | ecthr_b | 0.0578 | 0.3256 | 0.4432 | 0.6506 | 0.7390 | 0.7838 |
| qwen-3 embedding 0.6B | ecthr_a | 0.2642 | 0.4245 | 0.4348 | 0.4691 | 0.5436 | 0.5902 |
|                       | ecthr_b | 0.3946 | 0.4517 | 0.5960 | 0.5893 | 0.6037 | 0.6775 |
| qwen-3 embedding 4B   | ecthr_a | 0.2729 | 0.4360 | 0.4565 | 0.5288 | 0.5708 | 0.6105 |
|                       | ecthr_b | 0.4362 | 0.5585 | 0.5103 | 0.5807 | 0.6405 | 0.6994 |
| qwen-3 embedding 8B   | ecthr_a | 0.3004 | 0.4573 | 0.4850 | 0.5407 | 0.5979 | 0.6324 |
|                       | ecthr_b |        |        |        |        |        |        |

## [update] Micro F1 Score Comparison (Different Data Sizes)

| Model                 | Task    | 100    | 500    | 1000   | 2000   | 4500   | 9000   |
| :-------------------- | :------ | :----- | :----- | :----- | :----- | :----- | :----- |
| BERT-H 1e-3           | ecthr_a | 0.2821 | 0.2878 | 0.3066 |        | 0.3006 |        |
|                       | ecthr_b | 0.3556 | 0.3482 | 0.3354 | 0.3355 | 0.3467 | 0.3481 |
| BERT-H 2e-4           | ecthr_a | 0.2771 | 0.2873 | 0.0304 |        | 0.4080 | 0.5093 |
|                       | ecthr_b |        |        |        |        |        |        |
| BERT-H 5e-5           | ecthr_a | 0.2765 | 0.2938 | 0.2946 |        | 0.3654 | 0.4952 |
|                       | ecthr_b |        |        |        |        |        |        |
| BERT-H full-finetuned | ecthr_a | 0.2787 | 0.4583 | 0.5511 | 0.7090 | 0.7493 | 0.7625 |
|                       | ecthr_b | 0.3198 | 0.5701 | 0.6696 | 0.7814 | 0.7911 | 0.8198 |
| qwen-3 embedding 0.6B | ecthr_a | 0.4829 | 0.5749 | 0.5920 | 0.6131 | 0.6586 | 0.6747 |
|                       | ecthr_b | 0.5653 | 0.6158 | 0.6490 | 0.6573 | 0.6837 | 0.7176 |
| qwen-3 embedding 4B   | ecthr_a | 0.5198 | 0.5926 | 0.6217 | 0.6486 | 0.6793 | 0.7044 |
|                       | ecthr_b | 0.6011 | 0.6512 | 0.6627 | 0.6957 | 0.7246 | 0.7463 |
| qwen-3 embedding 8B   | ecthr_a | 0.5479 | 0.6147 | 0.6492 | 0.6653 | 0.6992 | 0.7234 |
|                       | ecthr_b |        |        |        |        |        |        |

## Comparative Analysis: Qwen Embeddings vs. BERT-H Full-Finetuned

This section compares the performance of Qwen Embeddings (0.6B, 4B, 8B) against the strongest baseline, BERT-H full-finetuned, across different dataset sizes. The values represent the percentage difference relative to the BERT-H baseline ((Qwen - BERT)/BERT × 100%).

| Metric             | Model     | 100              | 500    | 1000   | 2000            | 4500   | 9000   |
| :----------------- | :-------- | :--------------- | :----- | :----- | :-------------- | :----- | :----- |
| **Macro F1** | Qwen 0.6B | **+19.7%** | -19.0% | -17.6% | -16.2%          | -12.3% | -12.9% |
|                    | Qwen 4B   | **+23.6%** | -16.8% | -13.5% | -5.5%           | -8.0%  | -10.0% |
|                    | Qwen 8B   | **+36.1%** | -12.7% | -8.1%  | **-3.4%** | -3.6%  | -6.7%  |
| **Micro F1** | Qwen 0.6B | **+13.2%** | -15.6% | -13.0% | -15.0%          | -10.3% | -10.8% |
|                    | Qwen 4B   | **+19.0%** | -13.0% | -10.9% | -10.1%          | -7.5%  | -6.8%  |
|                    | Qwen 8B   | **+21.8%** | -9.7%  | -6.0%  | **-4.8%** | -7.8%  | -4.3%  |

### Observations

1. **Low-Resource Dominance (100 samples)**: All Qwen models significantly outperform the fully finetuned BERT-H in the extremely low-resource setting (100 samples). The largest model (8B) shows a remarkable **+36.1%** improvement in Macro F1, indicating superior zero-shot/few-shot generalization capabilities inherent in larger language models embeddings compared to the smaller BERT encoder which struggles to adapt with so few examples.
2. **Performance Gap at Mid-Range (500-1000 samples)**: As the dataset size increases to 500 and 1000, BERT-H dramatically improves (likely stabilizing its gradients), surpassing the Qwen embeddings. The gap is most pronounced for the smaller 0.6B model (~19% deficit), but narrows significantly for the larger 4B and 8B models.
3. **Scaling Law Effects**: There is a clear trend where larger Qwen models consistently close the gap with BERT-H.

   * At **2000 samples**, the 8B model is only **3.4%** behind in Macro F1.
   * At **9000 samples**, while BERT-H remains superior, the Qwen 8B model maintains a competitive performance within **4-7%** of the SOTA hierarchical baseline, without requiring any task-specific fine-tuning (Training-Free).
4. **Efficiency vs. Performance Trade-off**: While BERT-H full-finetuning achieves higher peak performance with sufficient data (>500 samples), it requires training a complex hierarchical model. The Qwen embedding approach is training-free (using k-NN). For applications where training data is scarce (<500) or training resources are limited, Qwen embeddings (especially 8B) offer a compelling alternative.

## Computational Cost Analysis (FLOPs)

We compared the theoretical computational cost (Floating Point Operations) between **LoRA Fine-tuning**, **Full Fine-tuning**, and **RAG (Inference Only)**.

* **Setup**: Hierarchical BERT (116M Params), Sequence Length 8192, 5 Epochs (for FT).
* **Total Training Tokens**: 8.19 x 10^7 (assuming 2000 samples).

| Method                         | FLOPs (Estimated) | Relative Cost           | Memory (Optimizer States) | Notes                                        |
| :----------------------------- | :---------------- | :---------------------- | :------------------------ | :------------------------------------------- |
| **RAG (Inference Only)** | **1.9e15**  | **1x (Baseline)** | **0 MB**            | No gradients, no optimizer states.           |
| **LoRA Fine-tuning**     | **3.8e16**  | **~20x**          | ~50 MB                    | Saves memory but FLOPs only reduced by ~33%. |
| **Full Fine-tuning**     | **5.7e16**  | **~30x**          | ~928 MB                   | Most expensive in both Compute and Memory.   |

**Conclusion**: RAG is not only more data-efficient but also **20-30x more compute-efficient** than fine-tuning methods, as it skips the expensive backpropagation process entirely.

## Next Steps

* **LoRA-Adapted RAG**: Evaluate the k-NN performance using the Qwen embedding model fine-tuned via Contrastive Loss (in progress). This is expected to bridge the gap to the Hierarchical SOTA.
* **Linear Probe**: Evaluate Logistic Regression on top of frozen embeddings as a stronger baseline than k-NN.
