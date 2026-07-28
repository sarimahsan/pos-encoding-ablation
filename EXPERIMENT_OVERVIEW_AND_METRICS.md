# 🔬 Positional Encoding Ablation Study — Overview & Data Collection Guide

> **Document Purpose**: This guide provides a complete, self-contained overview of the Positional Encoding Ablation Study (**RoPE vs. ALiBi vs. NoPE**). It details what each experimental run evaluates, what metrics and artifacts are collected, and highlights key areas where additional logging/metrics could be added based on your recommendations.

---

## 📌 1. Executive Summary & Study Objective

The goal of this research project is to empirically analyze the impact of **Positional Encodings** on decoder-only Transformer language models (~51M parameters) trained on **WikiText-103**.

Specifically, we evaluate:
1. **In-Domain Performance**: Loss and perplexity at the trained context length ($L_{train} \in \{256, 512\}$).
2. **Length Extrapolation Capability**: Zero-shot performance when testing at lengths up to $4\times$ the training context length ($L_{eval} \in \{768, 1024\}$).
3. **Attention Dynamics**: Spatial attention entropy and layer/head attention pattern behavior.
4. **Execution Setup**: Single-seed run (**Seed 42**) executed for **12,000 training steps (12k steps)** per experiment.

---

## 🏗️ 2. Fixed Model Architecture & Training Setup

All 6 experimental runs use an **identical 51.0M parameter decoder-only Transformer** to ensure strict ceteris paribus comparisons. Only the positional encoding component and training context length vary.

### 📐 Model Architecture (~51M Parameters)
| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | `51,037,184` (~51.0M) | Vocabulary tied with LM Head |
| **Number of Layers ($N$)** | `8` | Transformer decoder blocks |
| **Hidden Size ($d_{model}$)** | `512` | Token representation dimension |
| **Attention Heads ($H$)** | `8` | Head dimension $d_{k} = 64$ |
| **MLP Dimension ($d_{ff}$)** | `1376` | SwiGLU Gated MLP ($\approx \frac{8}{3} d_{model}$) |
| **Normalization** | `RMSNorm` | Pre-normalization architecture |
| **Activation** | `SwiGLU` | Gated linear unit activation |
| **Vocabulary Size** | `50,257` | GPT-2 BPE Tokenizer |
| **Weight Tying** | `True` | Shared Token Embedding & LM Head |

### ⚡ Training Hyperparameters
| Hyperparameter | Value | Notes |
| :--- | :--- | :--- |
| **Dataset** | WikiText-103 | Standard benchmark dataset |
| **Optimizer** | AdamW | $\beta_1=0.9, \beta_2=0.95, \text{weight\_decay}=0.1$ |
| **Peak Learning Rate** | `3e-4` | Cosine annealing schedule |
| **Warmup Steps** | `500` steps | Linear warmup |
| **Total Training Steps** | `12,000` steps (12k) | ~3.07M tokens for seq 256 / ~6.14M tokens for seq 512 |
| **Grad Clipping** | `1.0` | Global norm clipping |
| **Precision** | `fp16` | PyTorch Mixed Precision (`autocast` + `GradScaler`) |
| **Hardware Target** | Tesla T4 GPU | Google Colab standard environment |

---

## 🧪 3. Experiment Matrix (Runs R1 through R6)

The matrix consists of **6 distinct experimental configurations** (3 Positional Encodings $\times$ 2 Training Context Lengths):

| Run ID | Positional Encoding | Train Seq Len ($L_{train}$) | Batch Size | Grad Accum | Eval Context Lengths ($L_{eval}$) | Seed & Steps |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R1** | **RoPE** (Rotary) | **256** | 128 | 4 | 256 *(train)*, 768 *(extrap)*, 1024 *(extrap)* | Seed 42 (12k steps) |
| **R2** | **RoPE** (Rotary) | **512** | 64 | 4 | 512 *(train)*, 768 *(extrap)*, 1024 *(extrap)* | Seed 42 (12k steps) |
| **R3** | **ALiBi** (Linear Bias) | **256** | 128 | 4 | 256 *(train)*, 768 *(extrap)*, 1024 *(extrap)* | Seed 42 (12k steps) |
| **R4** | **ALiBi** (Linear Bias) | **512** | 64 | 4 | 512 *(train)*, 768 *(extrap)*, 1024 *(extrap)* | Seed 42 (12k steps) |
| **R5** | **NoPE** (No Pos Encoding) | **256** | 128 | 4 | 256 *(train)*, 768 *(extrap)*, 1024 *(extrap)* | Seed 42 (12k steps) |
| **R6** | **NoPE** (No Pos Encoding) | **512** | 64 | 4 | 512 *(train)*, 768 *(extrap)*, 1024 *(extrap)* | Seed 42 (12k steps) |

> [!NOTE]
> Batch sizes are adjusted (128 for seq 256, 64 for seq 512) to ensure a consistent throughput token budget per step across sequence length variations.

---

## 📊 4. What We Currently Collect From Each Run

Data collected from every run is saved into organized subdirectories per run and seed.

### 1️⃣ Training Step Telemetry (Logged every 100 steps)
- **`train_loss`**: Cross-entropy training loss over micro-batches.
- **`perplexity`**: Training perplexity ($\exp(\min(\text{loss}, 20))$).
- **`lr`**: Scheduled learning rate at current step.
- **`steps_per_sec`**: Real-time throughput (steps completed per second).
- *Storage Formats*: `logs/train_log.jsonl`, `logs/train_log.csv`, and optional **Weights & Biases (W&B)** integration.

### 2️⃣ Evaluation & Length Extrapolation Metrics (Logged every 1,000 steps & final step 12,000)
- **Train-Length Loss & Perplexity**: Evaluated at $L_{train}$ (256 or 512).
- **Extrapolation Loss & Perplexity**: Evaluated zero-shot at extended context lengths (768 and 1024 tokens).
- *Storage Formats*:
  - `metrics/val_metrics_train_length.json`
  - `metrics/extrapolation_results.json`
  - `metrics/full_eval_results.json`
  - `metrics/final_summary.json`

### 3️⃣ Attention Dynamics & Geometry Analysis
- **Attention Entropy (`attention_entropy_mean`)**: Measures sharpness vs. diffusion of attention weights across all heads/layers ($-\sum p \log p$).
- **Attention Sink Ratio (`attention_sink_ratio_mean`)**: Percentage of total attention weight assigned to token 0 (first token, key $j=0$). Evaluated per layer/head ($8 \times 8$ matrix) across all context lengths (256/512, 768, 1024).
- **Effective Attention Distance (`effective_distance_mean`)**: Expected relative token distance attended to by queries ($\sum_{i,j} (i - j) \cdot A_{i,j}$). Measures whether heads focus locally vs. reach far globally.
- **Diagonal Mass Ratio (`diagonal_mass_ratio_mean`, $K=16$)**: Fraction of total attention mass concentrated within a causal diagonal band of distance $|i - j| \le 16$.
- **Full Geometry Storage (`metrics/attention_geometry.json`)**: Contains full $8 \times 8$ (Layer $\times$ Head) matrices and per-layer lists for all 4 metrics at every evaluation length.
- **Attention Heatmaps (`heatmaps/`)**: Rendered PNG heatmaps for Layer 0 (Input) and Layer 7 (Output) at Head 0 across sequence contexts (`attn_heatmap_L0_H0_seq64.png`, `attn_heatmap_L7_H0_seq64.png`).

### 4️⃣ System, Hardware & Execution Metadata
- **`metadata/run_metadata.json`**:
  - Execution runtime (`wall_clock_seconds`, `wall_clock_human`).
  - Target steps (12,000) vs. actual step counts & early stopping indicators.
  - GPU hardware profiling (`Tesla T4`, total memory 15.64 GB, CUDA capability `7.5`, CUDA version `12.8`).
  - Environment specification (Python, PyTorch, OS kernel version).
  - Exact model parameter count (`51,037,184`).
  - Runtime anomaly/exception log (captures interrupts, OOMs, or numerical issues).

### 5️⃣ Model Checkpoints & Reproducibility
- **Model Weights**: `checkpoints/run_<name>_best.pt`, `run_<name>_step<N>.pt`, `run_<name>_step12000_final.pt`.
  - State dicts include: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `config`, and `best_val_loss`.
- **Eval Sample Lock**: `eval_data/eval_batch_indices.json` stores exact validation sample indices to ensure evaluation batches are strictly identical across models.

---

## 📁 5. Directory & Artifact Hierarchy

```text
outputs/R1_rope_seq256/
├── config/
│   └── config.json            # Full run hyperparameter state
├── checkpoints/
│   ├── run_R1_rope_seq256_best.pt
│   └── run_R1_rope_seq256_step12000_final.pt
├── eval_data/
│   └── eval_batch_indices.json # Fixed validation batch indices
├── heatmaps/
│   ├── attn_heatmap_L0_H0_seq64.png
│   └── attn_heatmap_L7_H0_seq64.png
├── logs/
│   ├── train_log.csv
│   ├── train_log.jsonl
│   └── eval_results.jsonl
├── metadata/
│   └── run_metadata.json      # GPU, runtime, wall-clock (e.g. ~3.5h on T4), system details
└── metrics/
    ├── val_metrics_train_length.json
    ├── extrapolation_results.json
    ├── attention_entropy.json
    ├── attention_geometry.json # Complete 8x8 matrices for Entropy, Sink Ratio, Distance, & Diag Mass
    ├── full_eval_results.json
    └── final_summary.json
```


---

## 💡 6. What Else Could We Collect? (Feedback & Ideas Request)

To make this ablation study as thorough and insightful as possible, we would love your suggestions on additional metrics, probing tasks, or telemetry to log. 

Here are some potential areas we are considering—**which of these (or others) would you recommend adding?**

> [!TIP]
> ### Category A: Task-Based & Downstream Extrapolation
> - **Passkey / Needle-in-a-Haystack Retrieval**: Inserting a specific key/fact at various relative depth positions ($0\%, 25\%, 50\%, 75\%, 100\%$) across context lengths (up to 2048 tokens).
> - **Synthetic Copy / Repeat Tasks**: Testing if RoPE/ALiBi/NoPE can maintain exact token replication over long spans.
> - **Out-of-Domain Zero-Shot Evaluation**: Evaluating on Lambada, Penn Treebank, or OpenWebText to measure general language modeling transfer.

> [!TIP]
> ### Category B: Detailed Attention Mechanism Analysis
> - **Effective Attention Distance / Locality Score**: Expected token distance attended to by each head ($\sum_{j} (i - j) \cdot A_{i,j}$). Measures whether heads focus locally or globally.
> - **Attention Sink Ratio**: Percentage of total attention weight assigned to the initial token ($t_0$). (RoPE vs. NoPE often display distinct sink behaviors).
> - **Diagonal Mass Ratio**: Sum of attention weights within a band of width $K$ along the causal diagonal ($\frac{\sum_{|i-j| \le K} A_{i,j}}{\sum A_{i,j}}$).
> - **Cross-Head Attention Cosine Similarity**: Pairwise cosine similarity between attention matrices of different heads within the same layer to quantify head redundancy.

> [!TIP]
> ### Category C: Representation & Dynamics Probing
> - **Hidden State Norms per Layer**: Mean vector norm $\|h_l\|_2$ across Transformer layers to check for representation explosion/vanishing.
> - **Cosine Similarity Across Layers / Positions**: Representation collapse metrics (how similar are hidden states at step $i$ vs $i+k$?).
> - **Gradient Norms per Component**: Logging gradient norm specifically for Token Embeddings, Attention Projections (Q/K/V), and MLP layers to track optimization stability.
> - **Weight Norm Drift**: Track $\|W_t - W_0\|_2$ over training steps to measure weight movement.

> [!TIP]
> ### Category D: Computational Efficiency & Hardware Profiling
> - **Peak VRAM Memory Allocation (MB)**: Exact GPU memory peak during forward/backward passes at various sequence lengths.
> - **Inference Latency & MFU (Model FLOPs Utilization)**: Time per token generated (ms/token) during extrapolation inference.
> - **KV-Cache Footprint**: Memory overhead when storing Q/K/V caches during context extension.

---

## ❓ Questions for You

1. Which of the proposed additional metrics in **Section 6** do you think would yield the most critical insights for comparing RoPE, ALiBi, and NoPE?
2. Are there specific visual plots (e.g., Loss vs. Token Distance curves, Per-Layer Entropy over training time) that you would like to see included in the final benchmark paper/dashboard?
3. Should we extend extrapolation evaluation beyond 1024 tokens (e.g., 2048 or 4096 tokens)?

---
*Created for collaborative review in the Positional Encoding Ablation Study repository.*
