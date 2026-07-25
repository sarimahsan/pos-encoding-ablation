# Positional Encoding Ablation Study (RoPE vs. ALiBi vs. NoPE)

A modular, config-driven PyTorch framework for conducting empirical positional encoding ablation experiments on decoder-only transformers (~51M parameters) using Google Colab T4 GPUs.

---

## 🏗️ Decoder-Only Transformer Architecture (~51M Parameters)

```mermaid
flowchart TD
    classDef input fill:#1e293b,stroke:#475569,stroke-width:2px,color:#f8fafc
    classDef norm fill:#334155,stroke:#64748b,stroke-width:2px,color:#f8fafc
    classDef attn fill:#312e81,stroke:#6366f1,stroke-width:2px,color:#f8fafc
    classDef dispatch fill:#581c87,stroke:#a855f7,stroke-width:2px,color:#f8fafc
    classDef mlp fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    classDef res fill:#1e1b4b,stroke:#818cf8,stroke-dasharray: 5 5,color:#f8fafc

    Input["Input Token IDs (B, T)"] :::input --> Embed["Token Embedding Layer\n(Vocab: 50,257 → d_model: 512)"] :::input
    Embed --> BlockStart

    subgraph TransformerBlock ["Transformer Block × 8 Layers (Pre-Norm Residual)"]
        direction TB
        BlockStart[Input x] --> Norm1["RMSNorm (d_model = 512)"] :::norm
        Norm1 --> QKV["Linear QKV Projection\n(3 × 512 = 1536)"] :::attn
        
        QKV --> Dispatch{"Positional Encoding Dispatch\n(config.pos_encoding)"} :::dispatch
        
        Dispatch -- "rope" --> RoPE["Rotary Position Embedding\nRotate Q & K pairs via cos/sin cache"] :::dispatch
        Dispatch -- "alibi" --> ALiBi["ALiBi Linear Bias\nAdd slope -|i-j| to attn scores (fp32)"] :::dispatch
        Dispatch -- "nope" --> NoPE["No Positional Encoding\nPass-through (Causal Mask only)"] :::dispatch
        
        RoPE --> AttnCore["Causal Scaled Dot-Product Attention\n(8 Heads, head_dim = 64)"] :::attn
        ALiBi --> AttnCore
        NoPE --> AttnCore
        
        AttnCore --> OutProj["Output Projection (512 → 512)"] :::attn
        OutProj --> Res1["[+] Residual Add: x = x + Attn(RMSNorm(x))"] :::res
        
        Res1 --> Norm2["RMSNorm (d_model = 512)"] :::norm
        Norm2 --> SwiGLU["SwiGLU Gated MLP\nW2( silu(W1 x) * W3 x )\n(d_model: 512 → d_ff: 1376 → d_model: 512)"] :::mlp
        SwiGLU --> Res2["[+] Residual Add: x = x + MLP(RMSNorm(x))"] :::res
    end

    Res2 --> FinalNorm["Final RMSNorm (d_model = 512)"] :::norm
    FinalNorm --> LMHead["Language Model Head\n(Tied Weights with Token Embedding)"] :::input
    LMHead --> Logits["Logits Output (B, T, Vocab: 50,257)"] :::input
```

---

## 📄 Interactive Research Guide & Dashboard
Open [`experiment_guide.html`](file:///e:/pos-encoding-ablation/experiment_guide.html) in your browser to view the interactive dashboard, pre-flight testing guide, full experiment matrix, and one-click Colab copy buttons.

---

## 🔬 Experiment Matrix (R1–R6) — Automatic 2-Seed Execution

Running any experiment automatically executes **2 seeds (Seed 42 & Seed 43)**, computes **Mean ± Std error bars**, and bundles everything into one zip file:

| Run ID | Positional Encoding | Train Seq Length | Eval Lengths (Extrapolation) | Automatic Seeds | CLI Command |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **R1** | RoPE | 256 | 256, 768, 1024 | Seed 42, Seed 43 | `python run_experiment.py --run R1` |
| **R2** | RoPE | 512 | 512, 768, 1024 | Seed 42, Seed 43 | `python run_experiment.py --run R2` |
| **R3** | ALiBi | 256 | 256, 768, 1024 | Seed 42, Seed 43 | `python run_experiment.py --run R3` |
| **R4** | ALiBi | 512 | 512, 768, 1024 | Seed 42, Seed 43 | `python run_experiment.py --run R4` |
| **R5** | NoPE | 256 | 256, 768, 1024 | Seed 42, Seed 43 | `python run_experiment.py --run R5` |
| **R6** | NoPE | 512 | 512, 768, 1024 | Seed 42, Seed 43 | `python run_experiment.py --run R6` |

---

## ⚡ Colab Execution Steps

```bash
# 1. Clone & install
!git clone https://github.com/<your-username>/pos-encoding-ablation.git
%cd pos-encoding-ablation
!pip install -r requirements.txt

# 2. Run 2-Minute Pre-Flight Verification Check
!python verify_colab.py

# 3. Run 2-seed experiment
!python run_experiment.py --run R1

# 4. Download combined zip (contains Seed 42 + Seed 43 + Aggregated Results)
from google.colab import files
files.download('outputs/R1_rope_seq256.zip')
```
