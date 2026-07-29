"""
TransformerConfig — single dataclass driving model, data, and training.

Every hyperparameter from the spec sheet lives here. Swap pos_encoding /
activation / norm without touching any other file.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TransformerConfig:
    """
    Complete configuration for model architecture, training, and data.

    Default values match the ~51M-param spec:
        8 layers, 512 hidden, 8 heads, SwiGLU MLP (d_ff=1376),
        GPT-2 BPE vocab, RMSNorm pre-norm, tied embeddings.
    """

    # ── Model architecture ───────────────────────────────────────────
    vocab_size: int = 50257            # GPT-2 BPE
    n_layers: int = 8                  # transformer blocks
    d_model: int = 512                 # hidden size
    n_heads: int = 8                   # attention heads
    head_dim: int = 64                 # d_model / n_heads
    d_ff: int = 1376                   # SwiGLU intermediate (8/3*512 → mult of 64)
    max_seq_len: int = 1024            # precompute RoPE / ALiBi up to this
    norm: str = "rmsnorm"              # pre-norm type
    activation: str = "swiglu"         # gated activation
    pos_encoding: str = "rope"         # one of: "rope", "alibi", "nope"
    dropout: float = 0.0              # unnecessary at this scale
    tie_embeddings: bool = True        # tie input embed & output lm_head
    init_std: float = 0.02            # normal init std

    # ── Training ─────────────────────────────────────────────────────
    train_seq_len: int = 256           # training context length
    eval_seq_lens: List[int] = field(default_factory=lambda: [256, 768, 1024])
    optimizer: str = "adamw"
    lr: float = 3e-4                   # peak learning rate
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    warmup_steps: int = 500
    train_steps: int = 6_000           # 6k steps = 1.68 epochs over WikiText-103 (~2.3 hours on T4)
    batch_size: int = 128              # sequences per step
    grad_accum_steps: int = 4          # micro-batch 32 (peak VRAM ~3.2GB on Tesla T4)
    precision: str = "fp16"            # T4 = Turing, no native bf16
    compile: bool = False              # disable PyTorch Inductor extra buffer overhead on T4
    grad_clip: float = 1.0            # global norm
    seed: int = 42

    # ── Logging & checkpoints ────────────────────────────────────────
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = "outputs"
    run_name: str = "experiment"
    use_wandb: bool = False

    # ── Data ─────────────────────────────────────────────────────────
    dataset: str = "wikitext-103"
    tokenizer_name: str = "gpt2"

    def __post_init__(self):
        assert self.pos_encoding in ("rope", "alibi", "nope"), \
            f"pos_encoding must be rope/alibi/nope, got {self.pos_encoding}"
        assert self.head_dim == self.d_model // self.n_heads, \
            f"head_dim ({self.head_dim}) != d_model/n_heads ({self.d_model // self.n_heads})"

    def to_dict(self) -> dict:
        """Serialize to dict (for JSON saving)."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def summary(self) -> str:
        """Pretty-print the experiment configuration."""
        lines = [
            "=" * 60,
            f"  Experiment: {self.run_name}",
            "=" * 60,
            f"  Pos Encoding : {self.pos_encoding.upper()}",
            f"  Train SeqLen : {self.train_seq_len}",
            f"  Eval SeqLens : {self.eval_seq_lens}",
            f"  Model        : {self.n_layers}L / {self.d_model}D / {self.n_heads}H",
            f"  MLP dim      : {self.d_ff} (SwiGLU)",
            f"  Vocab        : {self.vocab_size}",
            f"  Batch size   : {self.batch_size} seqs",
            f"  Train steps  : {self.train_steps}",
            f"  Peak LR      : {self.lr}",
            f"  Precision    : {self.precision}",
            f"  Output dir   : {self.output_dir}",
            "=" * 60,
        ]
        return "\n".join(lines)
