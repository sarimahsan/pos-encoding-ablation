"""
Multi-Head Attention with configurable positional encoding dispatch.

Single dispatch point:
    rope  → modifies Q, K before attention
    alibi → adds bias to scores after QK^T
    nope  → pass-through (causal mask is the only order signal)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TransformerConfig
from .embeddings import RotaryEmbedding, ALiBiBias


class Attention(nn.Module):
    """
    Causal multi-head attention with configurable positional encoding.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        # Projections
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Positional encoding modules (only one active)
        if config.pos_encoding == "rope":
            self.rotary = RotaryEmbedding(config.head_dim, config.max_seq_len)
        elif config.pos_encoding == "alibi":
            self.alibi = ALiBiBias(config.n_heads, config.max_seq_len)
        # nope: nothing to instantiate

        # Causal mask — precompute up to max_seq_len
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple:
        B, T, _ = x.shape

        # QKV projection → split into heads
        qkv = self.qkv_proj(x)                                     # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)                             # each (B, T, D)

        # Reshape to (B, H, T, Dh)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # ── Positional encoding dispatch ─────────────────────────────
        if self.config.pos_encoding == "rope":
            q, k = self.rotary(q, k, T)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.config.pos_encoding == "alibi":
            scores = self.alibi(scores, T)

        # Causal mask — extend if needed
        if T > self.causal_mask.size(0):
            mask = torch.tril(torch.ones(T, T, device=x.device))
            self.causal_mask = mask
        scores = scores.masked_fill(
            self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0) == 0,
            float("-inf"),
        )

        attn = F.softmax(scores, dim=-1)

        # Combine heads
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out), attn
