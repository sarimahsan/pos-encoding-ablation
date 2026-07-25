"""
TransformerBlock — single pre-norm residual block.

    x = x + Attention(RMSNorm(x))
    x = x + SwiGLU_MLP(RMSNorm(x))
"""

import torch
import torch.nn as nn

from config import TransformerConfig
from .normalization import RMSNorm
from .attention import Attention
from .mlp import SwiGLU_MLP


class TransformerBlock(nn.Module):
    """
    Pre-norm residual transformer block.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLU_MLP(config)

    def forward(self, x: torch.Tensor) -> tuple:
        attn_out, attn_weights = self.attn(self.attn_norm(x))
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, attn_weights
