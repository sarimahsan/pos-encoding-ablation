"""
SwiGLU MLP — Gated MLP with SiLU (Swish) activation.

    out = W2( silu(W1 x) * W3 x )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TransformerConfig


class SwiGLU_MLP(nn.Module):
    """
    Gated MLP with SiLU (Swish) activation.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
