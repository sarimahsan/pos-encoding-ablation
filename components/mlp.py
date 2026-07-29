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
    Gated MLP with SiLU (Swish) activation using fused w13 linear projection.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.w13 = nn.Linear(config.d_model, 2 * config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x, w3_x = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(w1_x) * w3_x)
