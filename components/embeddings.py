"""
Positional encoding modules — RoPE and ALiBi.

RoPE: Rotary Position Embedding (Su et al., 2021)
    Rotates Q,K pairs using precomputed cos/sin frequencies.

ALiBi: Attention with Linear Biases (Press et al., 2022)
    Adds a linear distance-based bias to attention scores.
    Designed for length extrapolation.
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Precomputes cos/sin cache up to max_seq_len. Dynamically extends
    the cache if a longer sequence is encountered at eval time.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)                       # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)                     # (T, D)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple:
        """
        Apply rotary embedding to Q and K.

        Args:
            q, k: (B, H, T, Dh)
            seq_len: current sequence length

        Returns:
            (q_rotated, k_rotated) with same shapes
        """
        if seq_len > self.cos_cache.size(0):
            self._build_cache(seq_len)

        cos = self.cos_cache[:seq_len].to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)    # (1, 1, T, D)
        sin = self.sin_cache[:seq_len].to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class ALiBiBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi).

    Precomputes the bias matrix with geometric slopes:
        slope_h = 2^(-8 * h / n_heads)  for h in 1..n_heads

    Bias is computed in fp32 to avoid fp16 overflow at long sequences
    (critical for extrapolation to 1024 on T4).
    """

    def __init__(self, n_heads: int, max_seq_len: int = 1024):
        super().__init__()
        self.n_heads = n_heads

        # Geometric slopes — matches BLOOM's published values for n_heads=8
        slopes = torch.tensor(
            [2.0 ** (-(8.0 * i) / n_heads) for i in range(1, n_heads + 1)],
            dtype=torch.float32,
        )
        self.register_buffer("slopes", slopes, persistent=False)
        self._build_bias(max_seq_len)

    def _build_bias(self, seq_len: int):
        """Build relative distance bias matrix: bias[i,j] = -slope * |i - j|."""
        pos = torch.arange(seq_len, dtype=torch.float32)
        rel = -(pos.unsqueeze(1) - pos.unsqueeze(0)).abs()          # (T, T)
        bias = self.slopes.unsqueeze(1).unsqueeze(2) * rel.unsqueeze(0)  # (H, T, T)
        self.register_buffer("bias", bias, persistent=False)

    def forward(self, attn_scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Add ALiBi bias to attention scores.

        Args:
            attn_scores: (B, H, T, T)
            seq_len: current sequence length

        Returns:
            attn_scores + bias, cast back to input dtype
        """
        if seq_len > self.bias.size(1):
            self._build_bias(seq_len)
        return attn_scores + self.bias[:, :seq_len, :seq_len].to(attn_scores.dtype)
