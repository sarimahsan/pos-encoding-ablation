"""
DecoderTransformer — full GPT-style decoder-only transformer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TransformerConfig
from .normalization import RMSNorm
from .block import TransformerBlock


class DecoderTransformer(nn.Module):
    """
    GPT-style decoder-only transformer (~51M params).
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model)

        if config.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights(config)

    def _init_weights(self, config: TransformerConfig):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=config.init_std)
                if "out_proj" in name or "w2" in name:
                    with torch.no_grad():
                        p.mul_(1.0 / math.sqrt(2.0 * config.n_layers))

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor = None, return_attn: bool = False
    ) -> dict:
        B, T = input_ids.shape
        x = self.token_embed(input_ids)

        attn_weights_all = [] if return_attn else None
        for block in self.blocks:
            x, attn_w = block(x)
            if return_attn:
                attn_weights_all.append(attn_w)

        x = self.final_norm(x)

        if self.config.tie_embeddings:
            logits = F.linear(x, self.token_embed.weight)
        else:
            logits = self.lm_head(x)

        result = {"logits": logits}
        if return_attn:
            result["attn_weights"] = attn_weights_all

        if targets is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
            )
            result["loss"] = loss

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
