"""
Cosine LR scheduler with linear warmup.
"""

import math
import torch

from config import TransformerConfig


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer, config: TransformerConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(
            1, config.train_steps - config.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
