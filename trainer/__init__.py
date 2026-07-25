"""ablation.trainer — Training loop and LR scheduling."""

from .trainer import Trainer
from .scheduler import build_cosine_scheduler

__all__ = ["Trainer", "build_cosine_scheduler"]
