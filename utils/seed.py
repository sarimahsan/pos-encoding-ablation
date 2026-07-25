"""Reproducibility — set all random seeds."""

import random
import torch
import numpy as np


def set_seed(seed: int):
    """
    Set all random seeds for full reproducibility.

    Sets: Python random, NumPy, PyTorch CPU & CUDA, cuDNN deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
