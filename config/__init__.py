"""
ablation.config — Experiment and model configuration.
"""

from .model_config import TransformerConfig
from .experiments import get_experiment_config, EXPERIMENT_MATRIX

__all__ = ["TransformerConfig", "get_experiment_config", "EXPERIMENT_MATRIX"]
