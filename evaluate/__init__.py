"""ablation.evaluate — Evaluation metrics, runners, and visualization."""

from .metrics import (
    evaluate_at_length,
    compute_attention_entropy,
    compute_attention_entropy_detailed,
    compute_attention_metrics_detailed,
)
from .heatmaps import generate_attention_heatmap
from .runner import run_full_evaluation

__all__ = [
    "evaluate_at_length",
    "compute_attention_entropy",
    "compute_attention_entropy_detailed",
    "compute_attention_metrics_detailed",
    "run_full_evaluation",
    "generate_attention_heatmap",
]

