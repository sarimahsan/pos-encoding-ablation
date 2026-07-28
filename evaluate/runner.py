"""
Full evaluation runner — evaluates a model across all configured seq lengths.
"""

import os
import json
import torch

from config import TransformerConfig
from .metrics import evaluate_at_length, compute_attention_metrics_detailed


@torch.no_grad()
def run_full_evaluation(
    model, dataloaders, config: TransformerConfig, device,
    use_amp: bool = True, save_dir: str = None,
) -> dict:
    if save_dir is None:
        save_dir = os.path.join(config.output_dir, "metrics")
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "run_name": config.run_name,
        "pos_encoding": config.pos_encoding,
        "train_seq_len": config.train_seq_len,
        "metrics": {},
    }

    geometry_summary = {
        "run_name": config.run_name,
        "pos_encoding": config.pos_encoding,
        "train_seq_len": config.train_seq_len,
        "eval_metrics": {},
    }

    print(f"\n{'='*60}")
    print(f"  Full Evaluation: {config.run_name}")
    print(f"{'='*60}")

    for eval_len in config.eval_seq_lens:
        key = f"val_{eval_len}"
        if key not in dataloaders:
            print(f"  Warning: No dataloader for seq_len={eval_len}, skipping.")
            continue

        print(f"\n  Evaluating at seq_len={eval_len}...")

        val_loss, val_ppl = evaluate_at_length(model, dataloaders[key], device, use_amp)
        attn_metrics = compute_attention_metrics_detailed(
            model, dataloaders[key], device, use_amp, max_batches=10, K=16
        )

        label = "train-length" if eval_len == config.train_seq_len else "extrapolation"
        results["metrics"][str(eval_len)] = {
            "type": label,
            "loss": round(val_loss, 4),
            "perplexity": round(val_ppl, 2),
            "attention_entropy_mean": attn_metrics["entropy"]["mean"],
            "attention_entropy_per_layer": attn_metrics["entropy"]["per_layer"],
            "attention_sink_ratio_mean": attn_metrics["sink_ratio"]["mean"],
            "attention_sink_ratio_per_layer": attn_metrics["sink_ratio"]["per_layer"],
            "effective_distance_mean": attn_metrics["effective_distance"]["mean"],
            "effective_distance_per_layer": attn_metrics["effective_distance"]["per_layer"],
            "diagonal_mass_ratio_mean": attn_metrics["diagonal_mass_ratio"]["mean"],
            "diagonal_mass_ratio_per_layer": attn_metrics["diagonal_mass_ratio"]["per_layer"],
            "attention_geometry": attn_metrics,
        }

        geometry_summary["eval_metrics"][str(eval_len)] = attn_metrics

        print(
            f"    [{label:>14s}] loss={val_loss:.4f}, ppl={val_ppl:.2f} | "
            f"entropy={attn_metrics['entropy']['mean']:.4f}, "
            f"sink_ratio={attn_metrics['sink_ratio']['mean']:.4f}, "
            f"eff_dist={attn_metrics['effective_distance']['mean']:.2f}, "
            f"diag_mass(K=16)={attn_metrics['diagonal_mass_ratio']['mean']:.4f}"
        )

    results_path = os.path.join(save_dir, "full_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    geom_path = os.path.join(save_dir, "attention_geometry.json")
    with open(geom_path, "w") as f:
        json.dump(geometry_summary, f, indent=2)

    print(f"\n  Results saved to {results_path}")
    print(f"  Attention geometry saved to {geom_path}")
    print(f"{'='*60}\n")

    return results

