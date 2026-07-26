"""
CLI entry point for running positional encoding ablation experiments.

Runs 2 seeds automatically (default: seed 42 & seed 43) per experiment,
saves outputs in separate seed subfolders, generates an aggregated statistical
summary (mean ± std), and zips the combined output folder for download.

Usage:
    python run_experiment.py --run R1    # Runs RoPE @ 256 for Seed 42 & Seed 43
    python run_experiment.py --run R2    # Runs RoPE @ 512 for Seed 42 & Seed 43
    python run_experiment.py --run R3    # Runs ALiBi @ 256 for Seed 42 & Seed 43
    python run_experiment.py --run R4    # Runs ALiBi @ 512 for Seed 42 & Seed 43
    python run_experiment.py --run R5    # Runs NoPE @ 256 for Seed 42 & Seed 43
    python run_experiment.py --run R6    # Runs NoPE @ 512 for Seed 42 & Seed 43

Optional overrides:
    --seeds 42 43 44    Override seeds to run (default: [42, 43])
    --train_steps N     Override training steps per seed (default: 50000)
    --batch_size N      Override batch size
    --grad_accum N      Gradient accumulation steps
    --wandb             Enable Weights & Biases logging
    --eval_only         Skip training, run eval on saved checkpoints
"""

import argparse
import os
import json
import shutil
import math
import numpy as np
import torch

from config import get_experiment_config
from components import DecoderTransformer
from data import get_dataloaders
from trainer import Trainer
from evaluate import run_full_evaluation, generate_attention_heatmap
from utils import set_seed, ArtifactManager


def compute_aggregated_metrics(seed_results: list) -> dict:
    """
    Compute mean and std across seeds for each metric and evaluation length.
    """
    if not seed_results:
        return {}

    eval_lens = list(seed_results[0]["metrics"].keys())
    aggregated = {
        "num_seeds": len(seed_results),
        "seeds": [r["seed"] for r in seed_results],
        "run_name": seed_results[0]["run_name"],
        "pos_encoding": seed_results[0]["pos_encoding"],
        "train_seq_len": seed_results[0]["train_seq_len"],
        "aggregated_metrics": {},
    }

    for length in eval_lens:
        losses = [r["metrics"][length]["loss"] for r in seed_results if length in r["metrics"]]
        ppls = [r["metrics"][length]["perplexity"] for r in seed_results if length in r["metrics"]]
        entropies = [r["metrics"][length].get("attention_entropy_mean", 0.0) for r in seed_results if length in r["metrics"]]

        metric_type = seed_results[0]["metrics"][length]["type"]

        aggregated["aggregated_metrics"][length] = {
            "type": metric_type,
            "loss": {
                "mean": round(float(np.mean(losses)), 4),
                "std": round(float(np.std(losses)), 4),
                "values": losses,
            },
            "perplexity": {
                "mean": round(float(np.mean(ppls)), 2),
                "std": round(float(np.std(ppls)), 2),
                "values": ppls,
            },
            "attention_entropy": {
                "mean": round(float(np.mean(entropies)), 4),
                "std": round(float(np.std(entropies)), 4),
                "values": entropies,
            },
        }

    return aggregated


def run_single_seed(args, base_config, seed: int, experiment_base_dir: str) -> dict:
    """Run full training and evaluation for a single seed."""
    config = get_experiment_config(args.run)

    # Apply overrides
    if args.train_steps is not None:
        config.train_steps = args.train_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.grad_accum is not None:
        config.grad_accum_steps = args.grad_accum
    if args.wandb:
        config.use_wandb = True

    config.seed = seed
    config.output_dir = os.path.join(experiment_base_dir, f"seed_{seed}")
    config.run_name = f"{base_config.run_name}_seed{seed}"

    print(f"\n{'='*60}")
    print(f"  STARTING RUN: {config.run_name} (Seed {seed})")
    print(f"{'='*60}\n")

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("  Building model...")
    model = DecoderTransformer(config)
    print(f"  Total parameters: {model.count_parameters():,} ({model.count_parameters()/1e6:.1f}M)")

    dataloaders = get_dataloaders(config)
    artifacts = ArtifactManager(config)

    if args.eval_only:
        ckpt_dir = artifacts.path("checkpoints")
        ckpt_path = None
        for pattern in ["best", "final"]:
            for f in os.listdir(ckpt_dir) if os.path.exists(ckpt_dir) else []:
                if pattern in f and f.endswith(".pt"):
                    ckpt_path = os.path.join(ckpt_dir, f)
                    break
            if ckpt_path:
                break

        if not ckpt_path:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir} for seed {seed}.")

        print(f"  Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
    else:
        trainer = Trainer(config, model, dataloaders)
        trainer.train()

        ckpt_dir = artifacts.path("checkpoints")
        best_files = [f for f in os.listdir(ckpt_dir) if "best" in f and f.endswith(".pt")] if os.path.exists(ckpt_dir) else []
        if best_files:
            best_path = os.path.join(ckpt_dir, best_files[0])
            print(f"  Loading best checkpoint for final evaluation: {best_path}")
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

        trainer.generate_heatmaps()

    metrics_result = run_full_evaluation(
        model, dataloaders, config, device,
        use_amp=(config.precision == "fp16"),
        save_dir=artifacts.path("metrics"),
    )
    metrics_result["seed"] = seed

    return metrics_result


def main():
    parser = argparse.ArgumentParser(
        description="Positional Encoding Ablation — Multi-Seed Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiment Matrix:
  R1: RoPE  @ seq_len=256    R2: RoPE  @ seq_len=512
  R3: ALiBi @ seq_len=256    R4: ALiBi @ seq_len=512
  R5: NoPE  @ seq_len=256    R6: NoPE  @ seq_len=512

Runs 2 seeds automatically (seed 42 & seed 43) and saves both in one zip file.
        """,
    )
    parser.add_argument(
        "--run", type=str, required=True,
        choices=["R1", "R2", "R3", "R4", "R5", "R6",
                 "r1", "r2", "r3", "r4", "r5", "r6"],
        help="Experiment run ID (R1-R6)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43],
                        help="Seeds to run (default: 42 43)")
    parser.add_argument("--train_steps", type=int, default=None, help="Override training steps per seed")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--eval_only", action="store_true", help="Skip training, eval saved checkpoints")

    args = parser.parse_args()

    base_config = get_experiment_config(args.run)
    experiment_base_dir = f"outputs/{base_config.run_name}"
    os.makedirs(experiment_base_dir, exist_ok=True)

    print(base_config.summary())
    print(f"  Seeds to execute: {args.seeds} (Automatic 2-Seed Mode)\n")

    seed_results = []
    for s in args.seeds:
        res = run_single_seed(args, base_config, s, experiment_base_dir)
        seed_results.append(res)

    # Compute and save aggregated metrics (Mean ± Std)
    aggregated = compute_aggregated_metrics(seed_results)
    agg_path = os.path.join(experiment_base_dir, "aggregated_summary.json")
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  AGGREGATED RESULTS ACROSS {len(args.seeds)} SEEDS (Mean ± Std)")
    print(f"{'='*60}")
    for length, m in aggregated["aggregated_metrics"].items():
        print(
            f"  seq_len={int(length):>4d} ({m['type']:>14s}): "
            f"Loss={m['loss']['mean']:.4f} ± {m['loss']['std']:.4f} | "
            f"PPL={m['perplexity']['mean']:.2f} ± {m['perplexity']['std']:.2f} | "
            f"Entropy={m['attention_entropy']['mean']:.4f} ± {m['attention_entropy']['std']:.4f}"
        )
    print(f"  Saved aggregated summary: {agg_path}")

    # Print folder tree of full experiment output
    print(f"\n{'='*60}")
    print(f"  Full Experiment Artifact Tree: {experiment_base_dir}/")
    print(f"{'='*60}")
    for dirpath, dirnames, filenames in os.walk(experiment_base_dir):
        level = dirpath.replace(experiment_base_dir, "").count(os.sep)
        indent = "  " * (level + 1)
        subindent = "  " * (level + 2)
        print(f"{indent}{os.path.basename(dirpath)}/")
        for f in sorted(filenames):
            size = os.path.getsize(os.path.join(dirpath, f))
            if size > 1e6:
                size_str = f"{size/1e6:.1f} MB"
            elif size > 1e3:
                size_str = f"{size/1e3:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"{subindent}{f}  ({size_str})")

    # Zip output folder containing both seeds
    zip_path = experiment_base_dir.rstrip("/\\")
    shutil.make_archive(zip_path, "zip", experiment_base_dir)
    zip_file = f"{zip_path}.zip"

    print(f"\n{'='*60}")
    print(f"  DONE! Combined 2-seed experiment complete.")
    print(f"  Downloadable zip (contains Seed 42 + Seed 43 + Aggregated Results):")
    print(f"    {zip_file}")
    print(f"  From Google Colab:")
    print(f"    from google.colab import files")
    print(f"    files.download('{zip_file}')")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
