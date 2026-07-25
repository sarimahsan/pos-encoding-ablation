"""Integration tests for trainer, evaluator, artifact management, and multi-seed aggregation."""

import os
import torch
import pytest
from torch.utils.data import DataLoader

from config import TransformerConfig
from components import DecoderTransformer
from data import PackedTextDataset
from trainer import Trainer
from evaluate import (
    evaluate_at_length,
    compute_attention_entropy_detailed,
    run_full_evaluation,
    generate_attention_heatmap,
)
from utils.artifacts import ArtifactManager
from run_experiment import compute_aggregated_metrics


def create_dummy_dataloader(seq_len=64, num_samples=20, batch_size=4):
    fake_tokens = torch.randint(0, 1000, (num_samples * (seq_len + 1),)).tolist()
    ds = PackedTextDataset(fake_tokens, seq_len)
    return DataLoader(ds, batch_size=batch_size)


def test_artifact_manager(tmp_path):
    output_dir = str(tmp_path / "test_out")
    cfg = TransformerConfig(output_dir=output_dir)
    mgr = ArtifactManager(cfg)

    mgr.save_config()
    assert os.path.exists(os.path.join(output_dir, "config", "config.json"))

    dummy_loaders = {"val_64": create_dummy_dataloader()}
    mgr.save_eval_batch_indices(dummy_loaders)
    assert os.path.exists(os.path.join(output_dir, "eval_data", "eval_batch_indices.json"))


def test_short_training_and_eval_pipeline(tmp_path):
    output_dir = str(tmp_path / "test_pipeline")
    cfg = TransformerConfig(
        output_dir=output_dir,
        train_seq_len=64,
        eval_seq_lens=[64, 128],
        n_layers=2,
        d_model=128,
        n_heads=4,
        head_dim=32,
        d_ff=256,
        train_steps=10,
        log_interval=5,
        eval_interval=5,
        save_interval=10,
    )
    model = DecoderTransformer(cfg)

    dataloaders = {
        "train": create_dummy_dataloader(seq_len=64, num_samples=16, batch_size=2),
        "val_64": create_dummy_dataloader(seq_len=64, num_samples=8, batch_size=2),
        "val_128": create_dummy_dataloader(seq_len=128, num_samples=4, batch_size=2),
    }

    trainer = Trainer(cfg, model, dataloaders)
    trainer.train()

    assert os.path.exists(os.path.join(output_dir, "config", "config.json"))
    assert os.path.exists(os.path.join(output_dir, "logs", "train_log.csv"))
    assert os.path.exists(os.path.join(output_dir, "logs", "train_log.jsonl"))
    assert os.path.exists(os.path.join(output_dir, "metrics", "val_metrics_train_length.json"))
    assert os.path.exists(os.path.join(output_dir, "metrics", "extrapolation_results.json"))
    assert os.path.exists(os.path.join(output_dir, "metrics", "attention_entropy.json"))
    assert os.path.exists(os.path.join(output_dir, "metadata", "run_metadata.json"))

    generate_attention_heatmap(
        model, dataloaders["val_64"], cfg, trainer.device, use_amp=False,
        save_dir=os.path.join(output_dir, "heatmaps")
    )
    assert len(os.listdir(os.path.join(output_dir, "heatmaps"))) >= 2


def test_compute_aggregated_metrics():
    dummy_results = [
        {
            "seed": 42,
            "run_name": "R1_rope_seq256",
            "pos_encoding": "rope",
            "train_seq_len": 256,
            "metrics": {
                "256": {"type": "train-length", "loss": 3.0, "perplexity": 20.0, "attention_entropy_mean": 2.5},
                "768": {"type": "extrapolation", "loss": 4.0, "perplexity": 50.0, "attention_entropy_mean": 3.5},
            },
        },
        {
            "seed": 43,
            "run_name": "R1_rope_seq256",
            "pos_encoding": "rope",
            "train_seq_len": 256,
            "metrics": {
                "256": {"type": "train-length", "loss": 3.2, "perplexity": 24.0, "attention_entropy_mean": 2.7},
                "768": {"type": "extrapolation", "loss": 4.2, "perplexity": 54.0, "attention_entropy_mean": 3.7},
            },
        },
    ]

    agg = compute_aggregated_metrics(dummy_results)
    assert agg["num_seeds"] == 2
    assert agg["aggregated_metrics"]["256"]["loss"]["mean"] == 3.1
    assert round(agg["aggregated_metrics"]["256"]["loss"]["std"], 2) == 0.1
    assert agg["aggregated_metrics"]["256"]["perplexity"]["mean"] == 22.0
