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
    assert os.path.exists(os.path.join(output_dir, "metrics", "attention_geometry.json"))
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


def test_csv_and_json_logging_integrity(tmp_path):
    """Verify CSV, JSONL, and JSON metrics logging integrity and non-empty outputs."""
    import csv
    import json

    output_dir = str(tmp_path / "test_logging")
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

    # 1. Verify train_log.csv content & structure
    csv_path = os.path.join(output_dir, "logs", "train_log.csv")
    assert os.path.exists(csv_path), "train_log.csv does not exist"

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        expected_fields = ["step", "train_loss", "perplexity", "lr", "steps_per_sec"]
        for field in expected_fields:
            assert field in fieldnames, f"Missing field '{field}' in train_log.csv header"

        rows = list(reader)
        assert len(rows) == 2, f"Expected 2 log rows in train_log.csv, got {len(rows)}"

        for row in rows:
            assert int(row["step"]) in [5, 10]
            assert float(row["train_loss"]) > 0.0
            assert float(row["perplexity"]) > 0.0
            assert float(row["lr"]) > 0.0
            assert float(row["steps_per_sec"]) >= 0.0

    # 2. Verify train_log.jsonl content & structure
    jsonl_path = os.path.join(output_dir, "logs", "train_log.jsonl")
    assert os.path.exists(jsonl_path), "train_log.jsonl does not exist"
    with open(jsonl_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "step" in data
            assert "train_loss" in data
            assert "perplexity" in data
            assert "lr" in data
            assert "steps_per_sec" in data

    # 3. Verify eval_results.jsonl contains attention geometry metrics
    eval_jsonl_path = os.path.join(output_dir, "logs", "eval_results.jsonl")
    assert os.path.exists(eval_jsonl_path), "eval_results.jsonl does not exist"
    with open(eval_jsonl_path, "r") as f:
        eval_lines = f.readlines()
        assert len(eval_lines) >= 2
        for line in eval_lines:
            entry = json.loads(line)
            assert "val_loss_64" in entry
            assert "val_ppl_64" in entry
            assert "attn_entropy_mean" in entry
            assert "attn_sink_ratio_mean" in entry
            assert "effective_distance_mean" in entry
            assert "diagonal_mass_ratio_mean" in entry

    # 4. Verify attention_geometry.json content & non-empty 8x8 matrices
    geom_file = os.path.join(output_dir, "metrics", "attention_geometry.json")
    assert os.path.exists(geom_file), "attention_geometry.json does not exist"
    with open(geom_file, "r") as f:
        geom_data = json.load(f)
        assert "eval_metrics" in geom_data
        eval_m = geom_data["eval_metrics"]["64"]
        for key in ["entropy", "sink_ratio", "effective_distance", "diagonal_mass_ratio"]:
            assert key in eval_m
            assert "mean" in eval_m[key]
            assert "per_layer" in eval_m[key]
            assert "per_layer_per_head" in eval_m[key]
            assert len(eval_m[key]["per_layer"]) == 2
            assert len(eval_m[key]["per_layer_per_head"]) == 2
            assert len(eval_m[key]["per_layer_per_head"][0]) == 4

    # 5. Verify full_eval_results.json after run_full_evaluation
    eval_results = run_full_evaluation(model, dataloaders, cfg, trainer.device, use_amp=False)
    assert "metrics" in eval_results
    assert "64" in eval_results["metrics"]
    assert "128" in eval_results["metrics"]
    for l_str in ["64", "128"]:
        m = eval_results["metrics"][l_str]
        assert "loss" in m
        assert "perplexity" in m
        assert "attention_entropy_mean" in m
        assert "attention_sink_ratio_mean" in m
        assert "effective_distance_mean" in m
        assert "diagonal_mass_ratio_mean" in m

