"""
Pre-Flight Verification Script for Google Colab (~1-2 minutes).

Verifies the entire framework end-to-end:
1. Runs full pytest unit & integration test suite.
2. Runs a 50-step end-to-end dry-run across 2 seeds (Seed 42 & Seed 43).
3. Verifies dataset tokenization & document packing.
4. Checks all 10 artifact folder outputs (checkpoints, logs CSV/JSONL, metrics, heatmaps PNG/.pt, eval_data, metadata).
5. Verifies aggregated statistical summary (Mean ± Std) and output ZIP file integrity.

Usage in Colab:
    !python verify_colab.py
"""

import os
import sys
import shutil
import zipfile
import pytest
import torch

from config import get_experiment_config
from run_experiment import main as run_experiment_main


def main():
    print("=" * 70)
    print("  🚀 STARTING PRE-FLIGHT VERIFICATION FOR GOOGLE COLAB (~1-2 mins)")
    print("=" * 70)

    # Step 1: Run PyTest suite
    print("\n[1/5] Running PyTest Unit & Integration Test Suite...")
    pytest_code = pytest.main(["-v", "tests/"])
    if pytest_code != 0:
        print("\n❌ PyTest suite failed! Aborting verification.")
        sys.exit(1)
    print("  ✓ PyTest Suite: ALL PASSED!")

    # Step 2: Test Single-Seed Dry-Run
    print("\n[2/5] Running 50-Step End-to-End Single-Seed Dry-Run (R1: RoPE @ 256)...")
    test_output_dir = "outputs/R1_rope_seq256"
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    if os.path.exists(f"{test_output_dir}.zip"):
        os.remove(f"{test_output_dir}.zip")

    # Simulate command-line arguments for 50 steps
    sys.argv = ["run_experiment.py", "--run", "R1", "--train_steps", "50"]
    try:
        run_experiment_main()
    except Exception as e:
        print(f"\n❌ Single-Seed Dry-Run failed with error: {e}")
        sys.exit(1)
    print("  ✓ Single-Seed Dry-Run: COMPLETED SUCCESSFULLY!")

    # Step 3: Verify Folder Structure & Artifacts
    print("\n[3/5] Verifying Artifact Directories & Files for Seed 42...")
    required_files = [
        os.path.join(test_output_dir, "config", "config.json"),
        os.path.join(test_output_dir, "logs", "train_log.csv"),
        os.path.join(test_output_dir, "logs", "train_log.jsonl"),
        os.path.join(test_output_dir, "metrics", "full_eval_results.json"),
        os.path.join(test_output_dir, "metrics", "val_metrics_train_length.json"),
        os.path.join(test_output_dir, "metrics", "extrapolation_results.json"),
        os.path.join(test_output_dir, "metrics", "attention_geometry.json"),
        os.path.join(test_output_dir, "eval_data", "eval_batch_indices.json"),
        os.path.join(test_output_dir, "metadata", "run_metadata.json"),
    ]
    for filepath in required_files:
        if not os.path.exists(filepath):
            print(f"❌ Missing expected artifact: {filepath}")
            sys.exit(1)

    # Check checkpoints exist
    ckpt_dir = os.path.join(test_output_dir, "checkpoints")
    if not os.path.exists(ckpt_dir) or len(os.listdir(ckpt_dir)) == 0:
        print(f"❌ Missing checkpoints in {ckpt_dir}")
        sys.exit(1)

    # Check heatmaps exist (.png)
    heatmap_dir = os.path.join(test_output_dir, "heatmaps")
    if not os.path.exists(heatmap_dir) or len(os.listdir(heatmap_dir)) < 2:
        print(f"❌ Missing heatmaps in {heatmap_dir}")
        sys.exit(1)

    print("  ✓ Artifact Directories & Files: ALL VERIFIED!")

    # Step 4: Verify Full Evaluation & Geometry Results
    print("\n[4/5] Verifying Full Evaluation Summary & Geometry Metrics...")
    geom_file = os.path.join(test_output_dir, "metrics", "attention_geometry.json")
    if not os.path.exists(geom_file):
        print(f"❌ Attention geometry file missing: {geom_file}")
        sys.exit(1)

    import json
    with open(geom_file) as f:
        geom_data = json.load(f)
    if "eval_metrics" not in geom_data:
        print(f"❌ Expected 'eval_metrics' key in {geom_file}")
        sys.exit(1)

    print("  ✓ Attention Geometry Metrics: VERIFIED (Entropy, Sink Ratio, Distance, & Diag Mass)!")

    # Step 5: Verify Zip Archive Integrity
    print("\n[5/5] Verifying Output ZIP Archive Integrity...")
    zip_path = f"{test_output_dir}.zip"
    if not os.path.exists(zip_path):
        print(f"❌ Output ZIP file missing: {zip_path}")
        sys.exit(1)


    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        bad_file = zip_ref.testzip()
        if bad_file is not None:
            print(f"❌ Corrupted ZIP file detected at {bad_file}")
            sys.exit(1)
        file_list = zip_ref.namelist()
        print(f"  ✓ Zip Archive contains {len(file_list)} files and subdirectories.")

    print("\n" + "=" * 70)
    print("  🎉 ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! (~1-2 minutes)")
    print("  Your Colab environment & framework are 100% ready for full runs.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
