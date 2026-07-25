"""
ArtifactManager — creates and manages the output folder structure.
"""

import os
import json
import shutil
import platform
import datetime
import torch

from config import TransformerConfig


class ArtifactManager:
    """Manages the organized output folder for a single experiment run."""

    SUBDIRS = [
        "checkpoints",
        "config",
        "logs",
        "metrics",
        "heatmaps",
        "eval_data",
        "metadata",
    ]

    def __init__(self, config: TransformerConfig):
        self.config = config
        self.base = config.output_dir
        self.dirs = {}
        self._create_dirs()

    def _create_dirs(self):
        self.dirs["root"] = self.base
        for name in self.SUBDIRS:
            path = os.path.join(self.base, name)
            os.makedirs(path, exist_ok=True)
            self.dirs[name] = path

    def path(self, subdir: str, filename: str = "") -> str:
        base = self.dirs.get(subdir, self.base)
        return os.path.join(base, filename) if filename else base

    def save_config(self):
        path = self.path("config", "config.json")
        with open(path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        print(f"  Config saved: {path}")

    def save_eval_batch_indices(self, dataloaders: dict):
        indices_info = {}
        for key, loader in dataloaders.items():
            if key.startswith("val_"):
                ds = loader.dataset
                indices_info[key] = {
                    "num_blocks": len(ds),
                    "seq_len": ds.seq_len,
                    "total_tokens": len(ds) * (ds.seq_len + 1),
                    "first_block_hash": int(ds.data[0].sum().item()) if len(ds) > 0 else None,
                    "last_block_hash": int(ds.data[-1].sum().item()) if len(ds) > 0 else None,
                }
        path = self.path("eval_data", "eval_batch_indices.json")
        with open(path, "w") as f:
            json.dump(indices_info, f, indent=2)
        print(f"  Eval batch indices saved: {path}")

    def save_run_metadata(
        self,
        model,
        global_step: int,
        best_val_loss: float,
        wall_clock_start: float,
        wall_clock_end: float,
        anomalies: list,
    ):
        cfg = self.config
        wall_secs = wall_clock_end - wall_clock_start if wall_clock_start and wall_clock_end else 0

        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "total_memory_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2),
                "capability": list(torch.cuda.get_device_capability(0)),
                "cuda_version": torch.version.cuda or "N/A",
            }

        metadata = {
            "run_name": cfg.run_name,
            "pos_encoding": cfg.pos_encoding,
            "train_seq_len": cfg.train_seq_len,
            "planned_steps": cfg.train_steps,
            "actual_steps_reached": global_step,
            "early_stopped": global_step < cfg.train_steps,
            "best_val_loss": best_val_loss,
            "wall_clock_seconds": round(wall_secs, 1),
            "wall_clock_human": str(datetime.timedelta(seconds=int(wall_secs))),
            "start_time": datetime.datetime.fromtimestamp(wall_clock_start).isoformat() if wall_clock_start else None,
            "end_time": datetime.datetime.fromtimestamp(wall_clock_end).isoformat() if wall_clock_end else None,
            "gpu": gpu_info,
            "system": {
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "os": platform.platform(),
            },
            "model_params": model.count_parameters(),
            "anomalies": anomalies,
            "seed": cfg.seed,
        }
        path = self.path("metadata", "run_metadata.json")
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Run metadata saved: {path}")

    def save_json(self, subdir: str, filename: str, data: dict):
        path = self.path(subdir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def print_tree(self):
        print(f"\n{'='*60}")
        print(f"  Artifact folder: {self.base}/")
        print(f"{'='*60}")
        for dirpath, dirnames, filenames in os.walk(self.base):
            level = dirpath.replace(self.base, "").count(os.sep)
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
        print(f"{'='*60}\n")

    def zip_for_download(self) -> str:
        zip_path = self.base.rstrip("/\\")
        shutil.make_archive(zip_path, "zip", self.base)
        zip_file = f"{zip_path}.zip"
        print(f"  Downloadable zip: {zip_file}")
        print(f"    from google.colab import files")
        print(f"    files.download('{zip_file}')")
        return zip_file
