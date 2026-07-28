"""
Training loop — AdamW, cosine LR, fp16 mixed precision, gradient clipping.
"""

import os
import csv
import math
import time
import json
import datetime
import torch
from torch.amp import GradScaler, autocast

from config import TransformerConfig
from components import DecoderTransformer
from evaluate import evaluate_at_length, compute_attention_metrics_detailed
from evaluate.heatmaps import generate_attention_heatmap
from utils.artifacts import ArtifactManager
from .scheduler import build_cosine_scheduler


class Trainer:
    """End-to-end training manager for a single experiment run."""

    def __init__(
        self,
        config: TransformerConfig,
        model: DecoderTransformer,
        dataloaders: dict,
    ):
        self.config = config
        self.model = model
        self.dataloaders = dataloaders
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        if hasattr(torch, "compile") and self.device.type == "cuda" and getattr(config, "compile", False):
            print("  Compiling model with PyTorch 2.x torch.compile()...")
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"  torch.compile failed, continuing uncompiled: {e}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

        self.scheduler = build_cosine_scheduler(self.optimizer, config)
        self.scaler = GradScaler("cuda", enabled=(config.precision == "fp16"))
        self.use_amp = config.precision == "fp16"

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_results = []
        self.anomalies = []
        self.wall_clock_start = None
        self.wall_clock_end = None

        self.artifacts = ArtifactManager(config)
        self.artifacts.save_config()
        self.artifacts.save_eval_batch_indices(dataloaders)

        self.wandb_run = None
        if config.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project="pos-encoding-ablation",
                    name=config.run_name,
                    config=config.to_dict(),
                )
            except ImportError:
                print("wandb not installed, falling back to CSV logging.")

    def train(self):
        cfg = self.config
        model = self.model
        train_loader = self.dataloaders["train"]

        print(f"\n{'='*60}")
        print(f"  Starting training on {self.device}")
        print(f"  Model params: {model.count_parameters():,}")
        print(f"{'='*60}\n")

        self.wall_clock_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.train()
        train_iter = iter(train_loader)
        running_loss = 0.0
        step_start = time.time()

        jsonl_path = self.artifacts.path("logs", "train_log.jsonl")
        csv_path = self.artifacts.path("logs", "train_log.csv")
        jsonl_file = open(jsonl_path, "w")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=["step", "train_loss", "perplexity", "lr", "steps_per_sec"],
        )
        csv_writer.writeheader()

        try:
            for step in range(1, cfg.train_steps + 1):
                self.global_step = step
                self.optimizer.zero_grad()
                accum_loss = 0.0

                for _ in range(cfg.grad_accum_steps):
                    try:
                        input_ids, targets = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        input_ids, targets = next(train_iter)

                    input_ids = input_ids.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    with autocast("cuda", enabled=self.use_amp):
                        output = model(input_ids, targets)
                        loss = output["loss"] / cfg.grad_accum_steps

                    self.scaler.scale(loss).backward()
                    accum_loss += loss.item()

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                running_loss += accum_loss

                if step % cfg.log_interval == 0:
                    avg_loss = running_loss / cfg.log_interval
                    ppl = math.exp(min(avg_loss, 20))
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - step_start
                    steps_per_sec = cfg.log_interval / elapsed

                    entry = {
                        "step": step,
                        "train_loss": round(avg_loss, 4),
                        "perplexity": round(ppl, 2),
                        "lr": round(lr, 8),
                        "steps_per_sec": round(steps_per_sec, 2),
                    }
                    self.train_losses.append(entry)
                    jsonl_file.write(json.dumps(entry) + "\n")
                    jsonl_file.flush()
                    csv_writer.writerow(entry)
                    csv_file.flush()

                    print(
                        f"  Step {step:>6d}/{cfg.train_steps} | "
                        f"loss {avg_loss:.4f} | ppl {ppl:.2f} | "
                        f"lr {lr:.2e} | {steps_per_sec:.1f} steps/s"
                    )

                    if self.wandb_run:
                        import wandb
                        wandb.log(entry, step=step)

                    running_loss = 0.0
                    step_start = time.time()

                if step % cfg.eval_interval == 0:
                    self._run_eval(step)

                if step % cfg.save_interval == 0:
                    self._save_checkpoint(step)

        except KeyboardInterrupt:
            self.anomalies.append({
                "step": self.global_step, "type": "keyboard_interrupt",
                "timestamp": datetime.datetime.now().isoformat(),
            })
            print(f"\n  Training interrupted at step {self.global_step}")
        except Exception as e:
            self.anomalies.append({
                "step": self.global_step, "type": "exception",
                "message": str(e),
                "timestamp": datetime.datetime.now().isoformat(),
            })
            raise
        finally:
            self.wall_clock_end = time.time()
            jsonl_file.close()
            csv_file.close()

        # HARD CHECKPOINT TRIGGER: Save final weights & metadata IMMEDIATELY
        # before running final eval or heatmaps so weights/logs are safe on disk.
        print(f"\n  [HARD TRIGGER] Saving final model checkpoint & metadata at step {self.global_step}...")
        self._save_checkpoint(self.global_step, final=True)
        self.artifacts.save_run_metadata(
            self.model, self.global_step, self.best_val_loss,
            self.wall_clock_start, self.wall_clock_end, self.anomalies,
        )

        # Post-training evaluation
        self._run_eval(self.global_step, final=True)

        if self.wandb_run:
            self.wandb_run.finish()

        print(f"\n{'='*60}")
        print(f"  Training complete! Best val loss: {self.best_val_loss:.4f}")
        print(f"  All artifacts saved to: {cfg.output_dir}/")
        print(f"{'='*60}\n")

    def _run_eval(self, step: int, final: bool = False):
        cfg = self.config
        self.model.eval()

        eval_results = {"step": step}
        train_len_metrics = {}
        extrap_metrics = {}

        print(f"\n  {'─'*50}")
        print(f"  Evaluation at step {step}")

        for eval_len in cfg.eval_seq_lens:
            key = f"val_{eval_len}"
            if key not in self.dataloaders:
                continue

            val_loss, val_ppl = evaluate_at_length(
                self.model, self.dataloaders[key], self.device, self.use_amp
            )
            eval_results[f"val_loss_{eval_len}"] = round(val_loss, 4)
            eval_results[f"val_ppl_{eval_len}"] = round(val_ppl, 2)

            is_train_len = eval_len == cfg.train_seq_len
            label = "train-len" if is_train_len else "extrap"
            print(f"    seq_len={eval_len:>4d} ({label:>9s}): loss={val_loss:.4f}, ppl={val_ppl:.2f}")

            entry = {"loss": round(val_loss, 4), "perplexity": round(val_ppl, 2)}
            if is_train_len:
                train_len_metrics[str(eval_len)] = entry
            else:
                extrap_metrics[str(eval_len)] = entry

            if is_train_len and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(step, best=True)

        train_key = f"val_{cfg.train_seq_len}"
        geom_data = None
        if train_key in self.dataloaders:
            geom_data = compute_attention_metrics_detailed(
                self.model, self.dataloaders[train_key], self.device,
                self.use_amp, max_batches=5, K=16
            )
            eval_results["attn_entropy_mean"] = geom_data["entropy"]["mean"]
            eval_results["attn_sink_ratio_mean"] = geom_data["sink_ratio"]["mean"]
            eval_results["effective_distance_mean"] = geom_data["effective_distance"]["mean"]
            eval_results["diagonal_mass_ratio_mean"] = geom_data["diagonal_mass_ratio"]["mean"]
            print(
                f"    attn_entropy={geom_data['entropy']['mean']:.4f} | "
                f"sink_ratio={geom_data['sink_ratio']['mean']:.4f} | "
                f"eff_dist={geom_data['effective_distance']['mean']:.2f} | "
                f"diag_mass(K=16)={geom_data['diagonal_mass_ratio']['mean']:.4f}"
            )

        self.val_results.append(eval_results)

        if self.wandb_run:
            import wandb
            wandb.log(eval_results, step=step)

        log_path = self.artifacts.path("logs", "eval_results.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(eval_results) + "\n")

        print(f"  {'─'*50}\n")

        if final:
            self.artifacts.save_json("metrics", "val_metrics_train_length.json", {
                "run_name": cfg.run_name, "pos_encoding": cfg.pos_encoding,
                "train_seq_len": cfg.train_seq_len, "step": step,
                "best_val_loss": self.best_val_loss, "final_metrics": train_len_metrics,
            })
            self.artifacts.save_json("metrics", "extrapolation_results.json", {
                "run_name": cfg.run_name, "pos_encoding": cfg.pos_encoding,
                "train_seq_len": cfg.train_seq_len, "step": step,
                "extrapolation_lengths": [l for l in cfg.eval_seq_lens if l != cfg.train_seq_len],
                "results": extrap_metrics,
            })
            if geom_data:
                self.artifacts.save_json("metrics", "attention_entropy.json", {
                    "run_name": cfg.run_name, "pos_encoding": cfg.pos_encoding,
                    "step": step, "mean_entropy": geom_data["entropy"]["mean"],
                    "per_layer": geom_data["entropy"]["per_layer"],
                    "per_layer_per_head": geom_data["entropy"]["per_layer_per_head"],
                })
                self.artifacts.save_json("metrics", "attention_geometry.json", {
                    "run_name": cfg.run_name, "pos_encoding": cfg.pos_encoding,
                    "train_seq_len": cfg.train_seq_len, "step": step,
                    "eval_metrics": {
                        str(cfg.train_seq_len): geom_data
                    },
                })
            self.artifacts.save_json("metrics", "final_summary.json", {
                "run_name": cfg.run_name, "pos_encoding": cfg.pos_encoding,
                "train_seq_len": cfg.train_seq_len,
                "best_val_loss": self.best_val_loss,
                "eval_history": self.val_results,
                "train_log_tail": self.train_losses[-10:],
            })

        self.model.train()

    def _save_checkpoint(self, step: int, best: bool = False, final: bool = False):
        cfg = self.config
        if best:
            filename = f"run_{cfg.run_name}_best.pt"
        elif final:
            filename = f"run_{cfg.run_name}_step{step}_final.pt"
        else:
            filename = f"run_{cfg.run_name}_step{step}.pt"

        path = self.artifacts.path("checkpoints", filename)
        raw_model = getattr(self.model, "_orig_mod", self.model)
        torch.save({
            "step": step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": cfg.to_dict(),
            "best_val_loss": self.best_val_loss,
        }, path)
        tag = " (best)" if best else (" (final)" if final else "")
        print(f"  Checkpoint saved{tag}: {path}")

    def generate_heatmaps(self):
        cfg = self.config
        save_dir = self.artifacts.path("heatmaps")
        print(f"\n  Generating attention heatmaps...")
        for eval_len in cfg.eval_seq_lens:
            key = f"val_{eval_len}"
            if key not in self.dataloaders:
                continue
            for layer_idx in [0, cfg.n_layers - 1]:
                generate_attention_heatmap(
                    self.model, self.dataloaders[key], cfg, self.device,
                    use_amp=(cfg.precision == "fp16"),
                    layer_idx=layer_idx, head_idx=0,
                    max_tokens=min(64, eval_len),
                    save_dir=save_dir,
                )
