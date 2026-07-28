"""
Evaluation metrics — loss, perplexity, attention entropy.

Primary:
    - Validation loss & perplexity (at train length + extrapolation)

Secondary:
    - Attention entropy: per-head, per-layer, and mean
      Low entropy = sharp/focused; high entropy = diffuse/struggling
"""

import math
import torch
from torch.amp import autocast


@torch.no_grad()
def evaluate_at_length(
    model, dataloader, device, use_amp: bool = True, max_batches: int = None
) -> tuple:
    """
    Compute average loss and perplexity on a dataloader.

    Returns:
        (avg_loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for i, (input_ids, targets) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with autocast("cuda", enabled=use_amp):
            output = model(input_ids, targets)
            total_loss += output["loss"].item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


@torch.no_grad()
def compute_attention_metrics_detailed(
    model, dataloader, device, use_amp: bool = True, max_batches: int = 5, K: int = 16
) -> dict:
    """
    Compute comprehensive attention geometry metrics:
        1. Attention Entropy (-sum(p * log p))
        2. Attention Sink Ratio (% mass on token 0)
        3. Effective Attention Distance (sum((i - j) * A_ij))
        4. Diagonal Mass Ratio (sum(A_ij for |i - j| <= K))

    Returns:
        {
            "entropy": {"mean": float, "per_layer": [...], "per_layer_per_head": [[...]]},
            "sink_ratio": {"mean": float, "per_layer": [...], "per_layer_per_head": [[...]]},
            "effective_distance": {"mean": float, "per_layer": [...], "per_layer_per_head": [[...]]},
            "diagonal_mass_ratio": {"K": 16, "mean": float, "per_layer": [...], "per_layer_per_head": [[...]]}
        }
    """
    model.eval()
    n_layers = len(model.blocks)
    n_heads = model.config.n_heads

    layer_head_entropy = [[0.0] * n_heads for _ in range(n_layers)]
    layer_head_sink = [[0.0] * n_heads for _ in range(n_layers)]
    layer_head_dist = [[0.0] * n_heads for _ in range(n_layers)]
    layer_head_diag = [[0.0] * n_heads for _ in range(n_layers)]
    n_samples = 0

    for i, (input_ids, targets) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with autocast("cuda", enabled=use_amp):
            output = model(input_ids, targets, return_attn=True)

        for layer_idx, attn_w in enumerate(output["attn_weights"]):
            # attn_w shape: (batch_size, n_heads, seq_len, seq_len)
            attn_w = attn_w.float()
            seq_len = attn_w.shape[-1]
            attn_w_clamped = attn_w.clamp(min=1e-8)

            # 1. Attention Entropy
            entropy = -(attn_w_clamped * attn_w_clamped.log()).sum(dim=-1)
            entropy_per_head = entropy.mean(dim=(0, 2))

            # 2. Attention Sink Ratio (% mass on token 0)
            sink_per_head = attn_w[:, :, :, 0].mean(dim=(0, 2))

            # 3. Effective Attention Distance (sum((i - j) * A_ij))
            q_pos = torch.arange(seq_len, device=device).unsqueeze(1)
            k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
            D = torch.clamp(q_pos - k_pos, min=0).float()
            dist_per_head = (attn_w * D).sum(dim=-1).mean(dim=(0, 2))

            # 4. Diagonal Mass Ratio within distance K
            mask_K = (q_pos >= k_pos) & ((q_pos - k_pos) <= K)
            diag_per_head = (attn_w * mask_K).sum(dim=-1).mean(dim=(0, 2))

            for h in range(n_heads):
                layer_head_entropy[layer_idx][h] += entropy_per_head[h].item()
                layer_head_sink[layer_idx][h] += sink_per_head[h].item()
                layer_head_dist[layer_idx][h] += dist_per_head[h].item()
                layer_head_diag[layer_idx][h] += diag_per_head[h].item()

        n_samples += 1

    denom = max(n_samples, 1)

    def format_metric(raw_matrix, extra_meta=None):
        mat = [[raw_matrix[l][h] / denom for h in range(n_heads)] for l in range(n_layers)]
        per_layer = [round(sum(mat[l]) / n_heads, 4) for l in range(n_layers)]
        per_layer_per_head = [
            [round(mat[l][h], 4) for h in range(n_heads)]
            for l in range(n_layers)
        ]
        overall_mean = round(sum(per_layer) / n_layers, 4)
        res = {
            "mean": overall_mean,
            "per_layer": per_layer,
            "per_layer_per_head": per_layer_per_head,
        }
        if extra_meta:
            res.update(extra_meta)
        return res

    res = {
        "entropy": format_metric(layer_head_entropy),
        "sink_ratio": format_metric(layer_head_sink),
        "effective_distance": format_metric(layer_head_dist),
        "diagonal_mass_ratio": format_metric(layer_head_diag, {"K": K}),
    }
    return res


@torch.no_grad()
def compute_attention_entropy(
    model, dataloader, device, use_amp: bool = True, max_batches: int = 5
) -> float:
    """Compute average attention entropy (scalar) across all heads/layers."""
    return compute_attention_metrics_detailed(
        model, dataloader, device, use_amp=use_amp, max_batches=max_batches
    )["entropy"]["mean"]


@torch.no_grad()
def compute_attention_entropy_detailed(
    model, dataloader, device, use_amp: bool = True, max_batches: int = 5
) -> dict:
    """Compute attention entropy with per-layer and per-head breakdown."""
    return compute_attention_metrics_detailed(
        model, dataloader, device, use_amp=use_amp, max_batches=max_batches
    )["entropy"]

