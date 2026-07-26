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
def compute_attention_entropy(
    model, dataloader, device, use_amp: bool = True, max_batches: int = 5
) -> float:
    """
    Compute average attention entropy (scalar) across all heads/layers.

    Entropy = -sum(p * log(p)), averaged over everything.
    """
    model.eval()
    total_entropy = 0.0
    n_samples = 0

    for i, (input_ids, targets) in enumerate(dataloader):
        if i >= max_batches:
            break

        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with autocast("cuda", enabled=use_amp):
            output = model(input_ids, targets)

        for attn_w in output["attn_weights"]:
            attn_w = attn_w.float().clamp(min=1e-8)
            entropy = -(attn_w * attn_w.log()).sum(dim=-1)
            total_entropy += entropy.mean().item()
            n_samples += 1

    return total_entropy / max(n_samples, 1)


@torch.no_grad()
def compute_attention_entropy_detailed(
    model, dataloader, device, use_amp: bool = True, max_batches: int = 5
) -> dict:
    """
    Compute attention entropy with per-layer and per-head breakdown.

    Returns:
        {
            "mean": float,
            "per_layer": [float, ...],
            "per_layer_per_head": [[float, ...], ...]
        }
    """
    model.eval()
    n_layers = len(model.blocks)
    n_heads = model.config.n_heads

    layer_head_entropy = [[0.0] * n_heads for _ in range(n_layers)]
    n_samples = 0

    for i, (input_ids, targets) in enumerate(dataloader):
        if i >= max_batches:
            break

        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with autocast("cuda", enabled=use_amp):
            output = model(input_ids, targets)

        for layer_idx, attn_w in enumerate(output["attn_weights"]):
            attn_w = attn_w.float().clamp(min=1e-8)
            entropy = -(attn_w * attn_w.log()).sum(dim=-1)
            per_head = entropy.mean(dim=(0, 2))
            for h in range(n_heads):
                layer_head_entropy[layer_idx][h] += per_head[h].item()

        n_samples += 1

    # Average
    for l in range(n_layers):
        for h in range(n_heads):
            layer_head_entropy[l][h] /= max(n_samples, 1)

    per_layer = [round(sum(layer_head_entropy[l]) / n_heads, 4) for l in range(n_layers)]
    per_layer_per_head = [
        [round(layer_head_entropy[l][h], 4) for h in range(n_heads)]
        for l in range(n_layers)
    ]
    overall_mean = sum(per_layer) / n_layers

    return {
        "mean": round(overall_mean, 4),
        "per_layer": per_layer,
        "per_layer_per_head": per_layer_per_head,
    }
