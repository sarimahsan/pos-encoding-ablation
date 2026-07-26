"""
Attention heatmap generation — PNG images + raw attention tensors.

Saves both a human-readable PNG and the raw .pt tensor for offline
analysis and reproduction without reloading the checkpoint.
"""

import os
import torch
from torch.amp import autocast


@torch.no_grad()
def generate_attention_heatmap(
    model, dataloader, config, device, use_amp: bool = True,
    layer_idx: int = 0, head_idx: int = 0, max_tokens: int = 64,
    save_dir: str = None,
):
    """
    Generate attention heatmap PNG + save raw attention weight tensor (.pt).

    Args:
        model: trained transformer
        dataloader: eval dataloader
        config: TransformerConfig
        device: cuda/cpu
        use_amp: fp16 autocast
        layer_idx: which layer to visualize
        head_idx: which attention head
        max_tokens: cap on sequence length for the heatmap
        save_dir: directory to save outputs
    """
    if save_dir is None:
        save_dir = os.path.join(config.output_dir, "heatmaps")
    os.makedirs(save_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping heatmap generation.")
        return

    model.eval()
    input_ids, targets = next(iter(dataloader))
    input_ids = input_ids[:1, :max_tokens].to(device)
    targets = targets[:1, :max_tokens].to(device)
    seq_len = input_ids.shape[1]

    with autocast("cuda", enabled=use_amp):
        output = model(input_ids, targets, return_attn=True)

    attn = output["attn_weights"][layer_idx][0, head_idx].float().cpu()
    attn_np = attn[:max_tokens, :max_tokens].numpy()

    # Save raw attention tensor
    raw_path = os.path.join(save_dir, f"attn_raw_L{layer_idx}_H{head_idx}_seq{seq_len}.pt")
    torch.save(
        {
            "attention_weights": attn[:max_tokens, :max_tokens],
            "input_ids": input_ids[0].cpu(),
            "layer_idx": layer_idx,
            "head_idx": head_idx,
            "seq_len": seq_len,
            "pos_encoding": config.pos_encoding,
            "run_name": config.run_name,
        },
        raw_path,
    )

    # Save PNG heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn_np, cmap="viridis", aspect="auto")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(
        f"Attention Heatmap - {config.pos_encoding.upper()} | "
        f"Layer {layer_idx}, Head {head_idx} | seq_len={seq_len}"
    )
    fig.colorbar(im, ax=ax, shrink=0.8)

    png_path = os.path.join(save_dir, f"attn_heatmap_L{layer_idx}_H{head_idx}_seq{seq_len}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Heatmap saved: {png_path}")
    print(f"  Raw attn data: {raw_path}")
