"""
Experiment matrix — R1 through R6.

6 runs: {RoPE, ALiBi, NoPE} × {seq_len 256, 512}
Each evaluated at train length + 768 + 1024 (extrapolation).
"""

from .model_config import TransformerConfig

# ── Experiment definitions ───────────────────────────────────────────
#   (pos_encoding, train_seq_len, eval_seq_lens, batch_size)
EXPERIMENT_MATRIX = {
    "R1": ("rope",  256, [256, 768, 1024], 128),
    "R2": ("rope",  512, [512, 768, 1024], 64),
    "R3": ("alibi", 256, [256, 768, 1024], 128),
    "R4": ("alibi", 512, [512, 768, 1024], 64),
    "R5": ("nope",  256, [256, 768, 1024], 128),
    "R6": ("nope",  512, [512, 768, 1024], 64),
}


def get_experiment_config(run_id: str) -> TransformerConfig:
    """
    Factory: return the exact config for experiment R1–R6.

    Batch size is halved for seq_len=512 to keep tokens/step constant.

    Args:
        run_id: One of R1, R2, R3, R4, R5, R6 (case-insensitive).

    Returns:
        TransformerConfig with all fields set for that experiment.
    """
    run_id = run_id.upper()
    if run_id not in EXPERIMENT_MATRIX:
        raise ValueError(
            f"Unknown run_id '{run_id}'. Must be one of: {list(EXPERIMENT_MATRIX.keys())}"
        )

    pos_enc, seq_len, eval_lens, bs = EXPERIMENT_MATRIX[run_id]
    return TransformerConfig(
        pos_encoding=pos_enc,
        train_seq_len=seq_len,
        eval_seq_lens=eval_lens,
        batch_size=bs,
        run_name=f"{run_id}_{pos_enc}_seq{seq_len}",
        output_dir=f"outputs/{run_id}_{pos_enc}_seq{seq_len}",
    )
