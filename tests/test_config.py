"""Unit tests for configuration & experiment matrix validation."""

import pytest
from config import TransformerConfig, get_experiment_config, EXPERIMENT_MATRIX


def test_default_config():
    cfg = TransformerConfig()
    assert cfg.vocab_size == 50257
    assert cfg.n_layers == 8
    assert cfg.d_model == 512
    assert cfg.n_heads == 8
    assert cfg.head_dim == 64
    assert cfg.pos_encoding == "rope"


def test_invalid_pos_encoding():
    with pytest.raises(AssertionError):
        TransformerConfig(pos_encoding="sinusoidal")


def test_invalid_head_dim():
    with pytest.raises(AssertionError):
        TransformerConfig(d_model=512, n_heads=8, head_dim=32)


def test_experiment_matrix_all_runs():
    for run_id in ["R1", "R2", "R3", "R4", "R5", "R6"]:
        cfg = get_experiment_config(run_id)
        assert cfg.run_name.startswith(run_id)
        assert cfg.pos_encoding in ("rope", "alibi", "nope")
        if cfg.train_seq_len == 256:
            assert cfg.batch_size == 128
        elif cfg.train_seq_len == 512:
            assert cfg.batch_size == 64


def test_invalid_experiment_run_id():
    with pytest.raises(ValueError):
        get_experiment_config("R7")
