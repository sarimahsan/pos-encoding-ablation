"""Unit tests for neural network components and forward passes."""

import torch
import pytest
from config import TransformerConfig
from components import (
    RMSNorm,
    RotaryEmbedding,
    ALiBiBias,
    Attention,
    SwiGLU_MLP,
    TransformerBlock,
    DecoderTransformer,
)


def test_rms_norm():
    norm = RMSNorm(dim=512)
    x = torch.randn(2, 16, 512)
    out = norm(x)
    assert out.shape == (2, 16, 512)


def test_rope_embedding():
    rope = RotaryEmbedding(head_dim=64, max_seq_len=256)
    q = torch.randn(2, 8, 32, 64)
    k = torch.randn(2, 8, 32, 64)
    q_rot, k_rot = rope(q, k, seq_len=32)
    assert q_rot.shape == (2, 8, 32, 64)
    assert k_rot.shape == (2, 8, 32, 64)


def test_alibi_bias():
    alibi = ALiBiBias(n_heads=8, max_seq_len=256)
    scores = torch.zeros(2, 8, 32, 32)
    scores_biased = alibi(scores, seq_len=32)
    assert scores_biased.shape == (2, 8, 32, 32)
    assert torch.allclose(scores_biased[0, 0, 0, 0], torch.tensor(0.0))
    assert (scores_biased[0, 0, 1, 0] < 0).item()


@pytest.mark.parametrize("pos_enc", ["rope", "alibi", "nope"])
def test_attention_modes(pos_enc):
    cfg = TransformerConfig(pos_encoding=pos_enc)
    attn = Attention(cfg)
    x = torch.randn(2, 16, 512)

    # Fast path (training): no attention weights materialized
    out_fast, weights_fast = attn(x, return_attn=False)
    assert out_fast.shape == (2, 16, 512)
    assert weights_fast is None

    # Slow path (eval): full attention weights returned
    out_slow, weights_slow = attn(x, return_attn=True)
    assert out_slow.shape == (2, 16, 512)
    assert weights_slow.shape == (2, 8, 16, 16)


def test_swiglu_mlp():
    cfg = TransformerConfig()
    mlp = SwiGLU_MLP(cfg)
    x = torch.randn(2, 16, 512)
    out = mlp(x)
    assert out.shape == (2, 16, 512)


@pytest.mark.parametrize("pos_enc", ["rope", "alibi", "nope"])
def test_decoder_transformer_param_count_and_forward(pos_enc):
    cfg = TransformerConfig(pos_encoding=pos_enc)
    model = DecoderTransformer(cfg)
    params = model.count_parameters()
    assert 50_000_000 <= params <= 55_000_000

    input_ids = torch.randint(0, 1000, (2, 16))
    targets = torch.randint(0, 1000, (2, 16))

    out = model(input_ids, targets, return_attn=True)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert len(out["attn_weights"]) == cfg.n_layers
    assert "loss" in out
    assert not torch.isnan(out["loss"])
