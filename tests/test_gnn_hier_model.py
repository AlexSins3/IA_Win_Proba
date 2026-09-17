"""Tests du modèle hiérarchique, agrégateurs et comparateur d'issue (P4/P5/P7)."""

from __future__ import annotations

import torch

from kata_pipeline.gnn.config import ModelConfig
from kata_pipeline.gnn.models.aggregators import AttentionAggregator, MeanAggregator
from kata_pipeline.gnn.models.kata_model import KataOutcomeComparator, KataScorer
from kata_pipeline.gnn.skeletons import get_schema

SCHEMA = get_schema("mediapipe_15")
IN_CH = 8
T = 20
J = 15


def _cfg(**kw):
    base = dict(hidden_channels=16, embedding_dim=32, num_layers=3, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


def _windows(b, nw):
    return torch.randn(b, nw, IN_CH, T, J)


def test_mean_aggregator_ignores_padding():
    agg = MeanAggregator()
    emb = torch.randn(2, 4, 8)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    out = agg(emb, mask)
    assert torch.allclose(out[0], emb[0, :2].mean(0), atol=1e-5)
    assert torch.allclose(out[1], emb[1, 0], atol=1e-5)


def test_attention_aggregator_shapes():
    agg = AttentionAggregator(8)
    emb = torch.randn(3, 5, 8)
    mask = torch.ones(3, 5)
    out = agg(emb, mask)
    assert out.shape == (3, 8)
    w = agg.attention_weights(emb, mask)
    assert torch.allclose(w.sum(1), torch.ones(3), atol=1e-5)


def test_scorer_output_shapes():
    model = KataScorer(_cfg(), SCHEMA, IN_CH).eval()
    x = _windows(2, 3)
    mask = torch.ones(2, 3)
    q, emb = model(x, mask)
    assert q.shape == (2,)
    assert emb.shape == (2, 32)


def test_comparator_antisymmetry_without_context():
    model = KataOutcomeComparator(_cfg(context_residual=False), SCHEMA, IN_CH).eval()
    xa, xb = _windows(2, 3), _windows(2, 4)
    ma, mb = torch.ones(2, 3), torch.ones(2, 4)
    with torch.no_grad():
        l_ab = model(xa, ma, xb, mb)["logit"]
        l_ba = model(xb, mb, xa, ma)["logit"]
    # logit = qb - qa  =>  antisymétrie exacte.
    assert torch.allclose(l_ab, -l_ba, atol=1e-4)


def test_comparator_probabilities_sum_to_one():
    model = KataOutcomeComparator(_cfg(), SCHEMA, IN_CH).eval()
    xa, xb = _windows(1, 2), _windows(1, 2)
    ma, mb = torch.ones(1, 2), torch.ones(1, 2)
    with torch.no_grad():
        p_b = torch.sigmoid(model(xa, ma, xb, mb)["logit"])
        p_a = torch.sigmoid(model(xb, mb, xa, ma)["logit"])
    assert torch.allclose(p_a + p_b, torch.ones(1), atol=1e-4)


def test_multistream_encoder_runs():
    model = KataScorer(_cfg(encoder="multistream_stgcn"), SCHEMA, IN_CH).eval()
    x = _windows(2, 2)
    mask = torch.ones(2, 2)
    q, emb = model(x, mask)
    assert q.shape == (2,)
