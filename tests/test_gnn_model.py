"""Tests des modèles ST-GCN et de l'anti-symétrie du comparateur (phase 2 / GNN)."""

from __future__ import annotations

import torch

from kata_pipeline.gnn.config import ModelConfig
from kata_pipeline.gnn.graph.skeleton import NUM_JOINTS
from kata_pipeline.gnn.models.heads import KataComparator, KataQualityNet
from kata_pipeline.gnn.models.stgcn import STGCNEncoder


def _config() -> ModelConfig:
    # Config réduite pour un test rapide, mêmes canaux d'entrée que la prod.
    return ModelConfig(hidden_channels=16, embedding_dim=32, num_layers=3)


def test_encoder_output_shape() -> None:
    torch.manual_seed(0)
    cfg = _config()
    enc = STGCNEncoder(cfg).eval()
    x = torch.randn(2, cfg.in_channels, 120, NUM_JOINTS)
    emb = enc(x)
    assert emb.shape == (2, cfg.embedding_dim)


def test_quality_net_outputs() -> None:
    torch.manual_seed(0)
    cfg = _config()
    net = KataQualityNet(cfg).eval()
    x = torch.randn(2, cfg.in_channels, 120, NUM_JOINTS)
    out = net(x)
    assert out["flag"].shape == (2,)
    assert out["score"].shape == (2,)
    # La marge de drapeaux passe par une sigmoïde -> [0, 1].
    assert torch.all((out["flag"] >= 0) & (out["flag"] <= 1))


def test_comparator_is_antisymmetric() -> None:
    torch.manual_seed(0)
    cfg = _config()
    model = KataComparator(cfg).eval()  # eval -> BatchNorm déterministe
    xa = torch.randn(3, cfg.in_channels, 120, NUM_JOINTS)
    xb = torch.randn(3, cfg.in_channels, 120, NUM_JOINTS)
    logit_ab = model(xa, xb)["logit"]
    logit_ba = model(xb, xa)["logit"]
    # logit(A,B) = q(B) - q(A) = -(q(A) - q(B)) = -logit(B,A).
    assert torch.allclose(logit_ab, -logit_ba, atol=1e-4)


def test_comparator_probabilities_sum_to_one() -> None:
    torch.manual_seed(0)
    cfg = _config()
    model = KataComparator(cfg).eval()
    xa = torch.randn(4, cfg.in_channels, 120, NUM_JOINTS)
    xb = torch.randn(4, cfg.in_channels, 120, NUM_JOINTS)
    p_ab = torch.sigmoid(model(xa, xb)["logit"])
    p_ba = torch.sigmoid(model(xb, xa)["logit"])
    assert torch.allclose(p_ab + p_ba, torch.ones_like(p_ab), atol=1e-4)
