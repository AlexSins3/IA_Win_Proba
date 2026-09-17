"""Modèle hiérarchique de prestation et comparateur antisymétrique (P4/P5).

Deux niveaux :
1. ``KataScorer`` : encode chaque fenêtre puis agrège -> score de qualité scalaire
   q(prestation) et embedding de prestation.
2. ``KataOutcomeComparator`` : compare deux prestations A et B. La sortie est
   **antisymétrique par construction** : ``logit = q(B) - q(A)`` (+ éventuel
   résidu contextuel antisymétrique), ce qui garantit
   ``P(A>B) = 1 - P(B>A)`` et une prédiction cohérente à l'échange A<->B.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kata_pipeline.gnn.config import ModelConfig
from kata_pipeline.gnn.models.aggregators import build_aggregator
from kata_pipeline.gnn.models.encoders import build_encoder
from kata_pipeline.gnn.skeletons import SkeletonSchema


class KataScorer(nn.Module):
    """Encodeur de fenêtres + agrégateur + tête de qualité scalaire.

    Entrée : ``x`` (B, Nw, C, T, J) et ``mask`` (B, Nw).
    Sortie : qualité (B,) et embedding de prestation (B, D).
    """

    def __init__(self, config: ModelConfig, schema: SkeletonSchema, in_channels: int,
                 bone_split: int | None = None):
        super().__init__()
        self.encoder = build_encoder(config, schema, in_channels, bone_split=bone_split)
        self.aggregator = build_aggregator(config.aggregator, self.encoder.embedding_dim)
        self.quality_head = nn.Sequential(
            nn.Linear(self.encoder.embedding_dim, self.encoder.embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(self.encoder.embedding_dim // 2, 1),
        )
        self.embedding_dim = self.encoder.embedding_dim

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, nw, c, t, j = x.shape
        flat = x.reshape(b * nw, c, t, j)
        emb = self.encoder(flat)  # (B*Nw, D)
        emb = emb.reshape(b, nw, -1)
        return self.aggregator(emb, mask)  # (B, D)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        perf_emb = self.encode(x, mask)
        quality = self.quality_head(perf_emb).squeeze(-1)  # (B,)
        return quality, perf_emb


class KataOutcomeComparator(nn.Module):
    """Comparateur antisymétrique de deux prestations."""

    def __init__(self, config: ModelConfig, schema: SkeletonSchema, in_channels: int,
                 bone_split: int | None = None):
        super().__init__()
        self.scorer = KataScorer(config, schema, in_channels, bone_split=bone_split)
        self.use_context = config.context_residual
        if self.use_context:
            d = self.scorer.embedding_dim
            self.context = nn.Sequential(
                nn.Linear(2 * d, d),
                nn.ReLU(inplace=True),
                nn.Linear(d, 1),
            )

    def _context_residual(self, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        # Antisymétrique : f(a,b) - f(b,a).
        fab = self.context(torch.cat([ea, eb], dim=1)).squeeze(-1)
        fba = self.context(torch.cat([eb, ea], dim=1)).squeeze(-1)
        return fab - fba

    def forward(self, xa, mask_a, xb, mask_b):
        qa, ea = self.scorer(xa, mask_a)
        qb, eb = self.scorer(xb, mask_b)
        logit = qb - qa  # P(B gagne) via sigmoid ; antisymétrique par construction
        if self.use_context:
            logit = logit + self._context_residual(ea, eb)
        return {
            "logit": logit,          # (B,) score en faveur de B
            "quality_a": qa,
            "quality_b": qb,
        }
