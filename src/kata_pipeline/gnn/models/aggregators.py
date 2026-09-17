"""Agrégateurs d'embeddings de fenêtres -> embedding de prestation (P5).

Une prestation de kata est découpée en Nw fenêtres. Chaque fenêtre est encodée
puis les embeddings sont agrégés en un vecteur unique. On expose deux stratégies :

- ``MeanAggregator``      : moyenne masquée (baseline robuste, peu de paramètres) ;
- ``AttentionAggregator`` : pondération apprise par fenêtre (met en avant les
  moments décisifs).

Les deux reçoivent un masque de padding (fenêtres factices ignorées).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WindowAggregator(nn.Module):
    """Base : (B, Nw, D) + mask (B, Nw) -> (B, D)."""

    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError


class MeanAggregator(WindowAggregator):
    """Moyenne des fenêtres valides."""

    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float().unsqueeze(-1)  # (B, Nw, 1)
        summed = (emb * m).sum(dim=1)
        count = m.sum(dim=1).clamp_min(1.0)
        return summed / count


class AttentionAggregator(WindowAggregator):
    """Attention additive : score par fenêtre, softmax masqué."""

    def __init__(self, embedding_dim: int, hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(emb).squeeze(-1)  # (B, Nw)
        scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, Nw, 1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return (emb * weights).sum(dim=1)

    def attention_weights(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(emb).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), float("-inf"))
        return torch.nan_to_num(torch.softmax(scores, dim=1), nan=0.0)


def build_aggregator(name: str, embedding_dim: int) -> WindowAggregator:
    name = (name or "mean").lower()
    if name == "mean":
        return MeanAggregator()
    if name == "attention":
        return AttentionAggregator(embedding_dim)
    raise ValueError(f"Agrégateur inconnu : '{name}'. Disponibles : mean, attention.")
