"""Têtes de tâche montées sur l'encodeur ST-GCN.

- ``KataQualityNet`` : embedding -> marge de drapeaux + score technique.
- ``KataComparator`` : encodeur siamois partagé -> P(A bat B), *anti-symétrique*.

L'anti-symétrie garantit que swapper (A, B) inverse la prédiction : c'est
essentiel pour un juge équitable indépendant de l'ordre de présentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kata_pipeline.gnn.config import ModelConfig
from kata_pipeline.gnn.models.stgcn import STGCNEncoder


class KataQualityNet(nn.Module):
    """Prédit la qualité d'un kata isolé (marge de drapeaux + score)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = STGCNEncoder(config)
        d = config.embedding_dim
        self.head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )
        self.flag_out = nn.Linear(d // 2, 1)  # marge normalisée [0,1] (sigmoïde)
        self.score_out = nn.Linear(d // 2, 1)  # score technique brut

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        emb = self.encoder(x)
        h = self.head(emb)
        return {
            "embedding": emb,
            "flag": torch.sigmoid(self.flag_out(h)).squeeze(-1),
            "score": self.score_out(h).squeeze(-1),
        }


class KataComparator(nn.Module):
    """Comparateur siamois anti-symétrique : logit(A>B) = g(f(A)) - g(f(B)).

    ``f`` = encodeur ST-GCN partagé, ``g`` = tête scalaire de "qualité latente".
    Le vainqueur est celui dont la qualité latente est la plus élevée. Comme la
    prédiction ne dépend que de la *différence*, l'identité et l'ordre n'ont
    aucune influence.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = STGCNEncoder(config)
        d = config.embedding_dim
        self.quality = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(d // 2, 1),
        )

    def latent_quality(self, x: torch.Tensor) -> torch.Tensor:
        return self.quality(self.encoder(x)).squeeze(-1)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> dict[str, torch.Tensor]:
        qa = self.latent_quality(xa)
        qb = self.latent_quality(xb)
        # logit > 0  => B est jugé meilleur (label 1 = B gagne).
        logit = qb - qa
        return {"logit": logit, "quality_a": qa, "quality_b": qb}
