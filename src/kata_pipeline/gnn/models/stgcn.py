"""ST-GCN : Spatial-Temporal Graph Convolutional Network (PyTorch pur).

Implémentation légère et autonome (pas de torch-geometric, pénible à installer
sous Windows). L'entrée est un tenseur (B, C, T, J) :
    B = batch, C = canaux/features, T = temps, J = articulations.

Chaque bloc applique :
- une convolution de graphe spatiale (mélange l'info entre articulations voisines
  via la matrice d'adjacence du squelette),
- une convolution temporelle 1D (capture la dynamique du geste),
- BatchNorm + ReLU + résiduel.

La sortie est un *embedding* de kata (vecteur), consommé par les têtes de tâche.
Réf. Yan et al., "Spatial Temporal Graph Convolutional Networks", AAAI 2018.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from kata_pipeline.gnn.config import ModelConfig
from kata_pipeline.gnn.graph.skeleton import build_adjacency


class GraphConv(nn.Module):
    """Convolution spatiale sur les articulations : X' = A_hat @ X @ W."""

    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor):
        super().__init__()
        # Adjacence fixe (squelette) enregistrée comme buffer non entraînable.
        self.register_buffer("adj", adjacency)
        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, T, J)
        x = self.linear(x)  # (B, C_out, T, J)
        # Mélange spatial : contraction sur la dimension articulations.
        x = torch.einsum("bctj,jk->bctk", x, self.adj)
        return x


class STGCNBlock(nn.Module):
    """Bloc spatio-temporel : graphe (spatial) + conv temporelle."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        temporal_kernel: int = 9,
        stride: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, adjacency)
        pad = (temporal_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel, 1),
                stride=(stride, 1),
                padding=(pad, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=(stride, 1)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.relu(x + res)


class STGCNEncoder(nn.Module):
    """Empile plusieurs blocs ST-GCN et produit un embedding de kata."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        adj = torch.from_numpy(build_adjacency()).float()

        c_in = config.in_channels
        c_hid = config.hidden_channels
        self.data_bn = nn.BatchNorm1d(c_in)

        blocks: list[nn.Module] = []
        channels = [c_in] + [c_hid] * (config.num_layers - 1) + [config.embedding_dim]
        for i in range(config.num_layers):
            stride = 2 if i in (1, 2) else 1  # sous-échantillonnage temporel léger
            blocks.append(
                STGCNBlock(
                    channels[i],
                    channels[i + 1],
                    adj,
                    stride=stride,
                    dropout=config.dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.embedding_dim = config.embedding_dim

    def forward(self, x: torch.Tensor, return_node: bool = False):
        # x : (B, C, T, J)
        b, c, t, j = x.shape
        # BatchNorm sur les canaux, appliquée frame/joint par frame/joint.
        x = x.permute(0, 1, 3, 2).reshape(b, c, j * t)
        x = self.data_bn(x)
        x = x.reshape(b, c, j, t).permute(0, 1, 3, 2).contiguous()

        for block in self.blocks:
            x = block(x)  # (B, C_emb, T', J)

        node_feat = x  # activations spatio-temporelles (pour la saillance)
        # Pooling global spatio-temporel -> embedding (B, C_emb).
        emb = x.mean(dim=(2, 3))
        if return_node:
            return emb, node_feat
        return emb
