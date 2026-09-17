"""Interface commune des encodeurs de mouvement + registre (P7).

Un ``MotionEncoder`` transforme une fenêtre (B, C, T, J) en embedding (B, D).
On peut ainsi remplacer le backbone (ST-GCN, multi-flux, CTR-GCN…) sans changer
le reste du modèle hiérarchique. Le squelette (adjacence) est passé explicitement
pour rester cohérent avec le schéma choisi.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kata_pipeline.gnn.config import ModelConfig
from kata_pipeline.gnn.models.stgcn import STGCNBlock
from kata_pipeline.gnn.skeletons import SkeletonSchema


class MotionEncoder(nn.Module):
    """Base : encode (B, C, T, J) -> (B, embedding_dim)."""

    embedding_dim: int

    def forward(self, x: torch.Tensor, return_node: bool = False):  # noqa: D401
        raise NotImplementedError


class STGCNMotionEncoder(MotionEncoder):
    """ST-GCN paramétré par un schéma de squelette arbitraire."""

    def __init__(self, config: ModelConfig, schema: SkeletonSchema, in_channels: int | None = None):
        super().__init__()
        adj = torch.from_numpy(schema.adjacency()).float()
        c_in = in_channels if in_channels is not None else config.in_channels
        c_hid = config.hidden_channels
        self.data_bn = nn.BatchNorm1d(c_in)
        channels = [c_in] + [c_hid] * (config.num_layers - 1) + [config.embedding_dim]
        blocks: list[nn.Module] = []
        for i in range(config.num_layers):
            stride = 2 if i in (1, 2) else 1
            blocks.append(
                STGCNBlock(channels[i], channels[i + 1], adj, stride=stride, dropout=config.dropout)
            )
        self.blocks = nn.ModuleList(blocks)
        self.embedding_dim = config.embedding_dim

    def forward(self, x: torch.Tensor, return_node: bool = False):
        b, c, t, j = x.shape
        x = x.permute(0, 1, 3, 2).reshape(b, c, j * t)
        x = self.data_bn(x)
        x = x.reshape(b, c, j, t).permute(0, 1, 3, 2).contiguous()
        for block in self.blocks:
            x = block(x)
        node_feat = x
        emb = x.mean(dim=(2, 3))
        if return_node:
            return emb, node_feat
        return emb


class MultiStreamSTGCNEncoder(MotionEncoder):
    """Deux branches ST-GCN (ex. articulations + os) fusionnées par concat.

    Les canaux d'entrée sont partagés entre branches selon ``split`` (nombre de
    canaux de la première branche). Utile pour combiner joint/joint_motion et
    bone/bone_motion sans les mélanger prématurément.
    """

    def __init__(self, config: ModelConfig, schema: SkeletonSchema, in_channels: int, split: int):
        super().__init__()
        assert 0 < split < in_channels, "split doit partager les canaux entre deux branches"
        self.split = split
        half = config.embedding_dim // 2
        cfg_a = config.model_copy(update={"embedding_dim": half})
        cfg_b = config.model_copy(update={"embedding_dim": config.embedding_dim - half})
        self.branch_a = STGCNMotionEncoder(cfg_a, schema, in_channels=split)
        self.branch_b = STGCNMotionEncoder(cfg_b, schema, in_channels=in_channels - split)
        self.embedding_dim = config.embedding_dim

    def forward(self, x: torch.Tensor, return_node: bool = False):
        xa = x[:, : self.split]
        xb = x[:, self.split :]
        if return_node:
            ea, na = self.branch_a(xa, return_node=True)
            eb, nb = self.branch_b(xb, return_node=True)
            return torch.cat([ea, eb], dim=1), (na, nb)
        ea = self.branch_a(xa)
        eb = self.branch_b(xb)
        return torch.cat([ea, eb], dim=1)


class MotionBERTStub(MotionEncoder):
    """Stub d'un backbone pré-entraîné type MotionBERT (P8).

    Interface conforme mais non fonctionnelle sans poids ni mapping de squelette
    validé. On refuse explicitement un squelette incompatible plutôt que de
    laisser passer silencieusement.
    """

    def __init__(self, config: ModelConfig, schema: SkeletonSchema, in_channels: int):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        # MotionBERT attend un squelette H36M (17 joints). On vérifie le mapping.
        if schema.num_joints not in (17,):
            raise NotImplementedError(
                "MotionBERTStub : mapping de squelette non fourni "
                f"(schéma '{schema.name}', {schema.num_joints} joints ; attendu 17 H36M). "
                "Fournir un mapping explicite avant utilisation."
            )

    def forward(self, x: torch.Tensor, return_node: bool = False):  # pragma: no cover - stub
        raise NotImplementedError("MotionBERTStub : poids pré-entraînés non fournis.")


_ENCODERS = {"stgcn", "multistream_stgcn", "motionbert"}


def build_encoder(config: ModelConfig, schema: SkeletonSchema, in_channels: int,
                  bone_split: int | None = None) -> MotionEncoder:
    """Fabrique l'encodeur demandé (``model.encoder``)."""

    name = (config.encoder or "stgcn").lower()
    if name == "stgcn":
        return STGCNMotionEncoder(config, schema, in_channels=in_channels)
    if name == "multistream_stgcn":
        split = bone_split if bone_split is not None else in_channels // 2
        return MultiStreamSTGCNEncoder(config, schema, in_channels=in_channels, split=split)
    if name == "motionbert":
        return MotionBERTStub(config, schema, in_channels=in_channels)
    raise ValueError(f"Encodeur inconnu : '{config.encoder}'. Disponibles : {sorted(_ENCODERS)}.")
