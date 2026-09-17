"""Dataset hiérarchique de paires de prestations (P14).

Contrairement à ``MatchPairDataset`` (une fenêtre par kata), celui-ci renvoie
**toutes** les fenêtres d'une prestation, plus un masque de padding, pour
alimenter le modèle hiérarchique ``KataOutcomeComparator``.

Chaîne par échantillon :
    PoseSequence (cache .npz) -> [augment train] -> preprocess -> fenêtres (Nw,C,T,J).

Le ``collate`` pad les prestations à un nombre commun de fenêtres et fournit les
masques. L'ordre A/B est tiré aléatoirement en entraînement (label = index du
gagnant) pour interdire tout biais de position.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from kata_pipeline.gnn.augment import AugmentConfig, augment_pose
from kata_pipeline.gnn.config import GNNConfig, PreprocessConfig
from kata_pipeline.gnn.pose.sequence import PoseSequence
from kata_pipeline.gnn.preprocess import feature_channels, preprocess_sequence, sliding_windows
from kata_pipeline.gnn.skeletons import SkeletonSchema, get_schema

logger = logging.getLogger(__name__)


def compute_in_channels(cfg: PreprocessConfig) -> int:
    """Nombre de canaux d'entrée déduit du prétraitement (streams/options)."""

    return len(feature_channels(cfg))


def _windows_from_path(path: str, cfg: GNNConfig, schema: SkeletonSchema,
                       augment: AugmentConfig | None, rng: np.random.Generator) -> np.ndarray:
    seq = PoseSequence.load(path)
    if augment is not None:
        seq = augment_pose(seq, augment, schema, rng)
    pre = preprocess_sequence(seq, cfg.preprocess, schema)
    return sliding_windows(pre, cfg.preprocess)  # (Nw, C, T, J)


class PerformancePairDataset(Dataset):
    """Paires (gagnant, perdant) -> toutes les fenêtres + masques."""

    def __init__(self, manifest: pd.DataFrame, config: GNNConfig, train: bool = True,
                 augment: AugmentConfig | None = None):
        self.config = config
        self.train = train
        self.schema = get_schema(config.skeleton.schema_name)
        self.augment = augment if (train and augment and augment.enabled) else None
        self.rng = np.random.default_rng(config.train.seed + 7)
        self._val_cache: dict[str, np.ndarray] = {}

        pairs: list[tuple[str, str]] = []
        for _, grp in manifest.groupby("id_match"):
            grp = grp[grp["pose_path"].apply(lambda p: Path(p).exists())]
            if len(grp) != 2:
                continue
            win = grp[grp["issue"] == "win"]
            los = grp[grp["issue"] == "loss"]
            if len(win) == 1 and len(los) == 1:
                pairs.append((win.iloc[0]["pose_path"], los.iloc[0]["pose_path"]))
        self.pairs = pairs
        logger.info("PerformancePairDataset : %d paires (%s).",
                    len(pairs), "train" if train else "val")

    def __len__(self) -> int:
        return len(self.pairs)

    def _windows(self, path: str) -> np.ndarray:
        if not self.train:
            if path not in self._val_cache:
                self._val_cache[path] = _windows_from_path(
                    path, self.config, self.schema, None, self.rng
                )
            return self._val_cache[path]
        return _windows_from_path(path, self.config, self.schema, self.augment, self.rng)

    def __getitem__(self, idx: int):
        winner_path, loser_path = self.pairs[idx]
        wx = self._windows(winner_path)
        lx = self._windows(loser_path)
        if self.train:
            swap = self.rng.random() < 0.5
        else:
            swap = (idx % 2 == 1)  # déterministe en val : ~50 % label-0, ~50 % label-1
        if swap:
            xa, xb, label = lx, wx, 1.0  # A=perdant, B=gagnant
        else:
            xa, xb, label = wx, lx, 0.0  # A=gagnant
        return {
            "xa": torch.from_numpy(xa).float(),
            "xb": torch.from_numpy(xb).float(),
            "label": torch.tensor(label, dtype=torch.float32),
        }


def collate_pairs(batch: list[dict], max_windows: int | None = None) -> dict:
    """Pad chaque prestation au même nombre de fenêtres et construit les masques."""

    def pad(side: str):
        tensors = [b[side] for b in batch]
        nw = max(t.shape[0] for t in tensors)
        if max_windows is not None:
            nw = min(nw, max_windows)
        c, t, j = tensors[0].shape[1:]
        out = torch.zeros(len(tensors), nw, c, t, j)
        mask = torch.zeros(len(tensors), nw)
        for i, ten in enumerate(tensors):
            k = min(ten.shape[0], nw)
            out[i, :k] = ten[:k]
            mask[i, :k] = 1.0
        return out, mask

    xa, mask_a = pad("xa")
    xb, mask_b = pad("xb")
    labels = torch.stack([b["label"] for b in batch])
    return {"xa": xa, "mask_a": mask_a, "xb": xb, "mask_b": mask_b, "label": labels}
