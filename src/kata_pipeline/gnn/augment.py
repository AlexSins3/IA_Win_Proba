"""Augmentations pose-safe pour l'entraînement (P15).

Toutes opèrent sur un ``PoseSequence`` (coordonnées normalisées image) et
préservent la cohérence des masques. Elles sont pensées pour le kata :

- ``spatial_jitter``    : bruit gaussien léger (robustesse au bruit d'extraction) ;
- ``joint_dropout``     : occlusion aléatoire d'articulations (simule des pertes) ;
- ``temporal_crop``     : recadrage temporel (invariance au timing global) ;
- ``horizontal_flip``   : miroir gauche/droite via la symétrie du schéma
  (un kata exécuté en miroir reste un kata valide) ;

L'échange A/B est géré au niveau du dataset de paires (obligatoire, pas ici).
Toutes sont désactivables et ne s'appliquent qu'en entraînement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kata_pipeline.gnn.pose.sequence import PoseSequence
from kata_pipeline.gnn.skeletons import SkeletonSchema


@dataclass
class AugmentConfig:
    enabled: bool = True
    spatial_jitter_std: float = 0.01
    joint_dropout_prob: float = 0.05
    temporal_crop_ratio: float = 0.1  # fraction max rognée de chaque côté
    horizontal_flip_prob: float = 0.5


def _replace(seq: PoseSequence, **kw) -> PoseSequence:
    base = dict(
        keypoints=seq.keypoints, confidence=seq.confidence, valid_mask=seq.valid_mask,
        interpolated_mask=seq.interpolated_mask, timestamps=seq.timestamps,
        fps=seq.fps, joint_names=seq.joint_names, metadata=seq.metadata,
    )
    base.update(kw)
    return PoseSequence(**base)


def spatial_jitter(seq: PoseSequence, std: float, rng: np.random.Generator) -> PoseSequence:
    if std <= 0:
        return seq
    noise = rng.normal(0.0, std, size=seq.keypoints.shape).astype(np.float32)
    noise *= seq.valid_mask[:, :, None]  # ne bruite pas les points absents
    return _replace(seq, keypoints=(seq.keypoints + noise).astype(np.float32))


def joint_dropout(seq: PoseSequence, prob: float, rng: np.random.Generator) -> PoseSequence:
    if prob <= 0:
        return seq
    T, J = seq.valid_mask.shape
    drop = rng.random((T, J)) < prob
    valid = seq.valid_mask & ~drop
    kp = seq.keypoints.copy()
    kp[drop] = 0.0
    return _replace(seq, keypoints=kp, valid_mask=valid)


def temporal_crop(seq: PoseSequence, max_ratio: float, rng: np.random.Generator) -> PoseSequence:
    T = seq.num_frames
    if max_ratio <= 0 or T < 8:
        return seq
    max_cut = int(T * max_ratio)
    if max_cut < 1:
        return seq
    left = int(rng.integers(0, max_cut + 1))
    right = int(rng.integers(0, max_cut + 1))
    hi = T - right
    if hi - left < T // 2:
        return seq
    sl = slice(left, hi)
    return _replace(
        seq,
        keypoints=seq.keypoints[sl],
        confidence=seq.confidence[sl],
        valid_mask=seq.valid_mask[sl],
        interpolated_mask=seq.interpolated_mask[sl],
        timestamps=seq.timestamps[sl],
    )


def horizontal_flip(seq: PoseSequence, schema: SkeletonSchema) -> PoseSequence:
    kp = seq.keypoints.copy()
    conf = seq.confidence.copy()
    valid = seq.valid_mask.copy()
    interp = seq.interpolated_mask.copy()
    # Miroir de l'axe x (centré à 0 après normalisation ; sinon 0.5 - x conviendrait).
    kp[:, :, 0] = -kp[:, :, 0]
    for a, b in schema.symmetry:
        kp[:, [a, b]] = kp[:, [b, a]]
        conf[:, [a, b]] = conf[:, [b, a]]
        valid[:, [a, b]] = valid[:, [b, a]]
        interp[:, [a, b]] = interp[:, [b, a]]
    return _replace(seq, keypoints=kp, confidence=conf, valid_mask=valid, interpolated_mask=interp)


def augment_pose(seq: PoseSequence, cfg: AugmentConfig, schema: SkeletonSchema,
                 rng: np.random.Generator) -> PoseSequence:
    """Applique la chaîne d'augmentations (train uniquement)."""

    if not cfg.enabled:
        return seq
    seq = spatial_jitter(seq, cfg.spatial_jitter_std, rng)
    seq = joint_dropout(seq, cfg.joint_dropout_prob, rng)
    seq = temporal_crop(seq, cfg.temporal_crop_ratio, rng)
    if rng.random() < cfg.horizontal_flip_prob:
        seq = horizontal_flip(seq, schema)
    return seq
