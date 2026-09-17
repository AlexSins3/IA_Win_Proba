"""Prétraitement robuste et features biomécaniques (P2/P3).

Chaîne : ``PoseSequence`` brut → interpolation contrôlée par la confiance et la
durée d'occlusion → normalisation → features multi-flux → fenêtrage temporel
défini en secondes.

Choix clés (issus de l'audit) :
- l'interpolation ne comble que les trous courts (``max_interp_gap_seconds``) et
  conserve ``valid_mask`` / ``interpolated_mask`` pour que le modèle sache ce qui
  est reconstruit ;
- les dérivées (vitesse/accélération) utilisent les ``timestamps`` réels via
  ``np.gradient`` (dt non supposé constant) ;
- les fenêtres sont définies en secondes puis converties en frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kata_pipeline.gnn.config import PreprocessConfig
from kata_pipeline.gnn.pose.sequence import PoseSequence
from kata_pipeline.gnn.skeletons import SkeletonSchema


@dataclass
class PreprocessedSequence:
    """Résultat du prétraitement, prêt pour le fenêtrage."""

    features: np.ndarray      # (T, J, C)
    timestamps: np.ndarray    # (T,)
    valid_mask: np.ndarray    # (T, J)
    interpolated_mask: np.ndarray  # (T, J)
    fps: float
    channel_names: list[str]


# --------------------------------------------------------------------------- #
# 1. Interpolation contrôlée.
# --------------------------------------------------------------------------- #
def clean_and_interpolate(seq: PoseSequence, cfg: PreprocessConfig) -> PoseSequence:
    """Marque invalides les points sous le seuil de confiance puis comble les
    trous *courts* uniquement (durée <= ``max_interp_gap_seconds``)."""

    kp = seq.keypoints.copy()
    conf = seq.confidence
    t = seq.timestamps
    valid = seq.valid_mask & (conf >= cfg.confidence_threshold)
    interpolated = np.zeros_like(valid)

    T, J, D = kp.shape
    for j in range(J):
        vj = valid[:, j]
        if vj.all():
            continue
        idx = np.where(vj)[0]
        if idx.size == 0:
            # Articulation jamais vue : reste à zéro, entièrement masquée.
            kp[:, j, :] = 0.0
            continue
        # Comble chaque trou interne si sa durée est courte.
        for a, b in _gaps(vj):
            # a..b sont des indices invalides ; bornes valides = a-1, b+1.
            left, right = a - 1, b + 1
            if left < 0 or right >= T:
                continue  # trou en bord : pas d'extrapolation.
            gap_dur = float(t[right] - t[left])
            if gap_dur > cfg.max_interp_gap_seconds:
                continue  # occlusion trop longue : on laisse masqué.
            for d in range(D):
                kp[a:b + 1, j, d] = np.interp(
                    t[a:b + 1], [t[left], t[right]], [kp[left, j, d], kp[right, j, d]]
                )
            interpolated[a:b + 1, j] = True

    valid = valid | interpolated
    return PoseSequence(
        keypoints=kp,
        confidence=conf,
        valid_mask=valid,
        interpolated_mask=interpolated,
        timestamps=t,
        fps=seq.fps,
        joint_names=seq.joint_names,
        metadata={**seq.metadata, "interpolated": True},
    )


def _gaps(valid_1d: np.ndarray) -> list[tuple[int, int]]:
    """Retourne les intervalles [start, end] d'indices invalides consécutifs."""

    gaps: list[tuple[int, int]] = []
    start = None
    for i, ok in enumerate(valid_1d):
        if not ok and start is None:
            start = i
        elif ok and start is not None:
            gaps.append((start, i - 1))
            start = None
    if start is not None:
        gaps.append((start, len(valid_1d) - 1))
    return gaps


# --------------------------------------------------------------------------- #
# 2. Lissage optionnel.
# --------------------------------------------------------------------------- #
def smooth(seq: PoseSequence, cfg: PreprocessConfig) -> PoseSequence:
    if cfg.smoothing == "none":
        return seq
    if cfg.smoothing != "moving_avg":
        raise ValueError(f"Lissage inconnu : {cfg.smoothing}")
    w = max(1, int(cfg.smoothing_window))
    if w == 1:
        return seq
    kernel = np.ones(w, dtype=np.float32) / w
    kp = seq.keypoints.copy()
    T, J, D = kp.shape
    pad = w // 2
    for j in range(J):
        for d in range(D):
            padded = np.pad(kp[:, j, d], pad, mode="edge")
            kp[:, j, d] = np.convolve(padded, kernel, mode="same")[pad:pad + T]
    return PoseSequence(
        keypoints=kp, confidence=seq.confidence, valid_mask=seq.valid_mask,
        interpolated_mask=seq.interpolated_mask, timestamps=seq.timestamps,
        fps=seq.fps, joint_names=seq.joint_names, metadata=seq.metadata,
    )


# --------------------------------------------------------------------------- #
# 3. Normalisation spatiale.
# --------------------------------------------------------------------------- #
def normalize(seq: PoseSequence, cfg: PreprocessConfig, schema: SkeletonSchema) -> np.ndarray:
    """Retourne des keypoints normalisés (T, J, D) selon ``cfg.normalization``."""

    kp = seq.keypoints.astype(np.float32).copy()
    if cfg.normalization == "none":
        return kp

    ci, cj = schema.center_joints
    center = 0.5 * (kp[:, ci, :] + kp[:, cj, :])  # (T, D)
    kp = kp - center[:, None, :]

    if cfg.normalization == "torso":
        si, sj = schema.shoulder_joints
        hi, hj = schema.hip_joints
        shoulder = 0.5 * (kp[:, si, :] + kp[:, sj, :])
        hip = 0.5 * (kp[:, hi, :] + kp[:, hj, :])
        scale = np.linalg.norm(shoulder - hip, axis=1)  # (T,)
    elif cfg.normalization == "bbox":
        mn = kp.min(axis=1)
        mx = kp.max(axis=1)
        scale = np.linalg.norm(mx - mn, axis=1)
    else:
        raise ValueError(f"Normalisation inconnue : {cfg.normalization}")

    med = float(np.median(scale[scale > 1e-6])) if np.any(scale > 1e-6) else 1.0
    scale = np.where(scale > 1e-6, scale, med)
    kp = kp / scale[:, None, None]
    return kp.astype(np.float32)


# --------------------------------------------------------------------------- #
# 4. Features multi-flux.
# --------------------------------------------------------------------------- #
def _time_derivative(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Dérivée temporelle le long de l'axe 0 avec dt réel."""

    if x.shape[0] < 2:
        return np.zeros_like(x)
    return np.gradient(x, t, axis=0)


def _bones(kp: np.ndarray, schema: SkeletonSchema) -> np.ndarray:
    """Vecteurs os : articulation - parent (T, J, D)."""

    parents = schema.parents()
    return kp - kp[:, parents, :]


def _angles(kp: np.ndarray, schema: SkeletonSchema) -> np.ndarray:
    """Angle articulaire par sommet (T, J), 0 si non défini."""

    T, J, _ = kp.shape
    out = np.zeros((T, J), dtype=np.float32)
    for _name, (a, b, c) in schema.angle_triplets.items():
        v1 = kp[:, a, :] - kp[:, b, :]
        v2 = kp[:, c, :] - kp[:, b, :]
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        denom = np.clip(n1 * n2, 1e-8, None)
        cos = np.clip((v1 * v2).sum(axis=1) / denom, -1.0, 1.0)
        out[:, b] = np.arccos(cos).astype(np.float32)
    return out


def feature_channels(cfg: PreprocessConfig) -> list[str]:
    """Noms des canaux générés (l'ordre définit ``in_channels``)."""

    d = 3 if cfg.use_z else 2
    axes = ["x", "y", "z"][:d]
    names: list[str] = []
    if "joint" in cfg.streams:
        names += [f"joint_{a}" for a in axes]
    if "bone" in cfg.streams:
        names += [f"bone_{a}" for a in axes]
    if "joint_motion" in cfg.streams:
        names += [f"jvel_{a}" for a in axes]
    if "bone_motion" in cfg.streams:
        names += [f"bvel_{a}" for a in axes]
    if cfg.use_acceleration and "joint" in cfg.streams:
        names += [f"jacc_{a}" for a in axes]
    if cfg.use_angles:
        names.append("angle")
    if cfg.use_confidence:
        names.append("conf")
    if cfg.use_masks:
        names.append("valid")
    return names


def compute_features(seq: PoseSequence, cfg: PreprocessConfig,
                     schema: SkeletonSchema) -> PreprocessedSequence:
    """Assemble les features (T, J, C) à partir d'une séquence nettoyée."""

    d = 3 if cfg.use_z else 2
    kp = normalize(seq, cfg, schema)[:, :, :d]  # (T, J, d)
    t = seq.timestamps
    parts: list[np.ndarray] = []  # chacune (T, J, k)

    if "joint" in cfg.streams:
        parts.append(kp)
    bones = None
    if "bone" in cfg.streams or "bone_motion" in cfg.streams:
        bones = _bones(kp, schema)
    if "bone" in cfg.streams:
        parts.append(bones)
    if "joint_motion" in cfg.streams:
        parts.append(_time_derivative(kp, t))
    if "bone_motion" in cfg.streams:
        parts.append(_time_derivative(bones, t))
    if cfg.use_acceleration and "joint" in cfg.streams:
        parts.append(_time_derivative(_time_derivative(kp, t), t))
    if cfg.use_angles:
        parts.append(_angles(kp, schema)[:, :, None])
    if cfg.use_confidence:
        parts.append(seq.confidence[:, :, None].astype(np.float32))
    if cfg.use_masks:
        parts.append(seq.valid_mask[:, :, None].astype(np.float32))

    features = np.concatenate(parts, axis=2).astype(np.float32)  # (T, J, C)
    return PreprocessedSequence(
        features=features,
        timestamps=t,
        valid_mask=seq.valid_mask,
        interpolated_mask=seq.interpolated_mask,
        fps=seq.fps,
        channel_names=feature_channels(cfg),
    )


def preprocess_sequence(seq: PoseSequence, cfg: PreprocessConfig,
                        schema: SkeletonSchema) -> PreprocessedSequence:
    """Pipeline complet : nettoyage → lissage → features."""

    cleaned = clean_and_interpolate(seq, cfg)
    cleaned = smooth(cleaned, cfg)
    return compute_features(cleaned, cfg, schema)


# --------------------------------------------------------------------------- #
# 5. Fenêtrage temporel (défini en secondes).
# --------------------------------------------------------------------------- #
def sliding_windows(pre: PreprocessedSequence, cfg: PreprocessConfig) -> np.ndarray:
    """Découpe en fenêtres (Nw, C, T_win, J), au plus ``max_windows``.

    La durée/pas sont donnés en secondes puis convertis via ``fps``.
    """

    features = pre.features  # (T, J, C)
    T = features.shape[0]
    win = max(1, int(round(cfg.window_duration_seconds * pre.fps)))
    stride = max(1, int(round(cfg.window_stride_seconds * pre.fps)))

    if T <= win:
        pad = np.repeat(features[-1:], win - T, axis=0) if T < win else features[:0]
        arr = np.concatenate([features, pad], axis=0) if T < win else features
        windows = [arr]
    else:
        starts = list(range(0, T - win + 1, stride))
        if starts[-1] + win < T:
            starts.append(T - win)
        windows = [features[s:s + win] for s in starts]

    # (Nw, T_win, J, C) -> (Nw, C, T_win, J)
    stacked = np.stack(windows).transpose(0, 3, 1, 2).astype(np.float32)
    if stacked.shape[0] > cfg.max_windows:
        sel = np.linspace(0, stacked.shape[0] - 1, cfg.max_windows).round().astype(int)
        stacked = stacked[sel]
    return stacked
