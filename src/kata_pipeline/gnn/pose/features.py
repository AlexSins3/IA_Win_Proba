"""Calcul des features par nœud à partir des keypoints bruts.

Pour chaque frame et chaque articulation on construit un vecteur de canaux :
    [x, y, z, vx, vy, vz, |v|, angle, confiance]

- (x, y, z)   : position normalisée (centrée bassin, mise à l'échelle torse),
- (vx,vy,vz)  : vitesse (dérivée temporelle, en unités normalisées/seconde),
- |v|         : norme de la vitesse (dynamique du geste — utile pour épaules,
                hanches, poignets, chevilles),
- angle       : angle articulaire local (coude, genou, épaule, hanche) ; 0 pour
                les articulations sans triplet défini,
- confiance   : visibilité MediaPipe.

Ces features sont volontairement *géométriques et anonymes* : elles décrivent le
mouvement, jamais l'identité.
"""

from __future__ import annotations

import numpy as np

from kata_pipeline.gnn.graph.skeleton import ANGLE_TRIPLETS, IDX, NUM_JOINTS

# Nombre de canaux produits par ``compute_node_features``.
NUM_CHANNELS = 9


def normalize_pose(keypoints: np.ndarray) -> np.ndarray:
    """Centre sur le bassin et met à l'échelle par la largeur d'épaules.

    Rend le modèle invariant à la position du sujet dans l'image et à sa taille
    apparente (distance à la caméra). keypoints : (T, J, 3).
    """

    kp = keypoints.copy().astype(np.float32)
    l_hip, r_hip = IDX["l_hip"], IDX["r_hip"]
    l_sh, r_sh = IDX["l_shoulder"], IDX["r_shoulder"]

    pelvis = (kp[:, l_hip] + kp[:, r_hip]) / 2.0  # (T, 3)
    kp -= pelvis[:, None, :]

    # Échelle robuste : distance moyenne épaules-bassin sur la séquence.
    shoulder_mid = (kp[:, l_sh] + kp[:, r_sh]) / 2.0
    torso = np.linalg.norm(shoulder_mid[:, :2], axis=1)  # (T,)
    scale = np.median(torso[torso > 1e-6]) if np.any(torso > 1e-6) else 1.0
    if scale < 1e-6:
        scale = 1.0
    return kp / scale


def _joint_angles(keypoints: np.ndarray) -> np.ndarray:
    """Angle (radians) au sommet de chaque triplet défini. (T, J)."""

    t = keypoints.shape[0]
    angles = np.zeros((t, NUM_JOINTS), dtype=np.float32)
    for name, (a, b, c) in ANGLE_TRIPLETS.items():
        ba = keypoints[:, a, :2] - keypoints[:, b, :2]
        bc = keypoints[:, c, :2] - keypoints[:, b, :2]
        num = (ba * bc).sum(axis=1)
        den = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8
        cos = np.clip(num / den, -1.0, 1.0)
        angles[:, IDX[name]] = np.arccos(cos)
    return angles


def compute_node_features(
    keypoints: np.ndarray,
    visibility: np.ndarray,
    fps: float,
    normalize: bool = True,
) -> np.ndarray:
    """Construit le tenseur de features par nœud.

    Entrées :
        keypoints  : (T, J, 3)
        visibility : (T, J)
        fps        : fréquence d'échantillonnage des poses
    Sortie :
        features   : (T, J, NUM_CHANNELS)
    """

    kp = normalize_pose(keypoints) if normalize else keypoints.astype(np.float32)

    # Vitesses par différences finies (unités normalisées / seconde).
    vel = np.zeros_like(kp)
    vel[1:] = (kp[1:] - kp[:-1]) * fps
    speed = np.linalg.norm(vel, axis=2, keepdims=True)  # (T, J, 1)

    angles = _joint_angles(kp)[:, :, None]  # (T, J, 1)
    conf = visibility[:, :, None].astype(np.float32)  # (T, J, 1)

    feats = np.concatenate([kp, vel, speed, angles, conf], axis=2)
    return feats.astype(np.float32)  # (T, J, 9)


def sliding_windows(
    features: np.ndarray,
    window_size: int,
    stride: int,
) -> list[np.ndarray]:
    """Découpe une séquence (T, J, C) en fenêtres (window_size, J, C).

    Les séquences plus courtes que ``window_size`` sont complétées par répétition
    de la dernière frame (padding). Retourne au moins une fenêtre.
    """

    t = features.shape[0]
    if t <= window_size:
        pad = np.repeat(features[-1:], window_size - t, axis=0) if t < window_size else None
        window = features if pad is None else np.concatenate([features, pad], axis=0)
        return [window]

    windows = []
    start = 0
    while start + window_size <= t:
        windows.append(features[start : start + window_size])
        start += stride
    # Garantir la couverture de la fin de séquence.
    if start < t:
        windows.append(features[t - window_size : t])
    return windows
