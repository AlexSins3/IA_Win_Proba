"""Topologie du squelette : nœuds, arêtes et matrices d'adjacence.

On s'appuie sur les 33 landmarks de MediaPipe Pose mais on n'en garde que 15
articulations stables et pertinentes pour le kata (épaules, coudes, poignets,
hanches, genoux, chevilles, pieds, nez). Les mains/pieds fins de MediaPipe sont
trop bruités sur des vidéos de compétition filmées de loin.

Ce module fournit :
- ``SUBSET_LANDMARKS`` : indices MediaPipe conservés,
- ``JOINT_NAMES`` : noms lisibles,
- ``SKELETON_EDGES`` : arêtes anatomiques (index dans le sous-ensemble),
- ``build_adjacency`` : matrice d'adjacence normalisée pour le ST-GCN.
"""

from __future__ import annotations

import numpy as np

# --- Indices MediaPipe Pose (33 landmarks) que l'on conserve --------------------
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
MP_NOSE = 0
MP_L_SHOULDER, MP_R_SHOULDER = 11, 12
MP_L_ELBOW, MP_R_ELBOW = 13, 14
MP_L_WRIST, MP_R_WRIST = 15, 16
MP_L_HIP, MP_R_HIP = 23, 24
MP_L_KNEE, MP_R_KNEE = 25, 26
MP_L_ANKLE, MP_R_ANKLE = 27, 28
MP_L_FOOT, MP_R_FOOT = 31, 32  # foot_index

# Ordre du sous-ensemble utilisé partout dans la phase 2.
SUBSET_LANDMARKS: list[int] = [
    MP_NOSE,
    MP_L_SHOULDER,
    MP_R_SHOULDER,
    MP_L_ELBOW,
    MP_R_ELBOW,
    MP_L_WRIST,
    MP_R_WRIST,
    MP_L_HIP,
    MP_R_HIP,
    MP_L_KNEE,
    MP_R_KNEE,
    MP_L_ANKLE,
    MP_R_ANKLE,
    MP_L_FOOT,
    MP_R_FOOT,
]

JOINT_NAMES: list[str] = [
    "nose",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
    "l_foot",
    "r_foot",
]

NUM_JOINTS = len(SUBSET_LANDMARKS)

# Index rapide nom -> position dans le sous-ensemble.
IDX = {name: i for i, name in enumerate(JOINT_NAMES)}

# --- Arêtes anatomiques (dans l'espace du sous-ensemble) ------------------------
SKELETON_EDGES: list[tuple[int, int]] = [
    (IDX["nose"], IDX["l_shoulder"]),
    (IDX["nose"], IDX["r_shoulder"]),
    (IDX["l_shoulder"], IDX["r_shoulder"]),
    (IDX["l_shoulder"], IDX["l_elbow"]),
    (IDX["l_elbow"], IDX["l_wrist"]),
    (IDX["r_shoulder"], IDX["r_elbow"]),
    (IDX["r_elbow"], IDX["r_wrist"]),
    (IDX["l_shoulder"], IDX["l_hip"]),
    (IDX["r_shoulder"], IDX["r_hip"]),
    (IDX["l_hip"], IDX["r_hip"]),
    (IDX["l_hip"], IDX["l_knee"]),
    (IDX["l_knee"], IDX["l_ankle"]),
    (IDX["l_ankle"], IDX["l_foot"]),
    (IDX["r_hip"], IDX["r_knee"]),
    (IDX["r_knee"], IDX["r_ankle"]),
    (IDX["r_ankle"], IDX["r_foot"]),
]

# Triplets d'articulations pour le calcul des angles (a - b - c), b = sommet.
ANGLE_TRIPLETS: dict[str, tuple[int, int, int]] = {
    "l_elbow": (IDX["l_shoulder"], IDX["l_elbow"], IDX["l_wrist"]),
    "r_elbow": (IDX["r_shoulder"], IDX["r_elbow"], IDX["r_wrist"]),
    "l_shoulder": (IDX["l_elbow"], IDX["l_shoulder"], IDX["l_hip"]),
    "r_shoulder": (IDX["r_elbow"], IDX["r_shoulder"], IDX["r_hip"]),
    "l_hip": (IDX["l_shoulder"], IDX["l_hip"], IDX["l_knee"]),
    "r_hip": (IDX["r_shoulder"], IDX["r_hip"], IDX["r_knee"]),
    "l_knee": (IDX["l_hip"], IDX["l_knee"], IDX["l_ankle"]),
    "r_knee": (IDX["r_hip"], IDX["r_knee"], IDX["r_ankle"]),
}


def build_adjacency(add_self_loops: bool = True) -> np.ndarray:
    """Construit la matrice d'adjacence normalisée symétriquement.

    Retourne A_hat = D^{-1/2} (A + I) D^{-1/2}, de forme (NUM_JOINTS, NUM_JOINTS),
    utilisée par les couches de convolution de graphe.
    """

    a = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
    for i, j in SKELETON_EDGES:
        a[i, j] = 1.0
        a[j, i] = 1.0
    if add_self_loops:
        a += np.eye(NUM_JOINTS, dtype=np.float32)

    deg = a.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    deg_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    d_mat = np.diag(deg_inv_sqrt)
    return (d_mat @ a @ d_mat).astype(np.float32)
