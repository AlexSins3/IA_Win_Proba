"""Schémas de squelette interchangeables (P3/P4).

Un ``SkeletonSchema`` décrit *tout* ce dont la pipeline a besoin pour un jeu
d'articulations donné, indépendamment de l'extracteur de pose :

- ``joint_names``      : noms lisibles ;
- ``source_indices``   : indices correspondants dans le modèle source
  (MediaPipe 33, MMPose whole-body…), pour mapper une détection brute ;
- ``edges``            : arêtes anatomiques (espace du schéma) ;
- ``symmetry``         : paires gauche/droite (pour l'augmentation par miroir) ;
- ``center_joints``    : articulations dont la moyenne définit le centre (bassin) ;
- ``shoulder_joints`` / ``hip_joints`` : références d'échelle corporelle (torse) ;
- ``angle_triplets``   : triplets pour les angles articulaires ;
- ``root``             : racine de l'arbre osseux (pour le stream *bone*).

On privilégie les points utiles au kata (tête, épaules, coudes, poignets,
hanches, genoux, chevilles, pieds). On évite d'injecter tout le visage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from kata_pipeline.gnn.graph.skeleton import (
    ANGLE_TRIPLETS as MP15_ANGLES,
)
from kata_pipeline.gnn.graph.skeleton import (
    JOINT_NAMES as MP15_NAMES,
)
from kata_pipeline.gnn.graph.skeleton import (
    SKELETON_EDGES as MP15_EDGES,
)
from kata_pipeline.gnn.graph.skeleton import (
    SUBSET_LANDMARKS as MP15_SOURCE,
)


@dataclass(frozen=True)
class SkeletonSchema:
    """Description complète d'un jeu d'articulations."""

    name: str
    joint_names: tuple[str, ...]
    source_indices: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    symmetry: tuple[tuple[int, int], ...]
    center_joints: tuple[int, int]
    shoulder_joints: tuple[int, int]
    hip_joints: tuple[int, int]
    angle_triplets: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    root: int = 0

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def index(self, name: str) -> int:
        return self.joint_names.index(name)

    def parents(self) -> list[int]:
        """Arbre osseux : parent de chaque articulation (racine = elle-même)."""

        return _build_parents(self.num_joints, self.edges, self.root)

    def adjacency(self, add_self_loops: bool = True) -> np.ndarray:
        """Adjacence normalisée symétriquement D^-1/2 (A+I) D^-1/2."""

        n = self.num_joints
        a = np.zeros((n, n), dtype=np.float32)
        for i, j in self.edges:
            a[i, j] = a[j, i] = 1.0
        if add_self_loops:
            a += np.eye(n, dtype=np.float32)
        deg = a.sum(axis=1)
        inv = np.zeros_like(deg)
        nz = deg > 0
        inv[nz] = 1.0 / np.sqrt(deg[nz])
        d = np.diag(inv)
        return (d @ a @ d).astype(np.float32)


def _build_parents(num_joints: int, edges: tuple[tuple[int, int], ...], root: int) -> list[int]:
    """Construit un arbre couvrant par BFS ; parent[root] = root."""

    adj: list[list[int]] = [[] for _ in range(num_joints)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    parents = [root] * num_joints
    seen = {root}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                parents[v] = u
                q.append(v)
    return parents


# --------------------------------------------------------------------------- #
# Schéma MediaPipe 15 (baseline actuelle).
# --------------------------------------------------------------------------- #
_MP15_IDX = {n: i for i, n in enumerate(MP15_NAMES)}
MEDIAPIPE_15 = SkeletonSchema(
    name="mediapipe_15",
    joint_names=tuple(MP15_NAMES),
    source_indices=tuple(MP15_SOURCE),
    edges=tuple(MP15_EDGES),
    symmetry=(
        (_MP15_IDX["l_shoulder"], _MP15_IDX["r_shoulder"]),
        (_MP15_IDX["l_elbow"], _MP15_IDX["r_elbow"]),
        (_MP15_IDX["l_wrist"], _MP15_IDX["r_wrist"]),
        (_MP15_IDX["l_hip"], _MP15_IDX["r_hip"]),
        (_MP15_IDX["l_knee"], _MP15_IDX["r_knee"]),
        (_MP15_IDX["l_ankle"], _MP15_IDX["r_ankle"]),
        (_MP15_IDX["l_foot"], _MP15_IDX["r_foot"]),
    ),
    center_joints=(_MP15_IDX["l_hip"], _MP15_IDX["r_hip"]),
    shoulder_joints=(_MP15_IDX["l_shoulder"], _MP15_IDX["r_shoulder"]),
    hip_joints=(_MP15_IDX["l_hip"], _MP15_IDX["r_hip"]),
    angle_triplets=dict(MP15_ANGLES),
    root=_MP15_IDX["l_hip"],
)


# --------------------------------------------------------------------------- #
# Schéma MediaPipe 33 (corps complet + mains grossières + pieds).
# --------------------------------------------------------------------------- #
_MP33_NAMES = (
    "nose", "l_eye_inner", "l_eye", "l_eye_outer", "r_eye_inner", "r_eye",
    "r_eye_outer", "l_ear", "r_ear", "mouth_l", "mouth_r", "l_shoulder",
    "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_pinky",
    "r_pinky", "l_index", "r_index", "l_thumb", "r_thumb", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle", "l_heel", "r_heel",
    "l_foot_index", "r_foot_index",
)
_M = {n: i for i, n in enumerate(_MP33_NAMES)}
_MP33_EDGES = (
    (_M["nose"], _M["l_eye_inner"]), (_M["l_eye_inner"], _M["l_eye"]),
    (_M["l_eye"], _M["l_eye_outer"]), (_M["l_eye_outer"], _M["l_ear"]),
    (_M["nose"], _M["r_eye_inner"]), (_M["r_eye_inner"], _M["r_eye"]),
    (_M["r_eye"], _M["r_eye_outer"]), (_M["r_eye_outer"], _M["r_ear"]),
    (_M["mouth_l"], _M["mouth_r"]),
    (_M["l_shoulder"], _M["r_shoulder"]),
    (_M["l_shoulder"], _M["l_elbow"]), (_M["l_elbow"], _M["l_wrist"]),
    (_M["l_wrist"], _M["l_pinky"]), (_M["l_wrist"], _M["l_index"]),
    (_M["l_wrist"], _M["l_thumb"]), (_M["l_pinky"], _M["l_index"]),
    (_M["r_shoulder"], _M["r_elbow"]), (_M["r_elbow"], _M["r_wrist"]),
    (_M["r_wrist"], _M["r_pinky"]), (_M["r_wrist"], _M["r_index"]),
    (_M["r_wrist"], _M["r_thumb"]), (_M["r_pinky"], _M["r_index"]),
    (_M["l_shoulder"], _M["l_hip"]), (_M["r_shoulder"], _M["r_hip"]),
    (_M["l_hip"], _M["r_hip"]),
    (_M["l_hip"], _M["l_knee"]), (_M["l_knee"], _M["l_ankle"]),
    (_M["l_ankle"], _M["l_heel"]), (_M["l_heel"], _M["l_foot_index"]),
    (_M["l_ankle"], _M["l_foot_index"]),
    (_M["r_hip"], _M["r_knee"]), (_M["r_knee"], _M["r_ankle"]),
    (_M["r_ankle"], _M["r_heel"]), (_M["r_heel"], _M["r_foot_index"]),
    (_M["r_ankle"], _M["r_foot_index"]),
)
_MP33_SYMMETRY = tuple(
    (_M[f"l_{s}"], _M[f"r_{s}"])
    for s in (
        "eye_inner", "eye", "eye_outer", "ear", "shoulder", "elbow", "wrist",
        "pinky", "index", "thumb", "hip", "knee", "ankle", "heel", "foot_index",
    )
) + ((_M["mouth_l"], _M["mouth_r"]),)
_MP33_ANGLES = {
    "l_elbow": (_M["l_shoulder"], _M["l_elbow"], _M["l_wrist"]),
    "r_elbow": (_M["r_shoulder"], _M["r_elbow"], _M["r_wrist"]),
    "l_shoulder": (_M["l_elbow"], _M["l_shoulder"], _M["l_hip"]),
    "r_shoulder": (_M["r_elbow"], _M["r_shoulder"], _M["r_hip"]),
    "l_hip": (_M["l_shoulder"], _M["l_hip"], _M["l_knee"]),
    "r_hip": (_M["r_shoulder"], _M["r_hip"], _M["r_knee"]),
    "l_knee": (_M["l_hip"], _M["l_knee"], _M["l_ankle"]),
    "r_knee": (_M["r_hip"], _M["r_knee"], _M["r_ankle"]),
}
MEDIAPIPE_33 = SkeletonSchema(
    name="mediapipe_33",
    joint_names=_MP33_NAMES,
    source_indices=tuple(range(33)),
    edges=_MP33_EDGES,
    symmetry=_MP33_SYMMETRY,
    center_joints=(_M["l_hip"], _M["r_hip"]),
    shoulder_joints=(_M["l_shoulder"], _M["r_shoulder"]),
    hip_joints=(_M["l_hip"], _M["r_hip"]),
    angle_triplets=_MP33_ANGLES,
    root=_M["l_hip"],
)


# --------------------------------------------------------------------------- #
# Registre.
# --------------------------------------------------------------------------- #
_SCHEMAS: dict[str, SkeletonSchema] = {
    MEDIAPIPE_15.name: MEDIAPIPE_15,
    MEDIAPIPE_33.name: MEDIAPIPE_33,
    # Alias pratique.
    "canonical_body": MEDIAPIPE_15,
    "canonical_body_hands_feet": MEDIAPIPE_33,
}


def get_schema(name: str) -> SkeletonSchema:
    """Récupère un schéma par nom (erreur claire si inconnu)."""

    if name not in _SCHEMAS:
        available = ", ".join(sorted(set(s.name for s in _SCHEMAS.values())))
        raise KeyError(f"Schéma de squelette inconnu : '{name}'. Disponibles : {available}.")
    return _SCHEMAS[name]


def available_schemas() -> list[str]:
    return sorted(_SCHEMAS.keys())
