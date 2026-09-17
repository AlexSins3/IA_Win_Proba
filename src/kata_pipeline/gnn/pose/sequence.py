"""Format interne canonique des poses, indépendant de l'extracteur (P6).

``PoseSequence`` est la frontière entre l'extraction (MediaPipe, MMPose…) et le
reste de la pipeline. Tous les extracteurs produisent cette structure, ce qui
rend l'encodeur et le prétraitement totalement agnostiques de la source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PoseSequence:
    """Séquence de poses d'une prestation.

    Conventions de formes (T = frames, J = articulations, D = 2 ou 3) :
      - ``keypoints``         : (T, J, D) coordonnées normalisées image (x,y[,z]) ;
      - ``confidence``        : (T, J) score de détection par articulation ;
      - ``valid_mask``        : (T, J) bool, True si réellement détectée ;
      - ``interpolated_mask`` : (T, J) bool, True si valeur reconstruite ;
      - ``timestamps``        : (T,) secondes (permet un dt réel non constant) ;
      - ``fps``               : cadence d'échantillonnage nominale ;
      - ``joint_names``       : noms des J articulations (schéma) ;
      - ``metadata``          : infos libres (extracteur, schéma, taille image…).
    """

    keypoints: np.ndarray
    confidence: np.ndarray
    valid_mask: np.ndarray
    interpolated_mask: np.ndarray
    timestamps: np.ndarray
    fps: float
    joint_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return self.keypoints.shape[0]

    @property
    def num_joints(self) -> int:
        return self.keypoints.shape[1]

    @property
    def dims(self) -> int:
        return self.keypoints.shape[2]

    def validate(self) -> None:
        """Vérifie la cohérence des formes (utile avant l'entraînement)."""

        t, j, d = self.keypoints.shape
        assert self.confidence.shape == (t, j), "confidence incohérente"
        assert self.valid_mask.shape == (t, j), "valid_mask incohérent"
        assert self.interpolated_mask.shape == (t, j), "interpolated_mask incohérent"
        assert self.timestamps.shape == (t,), "timestamps incohérents"
        assert len(self.joint_names) == j, "joint_names incohérents"
        assert d in (2, 3), "dimension spatiale attendue 2 ou 3"

    def save(self, path: str | Path) -> Path:
        """Sauvegarde en ``.npz`` (compressé)."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            keypoints=self.keypoints,
            confidence=self.confidence,
            valid_mask=self.valid_mask,
            interpolated_mask=self.interpolated_mask,
            timestamps=self.timestamps,
            fps=np.float32(self.fps),
            joint_names=np.array(self.joint_names),
            metadata=np.array(str(self.metadata)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PoseSequence":
        """Recharge un ``.npz`` produit par ``save`` (ou format legacy)."""

        data = np.load(path, allow_pickle=True)
        keys = set(data.keys())
        if "valid_mask" in keys:  # format canonique
            meta = {}
            if "metadata" in keys:
                try:
                    meta = eval(str(data["metadata"]))  # noqa: S307 - contenu maîtrisé
                except Exception:  # noqa: BLE001
                    meta = {}
            return cls(
                keypoints=data["keypoints"],
                confidence=data["confidence"],
                valid_mask=data["valid_mask"].astype(bool),
                interpolated_mask=data["interpolated_mask"].astype(bool),
                timestamps=data["timestamps"],
                fps=float(data["fps"]),
                joint_names=list(data["joint_names"]),
                metadata=meta,
            )
        # Format legacy (extractor.py v1) : keypoints/visibility/fps/joint_names.
        kp = data["keypoints"]
        vis = data["visibility"]
        t = kp.shape[0]
        fps = float(data["fps"])
        return cls(
            keypoints=kp,
            confidence=vis,
            valid_mask=np.ones(vis.shape, dtype=bool),
            interpolated_mask=np.zeros(vis.shape, dtype=bool),
            timestamps=np.arange(t, dtype=np.float32) / max(fps, 1e-6),
            fps=fps,
            joint_names=list(data["joint_names"]),
            metadata={"format": "legacy"},
        )
