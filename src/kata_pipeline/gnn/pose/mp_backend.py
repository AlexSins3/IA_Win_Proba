"""Backend MediaPipe basé sur l'API *Tasks* (PoseLandmarker).

Les builds récents de MediaPipe (0.10.x, Python 3.13) ne fournissent plus
l'ancienne API ``mediapipe.solutions``. On utilise donc ``PoseLandmarker`` de
l'API Tasks, qui expose les mêmes 33 landmarks BlazePose mais nécessite un
fichier modèle ``.task`` (téléchargé automatiquement et mis en cache).

Un seul point d'entrée : ``PoseBackend`` (context manager) avec ``process``.
Partagé par l'extraction (``pose/extractor.py``) et la visualisation
(``viz/overlay.py``) pour garantir une topologie identique.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np

from kata_pipeline.gnn.config import PoseConfig

logger = logging.getLogger(__name__)

# Modèles officiels (lite / full / heavy) mappés sur model_complexity 0/1/2.
_MODEL_URLS = {
    0: (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    ),
    1: (
        "pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    ),
    2: (
        "pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    ),
}

# Nombre de landmarks renvoyés par BlazePose / PoseLandmarker.
NUM_MP_LANDMARKS = 33


def default_model_dir() -> Path:
    """Dossier de cache local des modèles ``.task``."""

    return Path("models") / "mediapipe"


def ensure_model(complexity: int, model_dir: Path | None = None) -> Path:
    """Retourne le chemin du modèle ``.task``, le téléchargeant si nécessaire."""

    complexity = max(0, min(2, complexity))
    filename, url = _MODEL_URLS[complexity]
    model_dir = model_dir or default_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / filename
    if not path.exists():
        logger.info("Téléchargement du modèle MediaPipe : %s", url)
        urllib.request.urlretrieve(url, path)  # noqa: S310 (URL de confiance Google)
        logger.info("Modèle enregistré : %s", path)
    return path


class PoseBackend:
    """Encapsule un ``PoseLandmarker`` en mode VIDEO (context manager).

    Usage :
        with PoseBackend(config) as backend:
            kp, vis = backend.process(rgb, timestamp_ms)
    ``kp`` : (33, 3) x/y normalisés + z ; ``vis`` : (33,) visibilité.
    Retourne (None, None) si aucune pose détectée.
    """

    def __init__(self, config: PoseConfig, model_dir: Path | None = None):
        self.config = config
        self.model_dir = model_dir
        self._landmarker = None

    def __enter__(self) -> "PoseBackend":
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        model_path = ensure_model(self.config.model_complexity, self.model_dir)
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            min_pose_presence_confidence=self.config.min_detection_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._mp = mp
        return self

    def __exit__(self, *exc) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def process(
        self, rgb: np.ndarray, timestamp_ms: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Détecte la pose sur une frame RGB (uint8) à un timestamp donné (ms)."""

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))
        if not result.pose_landmarks:
            return None, None
        landmarks = result.pose_landmarks[0]
        kp = np.zeros((NUM_MP_LANDMARKS, 3), dtype=np.float32)
        vis = np.zeros((NUM_MP_LANDMARKS,), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            kp[i] = (lm.x, lm.y, lm.z)
            vis[i] = getattr(lm, "visibility", 0.0)
        return kp, vis
