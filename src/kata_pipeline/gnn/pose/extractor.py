"""Extraction de poses depuis une vidéo de kata avec MediaPipe Pose.

Sortie : un fichier ``.npz`` par clip contenant :
- ``keypoints`` : (T, J, 3) coordonnées normalisées (x, y, z) du sous-ensemble
  d'articulations, dans [0, 1] pour x/y (z relatif),
- ``visibility`` : (T, J) confiance MediaPipe par articulation,
- ``fps`` : fps d'échantillonnage effectif,
- ``joint_names`` : noms des articulations.

Les frames sans détection sont interpolées linéairement (le sujet reste dans le
cadre ; les trous viennent surtout d'occlusions courtes).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from kata_pipeline.gnn.config import PoseConfig
from kata_pipeline.gnn.graph.skeleton import JOINT_NAMES, SUBSET_LANDMARKS
from kata_pipeline.gnn.pose.mp_backend import PoseBackend

logger = logging.getLogger(__name__)


def _interpolate_gaps(keypoints: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Interpole linéairement les frames manquantes, par articulation.

    keypoints : (T, J, C) ; valid : (T,) booléen (frame détectée).
    """

    t = keypoints.shape[0]
    idx = np.arange(t)
    if valid.sum() < 2:
        return keypoints
    good = idx[valid]
    for j in range(keypoints.shape[1]):
        for c in range(keypoints.shape[2]):
            keypoints[:, j, c] = np.interp(idx, good, keypoints[good, j, c])
    return keypoints


def extract_poses(
    video_path: str | Path,
    config: PoseConfig | None = None,
) -> dict[str, np.ndarray | float | list[str]]:
    """Extrait la séquence de poses d'une vidéo.

    Retourne un dictionnaire prêt à être sauvegardé via ``np.savez``.
    Utilise l'API MediaPipe Tasks (``PoseLandmarker``), compatible Python 3.13.
    """

    config = config or PoseConfig()
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / config.target_fps)))

    frames_kp: list[np.ndarray] = []
    frames_vis: list[np.ndarray] = []
    valid: list[bool] = []

    with PoseBackend(config) as backend:
        frame_i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_i % step != 0:
                frame_i += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(frame_i / src_fps * 1000)
            full_kp, full_vis = backend.process(rgb, timestamp_ms)
            frame_i += 1

            kp = np.zeros((len(SUBSET_LANDMARKS), 3), dtype=np.float32)
            vis = np.zeros((len(SUBSET_LANDMARKS),), dtype=np.float32)
            if full_kp is not None:
                for out_i, mp_i in enumerate(SUBSET_LANDMARKS):
                    kp[out_i] = full_kp[mp_i]
                    vis[out_i] = full_vis[mp_i]
                valid.append(True)
            else:
                valid.append(False)
            frames_kp.append(kp)
            frames_vis.append(vis)

    cap.release()

    if not frames_kp:
        raise RuntimeError(f"Aucune frame lue dans {video_path}")

    keypoints = np.stack(frames_kp)  # (T, J, 3)
    visibility = np.stack(frames_vis)  # (T, J)
    valid_arr = np.array(valid, dtype=bool)
    keypoints = _interpolate_gaps(keypoints, valid_arr)

    n_missing = int((~valid_arr).sum())
    if n_missing:
        logger.warning(
            "%s : %d/%d frames sans détection (interpolées).",
            video_path.name,
            n_missing,
            len(valid_arr),
        )

    return {
        "keypoints": keypoints,
        "visibility": visibility,
        "fps": float(config.target_fps),
        "joint_names": list(JOINT_NAMES),
    }


def extract_and_save(
    video_path: str | Path,
    out_path: str | Path,
    config: PoseConfig | None = None,
) -> Path:
    """Extrait les poses et les sauvegarde en ``.npz``."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = extract_poses(video_path, config)
    np.savez_compressed(out_path, **data)
    logger.info("Poses sauvegardées : %s (T=%d)", out_path, data["keypoints"].shape[0])
    return out_path
