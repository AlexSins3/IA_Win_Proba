"""Extracteur MediaPipe conforme à l'interface ``PoseExtractor`` (P6).

Produit un ``PoseSequence`` canonique (keypoints + confiance + masques +
timestamps réels) au schéma choisi, et met en cache le résultat en ``.npz``.
N'interpole PAS ici : le nettoyage des trous est fait par ``preprocess`` afin de
rester dépendant de la confiance et de la durée des occlusions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from kata_pipeline.gnn.config import PoseConfig
from kata_pipeline.gnn.pose.mp_backend import PoseBackend
from kata_pipeline.gnn.pose.sequence import PoseSequence
from kata_pipeline.gnn.skeletons import SkeletonSchema

logger = logging.getLogger(__name__)


class MediaPipePoseExtractor:
    """Extrait des poses via MediaPipe Tasks (BlazePose 33 landmarks)."""

    def __init__(self, config: PoseConfig, schema: SkeletonSchema):
        self.config = config
        self.schema = schema

    def extract(self, video_path: str | Path) -> PoseSequence:
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step = max(1, int(round(src_fps / self.config.target_fps)))
        src_idx = np.asarray(self.schema.source_indices, dtype=int)
        n_joints = self.schema.num_joints

        kps: list[np.ndarray] = []
        confs: list[np.ndarray] = []
        valids: list[np.ndarray] = []
        times: list[float] = []

        with PoseBackend(self.config) as backend:
            frame_i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_i % step != 0:
                    frame_i += 1
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                t_sec = frame_i / src_fps
                full_kp, full_vis = backend.process(rgb, int(t_sec * 1000))
                frame_i += 1

                kp = np.zeros((n_joints, 3), dtype=np.float32)
                conf = np.zeros((n_joints,), dtype=np.float32)
                valid = np.zeros((n_joints,), dtype=bool)
                if full_kp is not None:
                    kp[:] = full_kp[src_idx]
                    conf[:] = full_vis[src_idx]
                    valid[:] = conf > 0.0
                kps.append(kp)
                confs.append(conf)
                valids.append(valid)
                times.append(t_sec)
        cap.release()

        if not kps:
            raise RuntimeError(f"Aucune frame lue dans {video_path}")

        keypoints = np.stack(kps)  # (T, J, 3)
        confidence = np.stack(confs)
        valid_mask = np.stack(valids)
        seq = PoseSequence(
            keypoints=keypoints,
            confidence=confidence,
            valid_mask=valid_mask,
            interpolated_mask=np.zeros_like(valid_mask),
            timestamps=np.asarray(times, dtype=np.float32),
            fps=float(self.config.target_fps),
            joint_names=list(self.schema.joint_names),
            metadata={
                "extractor": "mediapipe",
                "schema": self.schema.name,
                "src_fps": float(src_fps),
                "frame_size": [w, h],
                "model_complexity": self.config.model_complexity,
            },
        )
        seq.validate()
        return seq

    def extract_cached(self, video_path: str | Path, cache_path: str | Path,
                       overwrite: bool = False) -> PoseSequence:
        """Extrait avec cache disque (``.npz`` canonique)."""

        cache_path = Path(cache_path)
        if cache_path.exists() and not overwrite:
            return PoseSequence.load(cache_path)
        seq = self.extract(video_path)
        seq.save(cache_path)
        logger.info("Poses (canonique) sauvegardées : %s (T=%d)", cache_path, seq.num_frames)
        return seq
