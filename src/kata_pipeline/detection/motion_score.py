"""Calcul du score de mouvement à partir de la vidéo d'analyse."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from kata_pipeline.config import MotionConfig

logger = logging.getLogger(__name__)


class MotionScoreResult:
    """Résultat du calcul de score de mouvement."""

    def __init__(
        self,
        scores: list[float],
        timestamps: list[float],
        fps: float,
        smoothed_scores: list[float] | None = None,
    ):
        self.scores = scores
        self.timestamps = timestamps
        self.fps = fps
        self.smoothed_scores = smoothed_scores or scores

    def save(self, path: Path) -> None:
        """Sauvegarder les scores dans un fichier JSON reproductible."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "fps": self.fps,
            "num_frames": len(self.scores),
            "duration": self.timestamps[-1] if self.timestamps else 0.0,
            "timestamps": self.timestamps,
            "raw_scores": self.scores,
            "smoothed_scores": self.smoothed_scores,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Scores de mouvement sauvegardés: %s", path)

    @classmethod
    def load(cls, path: Path) -> "MotionScoreResult":
        """Charger les scores depuis un fichier JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            scores=data["raw_scores"],
            timestamps=data["timestamps"],
            fps=data["fps"],
            smoothed_scores=data.get("smoothed_scores"),
        )


def compute_motion_scores(
    video_path: Path,
    config: MotionConfig | None = None,
    analysis_fps: float | None = None,
    analysis_width: int | None = None,
) -> MotionScoreResult:
    """Calculer le score de mouvement frame par frame.

    Utilise ffmpeg en subprocess pour extraire les frames à basse résolution/fps
    directement en mémoire via pipe (beaucoup plus rapide que OpenCV frame-by-frame).

    Args:
        video_path: Chemin vers la vidéo source
        config: Configuration des paramètres de mouvement
        analysis_fps: FPS cible pour l'analyse
        analysis_width: Largeur cible pour le resize

    Returns:
        MotionScoreResult avec les scores bruts et lissés
    """
    if config is None:
        config = MotionConfig()

    if not video_path.exists():
        raise FileNotFoundError(f"Vidéo introuvable: {video_path}")

    # Déterminer les paramètres
    target_fps = analysis_fps or 5.0
    width = analysis_width or 480

    # Obtenir la résolution de sortie via ffprobe
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(video_path),
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    src_w, src_h = map(int, probe_result.stdout.strip().split(","))

    # Calculer la hauteur proportionnelle (arrondie au pair)
    out_h = int(src_h * width / src_w)
    out_h = out_h + (out_h % 2)  # arrondir au pair supérieur

    # Utiliser ffmpeg pour extraire frames en grayscale via pipe
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(video_path),
        "-vf", f"fps={target_fps},scale={width}:{out_h}",
        "-pix_fmt", "gray",
        "-f", "rawvideo",
        "pipe:1",
    ]

    logger.info(
        "Analyse mouvement: %s (ffmpeg pipe -> %dfps %dx%d gray)",
        video_path.name, target_fps, width, out_h,
    )

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_size = width * out_h
    scores: list[float] = []
    timestamps: list[float] = []
    prev_gray: np.ndarray | None = None
    frame_idx = 0
    kernel_size = config.blur_kernel_size

    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break

        gray = np.frombuffer(raw, dtype=np.uint8).reshape(out_h, width)
        gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            score = float(np.mean(diff))
            scores.append(score)
            timestamps.append(frame_idx / target_fps)
        else:
            scores.append(0.0)
            timestamps.append(0.0)

        prev_gray = gray
        frame_idx += 1

    proc.stdout.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        raise RuntimeError(f"ffmpeg pipe error: {stderr[:500]}")

    if not scores:
        raise RuntimeError("Aucune frame lue depuis la vidéo")

    logger.info("Mouvement calculé: %d scores, durée %.1fs", len(scores), timestamps[-1])

    # Lissage avec fenêtre glissante
    smoothed = smooth_scores(scores, target_fps, config.smoothing_window)

    return MotionScoreResult(
        scores=scores,
        timestamps=timestamps,
        fps=target_fps,
        smoothed_scores=smoothed,
    )

    # Lissage avec fenêtre glissante
    smoothed = smooth_scores(scores, effective_fps, config.smoothing_window)

    return MotionScoreResult(
        scores=scores,
        timestamps=timestamps,
        fps=effective_fps,
        smoothed_scores=smoothed,
    )


def smooth_scores(scores: list[float], fps: float, window_seconds: float) -> list[float]:
    """Lisser les scores avec une moyenne glissante.

    Args:
        scores: Scores bruts
        fps: Images par seconde
        window_seconds: Taille de la fenêtre en secondes

    Returns:
        Scores lissés
    """
    window_size = max(1, int(window_seconds * fps))
    if window_size <= 1:
        return scores.copy()

    scores_array = np.array(scores, dtype=np.float64)
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(scores_array, kernel, mode="same")

    return smoothed.tolist()
