"""Préparation des vidéos : extraction de plage, redimensionnement."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def is_ffmpeg_available() -> bool:
    """Vérifier si ffmpeg est installé."""
    return shutil.which("ffmpeg") is not None


def extract_time_range(
    input_path: Path,
    output_path: Path,
    start_time: float | None = None,
    end_time: float | None = None,
) -> Path:
    """Extraire une plage temporelle d'une vidéo avec ffmpeg.

    Args:
        input_path: Vidéo source
        output_path: Vidéo de sortie
        start_time: Début en secondes (None = début de la vidéo)
        end_time: Fin en secondes (None = fin de la vidéo)

    Returns:
        Chemin du fichier de sortie
    """
    if not is_ffmpeg_available():
        raise RuntimeError("ffmpeg n'est pas installé.")

    if not input_path.exists():
        raise FileNotFoundError(f"Vidéo source introuvable: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y"]

    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])

    cmd.extend(["-i", str(input_path)])

    if end_time is not None:
        if start_time is not None:
            duration = end_time - start_time
            cmd.extend(["-t", str(duration)])
        else:
            cmd.extend(["-to", str(end_time)])

    cmd.extend(["-c", "copy", str(output_path)])

    logger.info("Extraction plage: %s -> %s", input_path.name, output_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        logger.error("Erreur ffmpeg: %s", result.stderr)
        raise RuntimeError(f"Échec extraction: {result.stderr}")

    return output_path


def create_analysis_version(
    input_path: Path,
    output_path: Path,
    width: int = 480,
    fps: int = 2,
) -> Path:
    """Créer une version basse résolution et basse fréquence pour l'analyse.

    Args:
        input_path: Vidéo source
        output_path: Vidéo de sortie basse résolution
        width: Largeur cible en pixels (hauteur calculée automatiquement)
        fps: Images par seconde cibles

    Returns:
        Chemin du fichier de sortie
    """
    if not is_ffmpeg_available():
        raise RuntimeError("ffmpeg n'est pas installé.")

    if not input_path.exists():
        raise FileNotFoundError(f"Vidéo source introuvable: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"scale={width}:-2,fps={fps}",
        "-an",  # pas d'audio
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        str(output_path),
    ]

    logger.info(
        "Création version analyse: %s (%dpx, %dfps)", output_path.name, width, fps
    )
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        logger.error("Erreur ffmpeg: %s", result.stderr)
        raise RuntimeError(f"Échec création version analyse: {result.stderr}")

    return output_path


def get_video_duration(path: Path) -> float:
    """Obtenir la durée d'une vidéo en secondes via ffprobe."""
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe n'est pas installé.")

    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Échec ffprobe: {result.stderr}")

    return float(result.stdout.strip())


def prepare_video(
    source_path: Path,
    intermediate_dir: Path,
    live_id: str,
    useful_start: float | None = None,
    useful_end: float | None = None,
    analysis_width: int = 480,
    analysis_fps: int = 2,
) -> tuple[Path, Path]:
    """Pipeline complète de préparation vidéo.

    Returns:
        Tuple (chemin vidéo plage utile, chemin vidéo analyse)
    """
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # Si on a une plage utile, extraire
    if useful_start is not None or useful_end is not None:
        range_path = intermediate_dir / f"{live_id}_range.mp4"
        extract_time_range(source_path, range_path, useful_start, useful_end)
        source_for_analysis = range_path
    else:
        source_for_analysis = source_path
        range_path = source_path

    # Créer la version analyse
    analysis_path = intermediate_dir / f"{live_id}_analysis.mp4"
    create_analysis_version(
        source_for_analysis, analysis_path, width=analysis_width, fps=analysis_fps
    )

    return range_path, analysis_path
