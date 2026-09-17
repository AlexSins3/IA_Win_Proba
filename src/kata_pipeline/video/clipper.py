"""Extraction de clips vidéo individuels."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from kata_pipeline.schemas.clip import KataClip
from kata_pipeline.utils.filenames import generate_clip_filename

logger = logging.getLogger(__name__)


def extract_clip(
    source_path: Path,
    output_dir: Path,
    clip: KataClip,
    margin_before: float = 3.0,
    margin_after: float = 3.0,
) -> Path:
    """Extraire un clip vidéo individuel depuis la source.

    Args:
        source_path: Vidéo source (plage utile du live)
        output_dir: Dossier de sortie
        clip: Métadonnées du clip à extraire
        margin_before: Marge en secondes avant le début
        margin_after: Marge en secondes après la fin

    Returns:
        Chemin du clip extrait
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg n'est pas installé.")

    if not source_path.exists():
        raise FileNotFoundError(f"Vidéo source introuvable: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    filename = generate_clip_filename(clip)
    output_path = output_dir / filename

    start = max(0.0, clip.start_time - margin_before)
    duration = (clip.end_time + margin_after) - start

    # Seek rapide à 10s avant le point cible, puis seek précis post-input
    pre_seek = max(0.0, start - 10.0)
    post_seek = start - pre_seek

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{pre_seek:.3f}",
        "-i", str(source_path),
        "-ss", f"{post_seek:.3f}",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        str(output_path),
    ]

    logger.info("Extraction clip: %s (%.1fs -> %.1fs)", filename, start, start + duration)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        logger.error("Erreur extraction clip: %s", result.stderr)
        raise RuntimeError(f"Échec extraction clip {filename}: {result.stderr}")

    return output_path


def extract_all_clips(
    source_path: Path,
    output_dir: Path,
    clips: list[KataClip],
    margin_before: float = 3.0,
    margin_after: float = 3.0,
) -> list[tuple[KataClip, Path | None]]:
    """Extraire tous les clips d'une liste.

    Returns:
        Liste de tuples (clip, chemin_extrait ou None si erreur)
    """
    results: list[tuple[KataClip, Path | None]] = []

    for clip in clips:
        if clip.start_time == 0.0 and clip.end_time == 0.0:
            logger.warning("Clip %s sans timestamps, extraction skippée", clip.id_clip)
            results.append((clip, None))
            continue

        try:
            path = extract_clip(source_path, output_dir, clip, margin_before, margin_after)
            results.append((clip, path))
        except Exception as e:
            logger.error("Erreur extraction clip %s: %s", clip.id_clip, e)
            results.append((clip, None))

    extracted = sum(1 for _, p in results if p is not None)
    logger.info("Clips extraits: %d/%d", extracted, len(clips))
    return results
