"""Services utilisés par l'interface Streamlit de prédiction.

Les fichiers d'entrée sont d'abord confiés à FFmpeg, ce qui permet
d'accepter tout conteneur/codec que l'installation locale sait décoder. La
sortie normalisée est un MP4 H.264 sans piste audio ni métadonnées.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kata_pipeline.gnn.config import GNNConfig
from kata_pipeline.gnn.pose.base import build_extractor
from kata_pipeline.gnn.pose.sequence import PoseSequence
from kata_pipeline.gnn.skeletons import SkeletonSchema
from kata_pipeline.gnn.viz.overlay import render_motion_overlay_from_sequence


class VideoProcessingError(RuntimeError):
    """Erreur de validation ou de conversion vidéo présentable à l'utilisateur."""


@dataclass(frozen=True)
class VideoMetadata:
    """Informations utiles sur une vidéo, sans données identifiantes."""

    duration_seconds: float
    width: int
    height: int
    fps: float
    codec: str
    container: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary is None:
        raise VideoProcessingError(
            f"{name} est requis pour lire et convertir les vidéos, mais il est introuvable."
        )
    return binary


def _run_media_command(command: list[str], action: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "erreur inconnue").strip()
        # FFmpeg peut produire plusieurs pages de diagnostic. La fin contient
        # généralement la cause et reste lisible dans l'interface.
        details = "\n".join(details.splitlines()[-8:])
        raise VideoProcessingError(f"Échec pendant {action}.\n{details}")
    return result


def _parse_rate(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    numerator, separator, denominator = value.partition("/")
    try:
        return float(numerator) / float(denominator) if separator else float(numerator)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def probe_video(path: str | Path) -> VideoMetadata:
    """Valide la présence d'une piste vidéo et retourne ses métadonnées."""

    path = Path(path)
    ffprobe = _require_binary("ffprobe")
    result = _run_media_command(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,duration:format=duration,format_name",
            "-of",
            "json",
            str(path),
        ],
        "l'analyse du fichier vidéo",
    )
    try:
        payload = json.loads(result.stdout)
        stream = payload.get("streams", [])[0]
        fmt = payload.get("format", {})
    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
        raise VideoProcessingError(
            "Le fichier ne contient aucune piste vidéo exploitable."
        ) from exc

    duration_raw = stream.get("duration") or fmt.get("duration") or 0.0
    try:
        duration = float(duration_raw)
    except (TypeError, ValueError):
        duration = 0.0
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise VideoProcessingError("La piste vidéo a des dimensions invalides.")
    return VideoMetadata(
        duration_seconds=max(0.0, duration),
        width=width,
        height=height,
        fps=_parse_rate(stream.get("avg_frame_rate")),
        codec=str(stream.get("codec_name") or "inconnu"),
        container=str(fmt.get("format_name") or "inconnu"),
    )


def normalize_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_width: int = 1280,
    output_fps: int = 30,
) -> VideoMetadata:
    """Convertit une vidéo en MP4/H.264 CFR, yuv420p et sans audio.

    FFmpeg détecte le format à partir du contenu : l'extension du fichier
    déposé n'est donc pas utilisée comme preuve de validité.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file() or input_path.stat().st_size == 0:
        raise VideoProcessingError("Le fichier vidéo est vide ou introuvable.")
    probe_video(input_path)
    ffmpeg = _require_binary("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scale = f"scale=w='min({int(max_width)},iw)':h=-2"
    _run_media_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-vf",
            scale,
            "-r",
            str(int(output_fps)),
            "-fps_mode",
            "cfr",
            "-an",
            "-sn",
            "-dn",
            "-map_metadata",
            "-1",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        "la conversion en MP4/H.264",
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise VideoProcessingError("La conversion n'a produit aucune vidéo.")
    return probe_video(output_path)


def extract_pose_sequence(
    video_path: str | Path,
    config: GNNConfig,
    schema: SkeletonSchema,
) -> PoseSequence:
    """Extrait les poses dans le schéma attendu par le checkpoint."""

    extractor = build_extractor(config.pose, schema)
    return extractor.extract(video_path)


def pose_summary(sequence: PoseSequence, confidence_threshold: float = 0.3) -> dict[str, Any]:
    """Calcule les indicateurs de qualité utiles à l'écran de résultat."""

    detected = sequence.valid_mask & (sequence.confidence >= confidence_threshold)
    detected_confidence = sequence.confidence[detected]
    if sequence.num_frames > 1:
        frame_step = float(np.median(np.diff(sequence.timestamps)))
    else:
        frame_step = 1.0 / max(float(sequence.fps), 1e-6)
    duration = float(sequence.timestamps[-1] - sequence.timestamps[0] + frame_step)
    return {
        "duration_seconds": round(max(0.0, duration), 2),
        "pose_frames": int(sequence.num_frames),
        "frame_detection_rate": round(float(detected.any(axis=1).mean()), 4),
        "joint_detection_rate": round(float(detected.mean()), 4),
        "mean_joint_confidence": round(
            float(detected_confidence.mean()) if detected_confidence.size else 0.0,
            4,
        ),
    }


def render_web_motion_overlay(
    video_path: str | Path,
    output_path: str | Path,
    sequence: PoseSequence,
    schema: SkeletonSchema,
    confidence_threshold: float = 0.3,
) -> Path:
    """Crée un overlay squelette puis garantit un MP4 lisible par navigateur."""

    output_path = Path(output_path)
    raw_path = output_path.with_name(f"{output_path.stem}_raw.mp4")
    render_motion_overlay_from_sequence(
        video_path,
        raw_path,
        sequence,
        schema,
        confidence_threshold=confidence_threshold,
    )
    ffmpeg = _require_binary("ffmpeg")
    try:
        _run_media_command(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(raw_path),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            "la finalisation de la vidéo squelette",
        )
    finally:
        raw_path.unlink(missing_ok=True)
    return output_path
