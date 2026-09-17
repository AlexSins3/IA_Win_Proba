"""Pipeline et interface de prédiction d'un match à partir de deux vidéos."""

from kata_pipeline.prediction.pipeline import (
    VideoMetadata,
    extract_pose_sequence,
    normalize_video,
    pose_summary,
    probable_match_score,
    probe_video,
    render_web_motion_overlay,
)

__all__ = [
    "VideoMetadata",
    "extract_pose_sequence",
    "normalize_video",
    "pose_summary",
    "probable_match_score",
    "probe_video",
    "render_web_motion_overlay",
]
