"""Visualisation vidéo des mouvements et de la saillance du modèle."""

from kata_pipeline.gnn.viz.overlay import (
    compute_joint_saliency,
    render_model_saliency,
    render_motion_overlay,
    render_motion_overlay_from_sequence,
)

__all__ = [
    "compute_joint_saliency",
    "render_model_saliency",
    "render_motion_overlay",
    "render_motion_overlay_from_sequence",
]
