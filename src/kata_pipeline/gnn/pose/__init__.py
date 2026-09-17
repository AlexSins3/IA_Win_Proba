"""Extraction de poses et features à partir des clips vidéo."""

from kata_pipeline.gnn.pose.extractor import extract_and_save, extract_poses
from kata_pipeline.gnn.pose.features import (
    NUM_CHANNELS,
    compute_node_features,
    normalize_pose,
    sliding_windows,
)

__all__ = [
    "NUM_CHANNELS",
    "compute_node_features",
    "extract_and_save",
    "extract_poses",
    "normalize_pose",
    "sliding_windows",
]
