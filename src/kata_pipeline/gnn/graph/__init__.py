"""Construction du graphe spatio-temporel du squelette."""

from kata_pipeline.gnn.graph.skeleton import (
    ANGLE_TRIPLETS,
    JOINT_NAMES,
    NUM_JOINTS,
    SKELETON_EDGES,
    SUBSET_LANDMARKS,
    build_adjacency,
)

__all__ = [
    "ANGLE_TRIPLETS",
    "JOINT_NAMES",
    "NUM_JOINTS",
    "SKELETON_EDGES",
    "SUBSET_LANDMARKS",
    "build_adjacency",
]
