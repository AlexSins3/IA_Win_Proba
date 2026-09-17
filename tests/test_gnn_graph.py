"""Tests de la topologie du squelette et de l'adjacence (phase 2 / GNN)."""

from __future__ import annotations

import numpy as np

from kata_pipeline.gnn.graph.skeleton import (
    JOINT_NAMES,
    NUM_JOINTS,
    SKELETON_EDGES,
    build_adjacency,
)


def test_joint_count_is_15() -> None:
    assert NUM_JOINTS == 15
    assert len(JOINT_NAMES) == 15


def test_adjacency_shape_and_symmetry() -> None:
    a = build_adjacency()
    assert a.shape == (NUM_JOINTS, NUM_JOINTS)
    # Symétrie (graphe non orienté normalisé symétriquement).
    assert np.allclose(a, a.T, atol=1e-6)


def test_adjacency_has_self_loops_and_is_finite() -> None:
    a = build_adjacency(add_self_loops=True)
    assert np.all(np.diag(a) > 0.0)  # boucles sur soi
    assert np.isfinite(a).all()


def test_edges_reference_valid_joints() -> None:
    for i, j in SKELETON_EDGES:
        assert 0 <= i < NUM_JOINTS
        assert 0 <= j < NUM_JOINTS
        assert i != j
