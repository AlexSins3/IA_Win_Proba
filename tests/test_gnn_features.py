"""Tests des features par nœud et du fenêtrage temporel (phase 2 / GNN)."""

from __future__ import annotations

import numpy as np

from kata_pipeline.gnn.graph.skeleton import NUM_JOINTS
from kata_pipeline.gnn.pose.features import (
    NUM_CHANNELS,
    compute_node_features,
    sliding_windows,
)


def _fake_sequence(t: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    kp = rng.random((t, NUM_JOINTS, 3)).astype(np.float32)
    vis = np.ones((t, NUM_JOINTS), dtype=np.float32)
    return kp, vis


def test_feature_shape() -> None:
    kp, vis = _fake_sequence(20)
    feats = compute_node_features(kp, vis, fps=10.0, normalize=True)
    assert feats.shape == (20, NUM_JOINTS, NUM_CHANNELS)
    assert np.isfinite(feats).all()


def test_velocity_scales_with_fps() -> None:
    # Mouvement constant : x augmente de 0.1 par frame sur toutes les articulations.
    t = 5
    kp = np.zeros((t, NUM_JOINTS, 3), dtype=np.float32)
    kp[:, :, 0] = np.arange(t)[:, None] * 0.1
    vis = np.ones((t, NUM_JOINTS), dtype=np.float32)
    feats = compute_node_features(kp, vis, fps=10.0, normalize=False)
    # Canal vx (index 3) doit valoir ~ 0.1 * fps = 1.0 après la 1re frame.
    vx = feats[1:, 0, 3]
    assert np.allclose(vx, 1.0, atol=1e-4)


def test_sliding_windows_padding_short_sequence() -> None:
    kp, vis = _fake_sequence(30)
    feats = compute_node_features(kp, vis, fps=10.0)
    windows = sliding_windows(feats, window_size=120, stride=60)
    assert len(windows) == 1
    assert windows[0].shape == (120, NUM_JOINTS, NUM_CHANNELS)


def test_sliding_windows_covers_end() -> None:
    kp, vis = _fake_sequence(300)
    feats = compute_node_features(kp, vis, fps=10.0)
    windows = sliding_windows(feats, window_size=120, stride=60)
    assert len(windows) >= 3
    for w in windows:
        assert w.shape == (120, NUM_JOINTS, NUM_CHANNELS)
