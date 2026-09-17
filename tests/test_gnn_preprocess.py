"""Tests du prétraitement robuste et des features (P2/P3)."""

from __future__ import annotations

import numpy as np

from kata_pipeline.gnn.config import PreprocessConfig
from kata_pipeline.gnn.pose.sequence import PoseSequence
from kata_pipeline.gnn.preprocess import (
    clean_and_interpolate,
    compute_features,
    feature_channels,
    preprocess_sequence,
    sliding_windows,
)
from kata_pipeline.gnn.skeletons import get_schema

SCHEMA = get_schema("mediapipe_15")


def _make_seq(T=60, fps=10.0, jitter=True):
    rng = np.random.default_rng(0)
    kp = rng.normal(0.5, 0.1, size=(T, 15, 3)).astype(np.float32)
    conf = np.ones((T, 15), dtype=np.float32)
    valid = np.ones((T, 15), dtype=bool)
    ts = np.arange(T, dtype=np.float32) / fps
    return PoseSequence(kp, conf, valid, np.zeros_like(valid), ts, fps,
                        list(SCHEMA.joint_names), {})


def test_short_gap_interpolated():
    seq = _make_seq()
    # Trou court (2 frames) sur l'articulation 3.
    seq.valid_mask[10:12, 3] = False
    seq.confidence[10:12, 3] = 0.0
    cfg = PreprocessConfig(max_interp_gap_seconds=0.5)
    out = clean_and_interpolate(seq, cfg)
    assert out.interpolated_mask[10:12, 3].all()
    assert out.valid_mask[10:12, 3].all()


def test_long_gap_left_masked():
    seq = _make_seq(T=100)
    # Occlusion longue (30 frames = 3 s @10fps > 0.5 s).
    seq.valid_mask[20:50, 5] = False
    seq.confidence[20:50, 5] = 0.0
    cfg = PreprocessConfig(max_interp_gap_seconds=0.5)
    out = clean_and_interpolate(seq, cfg)
    assert not out.valid_mask[20:50, 5].any()
    assert not out.interpolated_mask[20:50, 5].any()


def test_always_absent_joint_zeroed():
    seq = _make_seq()
    seq.valid_mask[:, 7] = False
    seq.confidence[:, 7] = 0.0
    out = clean_and_interpolate(seq, PreprocessConfig())
    assert np.allclose(out.keypoints[:, 7], 0.0)
    assert not out.valid_mask[:, 7].any()


def test_feature_channels_match_compute():
    cfg = PreprocessConfig(streams=["joint", "joint_motion"], use_angles=True, use_confidence=True)
    seq = clean_and_interpolate(_make_seq(), cfg)
    pre = compute_features(seq, cfg, SCHEMA)
    assert pre.features.shape[2] == len(feature_channels(cfg))
    # joint(3)+jvel(3)+angle(1)+conf(1) = 8
    assert pre.features.shape[2] == 8


def test_derivative_uses_real_dt():
    # Vitesse constante : position linéaire en t -> dérivée ~ constante.
    T = 20
    fps = 10.0
    ts = np.arange(T, dtype=np.float32) / fps
    kp = np.zeros((T, 15, 3), dtype=np.float32)
    kp[:, 0, 0] = ts  # x = t -> vx = 1
    conf = np.ones((T, 15), dtype=np.float32)
    valid = np.ones((T, 15), dtype=bool)
    seq = PoseSequence(kp, conf, valid, np.zeros_like(valid), ts, fps,
                       list(SCHEMA.joint_names), {})
    cfg = PreprocessConfig(streams=["joint", "joint_motion"], normalization="none",
                           use_angles=False, use_confidence=False)
    pre = compute_features(seq, cfg, SCHEMA)
    # canaux : [x,y,z, vx,vy,vz]; vx est l'indice 3.
    vx = pre.features[:, 0, 3]
    assert np.allclose(vx.mean(), 1.0, atol=1e-3)


def test_sliding_windows_shape_and_cap():
    cfg = PreprocessConfig(window_duration_seconds=2.0, window_stride_seconds=1.0, max_windows=4)
    seq = _make_seq(T=100, fps=10.0)
    pre = preprocess_sequence(seq, cfg, SCHEMA)
    win = sliding_windows(pre, cfg)
    Nw, C, T, J = win.shape
    assert Nw <= 4
    assert J == 15
    assert C == pre.features.shape[2]
    assert T == 20  # 2 s @10 fps
