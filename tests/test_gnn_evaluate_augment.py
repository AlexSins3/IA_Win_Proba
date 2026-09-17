"""Tests des métriques d'issue et des augmentations (P15/P16)."""

from __future__ import annotations

import numpy as np

from kata_pipeline.gnn.augment import AugmentConfig, augment_pose, horizontal_flip, joint_dropout
from kata_pipeline.gnn.evaluate import expected_calibration_error, outcome_metrics
from kata_pipeline.gnn.pose.sequence import PoseSequence
from kata_pipeline.gnn.skeletons import get_schema

SCHEMA = get_schema("mediapipe_15")


def _seq(T=40):
    rng = np.random.default_rng(1)
    kp = rng.normal(0.0, 0.1, size=(T, 15, 3)).astype(np.float32)
    conf = np.ones((T, 15), dtype=np.float32)
    valid = np.ones((T, 15), dtype=bool)
    ts = np.arange(T, dtype=np.float32) / 10.0
    return PoseSequence(kp, conf, valid, np.zeros_like(valid), ts, 10.0,
                        list(SCHEMA.joint_names), {})


def test_outcome_metrics_perfect():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    m = outcome_metrics(y, p)
    assert m["accuracy"] == 1.0
    assert m["brier"] < 0.1


def test_outcome_metrics_swap_symmetry():
    y = np.array([0, 1])
    p_b = np.array([0.3, 0.7])
    p_b_sw = 1.0 - p_b  # antisymétrie parfaite
    m = outcome_metrics(y, p_b, p_b_swapped=p_b_sw)
    assert m["swap_symmetry_error"] < 1e-6
    assert m["swap_consistency"] == 1.0


def test_ece_bounds():
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.4, 0.6, 0.3, 0.7, 0.9])
    ece = expected_calibration_error(y, p, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_horizontal_flip_swaps_symmetry():
    seq = _seq()
    li = SCHEMA.index("l_wrist")
    ri = SCHEMA.index("r_wrist")
    original_left = seq.keypoints[:, li].copy()
    flipped = horizontal_flip(seq, SCHEMA)
    # Après miroir, le poignet droit = poignet gauche d'origine avec x négé.
    assert np.allclose(flipped.keypoints[:, ri, 1], original_left[:, 1])
    assert np.allclose(flipped.keypoints[:, ri, 0], -original_left[:, 0])


def test_joint_dropout_updates_mask():
    seq = _seq()
    rng = np.random.default_rng(0)
    out = joint_dropout(seq, prob=1.0, rng=rng)  # tout tomber
    assert not out.valid_mask.any()
    assert np.allclose(out.keypoints, 0.0)


def test_augment_preserves_shapes():
    seq = _seq(T=60)
    rng = np.random.default_rng(3)
    out = augment_pose(seq, AugmentConfig(), SCHEMA, rng)
    assert out.keypoints.shape[1:] == (15, 3)
    assert out.valid_mask.shape[1] == 15
