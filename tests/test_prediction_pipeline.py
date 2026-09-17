"""Tests des briques sans interface de la prédiction Streamlit."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from kata_pipeline.gnn.pose.sequence import PoseSequence
from kata_pipeline.gnn.skeletons import get_schema
from kata_pipeline.prediction.pipeline import (
    normalize_video,
    pose_summary,
    probable_match_score,
    probe_video,
    render_web_motion_overlay,
)


def _sequence() -> PoseSequence:
    schema = get_schema("mediapipe_15")
    frames = 5
    keypoints = np.full((frames, schema.num_joints, 3), 0.5, dtype=np.float32)
    confidence = np.ones((frames, schema.num_joints), dtype=np.float32)
    confidence[2, :] = 0.1
    valid = np.ones_like(confidence, dtype=bool)
    timestamps = np.arange(frames, dtype=np.float32) / 10.0
    return PoseSequence(
        keypoints=keypoints,
        confidence=confidence,
        valid_mask=valid,
        interpolated_mask=np.zeros_like(valid),
        timestamps=timestamps,
        fps=10.0,
        joint_names=list(schema.joint_names),
    )


def test_pose_summary_reports_detection_quality():
    summary = pose_summary(_sequence(), confidence_threshold=0.3)

    assert summary["duration_seconds"] == 0.5
    assert summary["pose_frames"] == 5
    assert summary["frame_detection_rate"] == 0.8
    assert summary["joint_detection_rate"] == 0.8
    assert summary["mean_joint_confidence"] == 1.0


@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        (0.0, (3, 2)),
        (0.0999, (3, 2)),
        (0.10, (4, 1)),
        (0.30, (4, 1)),
        (0.3001, (5, 0)),
        (1.0, (5, 0)),
    ],
)
def test_probable_match_score_scale(gap, expected):
    assert probable_match_score(gap) == expected


@pytest.mark.parametrize("gap", [-0.01, 1.01])
def test_probable_match_score_rejects_invalid_gap(gap):
    with pytest.raises(ValueError):
        probable_match_score(gap)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg absent")
@pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffprobe absent")
def test_normalize_video_to_browser_mp4(tmp_path):
    source = tmp_path / "input.mkv"
    output = tmp_path / "output.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=0.5:r=24",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
    )

    metadata = normalize_video(source, output, max_width=160, output_fps=30)
    probed = probe_video(output)

    assert output.is_file()
    assert metadata.codec == "h264"
    assert metadata.width == 160
    assert metadata.height == 120
    assert probed.fps == pytest.approx(30.0)

    overlay = tmp_path / "overlay.mp4"
    render_web_motion_overlay(output, overlay, _sequence(), get_schema("mediapipe_15"))
    overlay_metadata = probe_video(overlay)
    assert overlay.is_file()
    assert overlay_metadata.codec == "h264"
    assert overlay_metadata.width == 160
