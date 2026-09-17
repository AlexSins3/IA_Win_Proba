"""Tests du split de groupe sans fuite (phase 2 / GNN)."""

from __future__ import annotations

import pandas as pd

from kata_pipeline.gnn.config import GNNConfig, PathsConfig
from kata_pipeline.gnn.data.dataset import (
    _expected_stem,
    build_extraction_manifest,
    build_training_manifest,
    split_manifest,
)
from kata_pipeline.storage.google_drive import ArchiveVideoRef, save_archive_index


def _fake_manifest() -> pd.DataFrame:
    # 6 katas, 3 matchs, 6 athlètes, 2 compétitions.
    return pd.DataFrame(
        [
            {"id_match": "m1", "athlete": "A", "competition": "C1", "issue": "win"},
            {"id_match": "m1", "athlete": "B", "competition": "C1", "issue": "loss"},
            {"id_match": "m2", "athlete": "C", "competition": "C1", "issue": "win"},
            {"id_match": "m2", "athlete": "D", "competition": "C1", "issue": "loss"},
            {"id_match": "m3", "athlete": "E", "competition": "C2", "issue": "win"},
            {"id_match": "m3", "athlete": "F", "competition": "C2", "issue": "loss"},
        ]
    )


def test_split_by_match_keeps_pairs_together() -> None:
    m = _fake_manifest()
    train, val = split_manifest(m, val_ratio=0.34, seed=1, split_by="match")
    train_matches = set(train["id_match"])
    val_matches = set(val["id_match"])
    assert train_matches.isdisjoint(val_matches)
    assert len(train) + len(val) == len(m)


def test_split_by_athlete_no_leak() -> None:
    m = _fake_manifest()
    train, val = split_manifest(m, val_ratio=0.5, seed=3, split_by="athlete")
    assert set(train["athlete"]).isdisjoint(set(val["athlete"]))


def test_split_by_competition_no_leak() -> None:
    m = _fake_manifest()
    train, val = split_manifest(m, val_ratio=0.5, seed=0, split_by="competition")
    assert set(train["competition"]).isdisjoint(set(val["competition"]))


def test_unknown_split_key_falls_back_to_match() -> None:
    m = _fake_manifest()
    train, val = split_manifest(m, val_ratio=0.34, seed=1, split_by="does_not_exist")
    assert set(train["id_match"]).isdisjoint(set(val["id_match"]))


def _clip_row(competition_type: str = "K1") -> dict[str, object]:
    return {
        "id_match": "K1_Paris_F_Pool_1_1",
        "color": "red",
        "athlete": "Athlete One",
        "competition": "K1_Paris",
        "competition_type": competition_type,
        "issue": "win",
        "flag_result": 5,
        "score": None,
        "round": "Pool_1",
        "match_order": 1,
        "category": "Female Kata",
        "kata": "Anan",
    }


def _config(tmp_path) -> GNNConfig:
    return GNNConfig(
        paths=PathsConfig(
            clips_dir=tmp_path / "clips",
            output_dir=tmp_path / "output",
            poses_dir=tmp_path / "poses",
            models_dir=tmp_path / "models",
            viz_dir=tmp_path / "viz",
            runs_dir=tmp_path / "runs",
        )
    )


def test_training_manifest_does_not_require_clips(tmp_path) -> None:
    cfg = _config(tmp_path)
    row = _clip_row()
    cfg.paths.output_dir.mkdir(parents=True)
    pd.DataFrame([row]).to_csv(cfg.paths.output_dir / "sample_clips.csv", index=False)

    stem = _expected_stem(pd.Series(row))
    pose_path = cfg.paths.poses_dir / "K1" / f"{stem}.npz"
    pose_path.parent.mkdir(parents=True)
    pose_path.write_bytes(b"pose-placeholder")

    manifest = build_training_manifest(cfg)

    assert len(manifest) == 1
    assert manifest.loc[0, "pose_path"] == str(pose_path)
    assert "clip_path" not in manifest.columns
    assert not cfg.paths.clips_dir.exists()


def test_extraction_manifest_routes_new_pose_and_excludes_pending(tmp_path) -> None:
    cfg = _config(tmp_path)
    row = _clip_row()
    cfg.paths.output_dir.mkdir(parents=True)
    pd.DataFrame([row]).to_csv(cfg.paths.output_dir / "sample_clips.csv", index=False)

    stem = _expected_stem(pd.Series(row))
    clip_path = cfg.paths.clips_dir / "K1" / "pool_1" / f"{stem}.mp4"
    clip_path.parent.mkdir(parents=True)
    clip_path.write_bytes(b"clip-placeholder")
    pending = cfg.paths.clips_dir / "K1" / "pending" / "unrelated.mp4"
    pending.parent.mkdir(parents=True)
    pending.write_bytes(b"pending-placeholder")

    manifest = build_extraction_manifest(cfg, competition_type="K1")

    assert len(manifest) == 1
    assert manifest.loc[0, "clip_path"] == str(clip_path)
    assert manifest.loc[0, "pose_path"] == str(
        cfg.paths.poses_dir / "K1" / f"{stem}.npz"
    )


def test_extraction_manifest_accepts_drive_reference_without_local_clip(tmp_path) -> None:
    cfg = _config(tmp_path)
    row = _clip_row()
    cfg.paths.output_dir.mkdir(parents=True)
    pd.DataFrame([row]).to_csv(cfg.paths.output_dir / "sample_clips.csv", index=False)
    stem = _expected_stem(pd.Series(row))
    cfg.archive.enabled = True
    cfg.archive.index_file = tmp_path / "drive-index.csv"
    cfg.archive.cache_dir = tmp_path / "cache"
    save_archive_index(
        cfg.archive.index_file,
        [
            ArchiveVideoRef(
                provider="google_drive",
                root_folder_id="root",
                drive_file_id="file-id",
                name=f"{stem}.mp4",
                stem=stem,
                relative_path=f"clips/K1/pool_1/{stem}.mp4",
                competition_type="K1",
                mime_type="video/mp4",
                size=123,
                md5_checksum="abc",
                modified_time="2026-09-03T10:00:00Z",
            )
        ],
    )

    manifest = build_extraction_manifest(cfg, competition_type="K1")

    assert len(manifest) == 1
    assert manifest.loc[0, "video_source"] == "google_drive"
    assert manifest.loc[0, "drive_file_id"] == "file-id"
    assert manifest.loc[0, "clip_path"] == str(
        cfg.archive.cache_dir / "K1" / f"{stem}.mp4"
    )
